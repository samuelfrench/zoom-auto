"""Audio sender for Zoom meetings.

Sends TTS-generated PCM audio back into the Zoom meeting
so other participants can hear the bot's responses.

Audio from TTS (typically 22 kHz or 16 kHz) is resampled to the
Zoom SDK's expected rate (32 kHz) and fed in frame-sized chunks.
"""

from __future__ import annotations

import asyncio
import logging

from zoom_auto.config import ZoomConfig
from zoom_auto.zoom.audio_capture import resample_pcm

logger = logging.getLogger(__name__)

# Zoom SDK expects 32 kHz mono 16-bit PCM
_SDK_SAMPLE_RATE = 32000
_BYTES_PER_SAMPLE = 2

# Default frame duration sent to the SDK (20 ms)
_FRAME_DURATION_MS = 20
_FRAME_SAMPLES = _SDK_SAMPLE_RATE * _FRAME_DURATION_MS // 1000  # 640
_FRAME_BYTES = _FRAME_SAMPLES * _BYTES_PER_SAMPLE  # 1280


class AudioSender:
    """Sends PCM audio data to a Zoom meeting.

    Accepts audio frames (from TTS) and feeds them into the Zoom
    Meeting SDK's audio output channel.  Audio is queued internally
    and sent at the correct pacing (one 20 ms frame every 20 ms)
    to avoid overwhelming the SDK.

    Args:
        config: Zoom configuration.
        sdk_sample_rate: Sample rate expected by the SDK (default 32 kHz).
        send_callback: Optional function to call for each frame of audio
            to send.  Signature: ``(pcm_bytes: bytes) -> None``.
            If not provided, frames are logged but not sent (useful for
            testing without a live SDK).
    """

    def __init__(
        self,
        config: ZoomConfig | None = None,
        sdk_sample_rate: int = _SDK_SAMPLE_RATE,
        send_callback: object | None = None,
    ) -> None:
        self.config = config or ZoomConfig()
        self._sdk_sample_rate = sdk_sample_rate
        self._send_callback = send_callback
        self._active = False
        self._sending = False
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._send_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        """Initialise the audio sender and start the send loop.

        The send loop drains queued audio and paces it at the
        correct frame rate.
        """
        if self._active:
            logger.warning("AudioSender already active")
            return
        self._active = True
        self._send_task = asyncio.create_task(self._send_loop())
        logger.info("AudioSender started (sdk_rate=%d)", self._sdk_sample_rate)

    async def stop(self) -> None:
        """Stop sending audio and cancel the send loop."""
        if not self._active:
            return
        self._active = False
        # Push sentinel to unblock the send loop
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._send_task is not None:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None
        self._sending = False
        logger.info("AudioSender stopped")

    async def send_frame(
        self, pcm_data: bytes, sample_rate: int = 16000
    ) -> None:
        """Send a single PCM audio frame to the meeting.

        The frame is resampled to the SDK rate if necessary and
        queued for sending.

        Args:
            pcm_data: Raw 16-bit PCM audio bytes.
            sample_rate: Sample rate of the audio data.
        """
        if not self._active:
            logger.warning("send_frame called but sender not active")
            return

        resampled = resample_pcm(
            pcm_data, sample_rate, self._sdk_sample_rate
        )
        await self._queue.put(resampled)

    async def send_audio(
        self, pcm_data: bytes, sample_rate: int = 16000
    ) -> None:
        """Send a complete audio buffer to the meeting, chunked into frames.

        The buffer is resampled to the SDK rate, split into
        20 ms frames, and queued for paced sending.

        Args:
            pcm_data: Raw 16-bit PCM audio bytes (complete utterance).
            sample_rate: Sample rate of the audio data.
        """
        if not self._active:
            logger.warning("send_audio called but sender not active")
            return

        if len(pcm_data) == 0:
            return

        # Resample the entire buffer
        resampled = resample_pcm(
            pcm_data, sample_rate, self._sdk_sample_rate
        )

        # Chunk into SDK-sized frames
        frame_bytes = (
            self._sdk_sample_rate
            * _FRAME_DURATION_MS
            // 1000
            * _BYTES_PER_SAMPLE
        )

        offset = 0
        while offset < len(resampled):
            chunk = resampled[offset : offset + frame_bytes]
            # Pad the last chunk with silence if needed
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            await self._queue.put(chunk)
            offset += frame_bytes

    @property
    def is_active(self) -> bool:
        """Whether the audio sender is active."""
        return self._active

    @property
    def is_sending(self) -> bool:
        """Whether audio is currently being sent (queue non-empty)."""
        return self._sending

    @property
    def queue_size(self) -> int:
        """Number of frames waiting to be sent."""
        return self._queue.qsize()

    # ------------------------------------------------------------------ #
    #  Internal send loop                                                 #
    # ------------------------------------------------------------------ #

    async def _send_loop(self) -> None:
        """Drain the queue and send frames at the correct pacing."""
        frame_interval = _FRAME_DURATION_MS / 1000.0

        while self._active:
            try:
                chunk = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except TimeoutError:
                self._sending = False
                continue

            if chunk is None:
                # Sentinel — stop
                self._sending = False
                break

            self._sending = True
            self._deliver_frame(chunk)

            # Pace at frame rate so we don't flood the SDK
            await asyncio.sleep(frame_interval)

            if self._queue.empty():
                self._sending = False

    def _deliver_frame(self, pcm_data: bytes) -> None:
        """Deliver a single frame via the SDK or callback.

        Args:
            pcm_data: A single frame of PCM audio at the SDK rate.
        """
        if self._send_callback is not None:
            try:
                self._send_callback(pcm_data)  # type: ignore[operator]
            except Exception:
                logger.exception("Error in send_callback")
        else:
            logger.debug(
                "Audio frame ready (%d bytes), no send_callback set",
                len(pcm_data),
            )
