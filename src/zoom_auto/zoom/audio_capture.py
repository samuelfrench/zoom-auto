"""Audio capture from Zoom meetings.

Receives per-speaker PCM audio frames from the Zoom Meeting SDK
and routes them to the audio pipeline for processing.

The SDK delivers audio at 32 kHz; we resample to the configured
target rate (default 16 kHz for STT).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from zoom_auto.config import ZoomConfig

logger = logging.getLogger(__name__)

# Zoom Meeting SDK delivers audio at 32 kHz mono 16-bit PCM
_SDK_SAMPLE_RATE = 32000
_BYTES_PER_SAMPLE = 2  # 16-bit PCM


@dataclass
class AudioFrame:
    """A single audio frame from a meeting participant.

    Attributes:
        speaker_id: Unique identifier for the speaker.
        speaker_name: Display name of the speaker.
        pcm_data: Raw 16-bit PCM audio bytes (at target sample rate).
        sample_rate: Sample rate in Hz.
        timestamp_ms: Timestamp in milliseconds since epoch.
    """

    speaker_id: int
    speaker_name: str
    pcm_data: bytes
    sample_rate: int
    timestamp_ms: int


def resample_pcm(
    pcm_data: bytes,
    source_rate: int,
    target_rate: int,
) -> bytes:
    """Resample 16-bit PCM audio from source_rate to target_rate.

    Uses linear interpolation for fast downsampling.

    Args:
        pcm_data: Raw 16-bit mono PCM bytes.
        source_rate: Source sample rate in Hz.
        target_rate: Target sample rate in Hz.

    Returns:
        Resampled 16-bit PCM bytes.
    """
    if source_rate == target_rate:
        return pcm_data

    if len(pcm_data) == 0:
        return pcm_data

    samples = np.frombuffer(pcm_data, dtype=np.int16)
    num_source = len(samples)
    num_target = int(num_source * target_rate / source_rate)

    if num_target == 0:
        return b""

    # Linear interpolation indices
    indices = np.linspace(0, num_source - 1, num_target)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, num_source - 1)
    frac = (indices - idx_floor).astype(np.float32)

    resampled = (
        samples[idx_floor].astype(np.float32) * (1.0 - frac)
        + samples[idx_ceil].astype(np.float32) * frac
    )
    return resampled.astype(np.int16).tobytes()


class AudioCapture:
    """Captures per-speaker audio from Zoom Meeting SDK.

    Provides an async iterator interface for consuming audio frames
    as they arrive from the meeting.

    Audio is received via the ``on_audio_frame`` callback (which the
    Zoom SDK adapter calls) and buffered in an asyncio queue. Consumers
    iterate with ``async for frame in capture.frames()``.

    Args:
        config: Zoom configuration (sample rate, channels, etc.).
        target_sample_rate: Target sample rate for STT (default 16 kHz).
        max_buffer_size: Maximum frames in the async queue before
            back-pressure kicks in.
    """

    def __init__(
        self,
        config: ZoomConfig | None = None,
        target_sample_rate: int = 16000,
        max_buffer_size: int = 500,
    ) -> None:
        self.config = config or ZoomConfig()
        self._target_sample_rate = target_sample_rate
        self._max_buffer_size = max_buffer_size
        self._active = False
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(
            maxsize=max_buffer_size
        )
        # Map of user_id -> display_name for speaker tracking
        self._speaker_names: dict[int, str] = {}

    async def start(self) -> None:
        """Start capturing audio from the meeting.

        After calling start(), audio frames pushed via ``on_audio_frame``
        will be queued for consumers.
        """
        if self._active:
            logger.warning("AudioCapture already active")
            return
        self._active = True
        # Clear any stale frames
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("AudioCapture started (target_rate=%d)", self._target_sample_rate)

    async def stop(self) -> None:
        """Stop capturing audio.

        Sends a sentinel (None) so that any consumers blocked on
        ``frames()`` will cleanly exit.
        """
        if not self._active:
            return
        self._active = False
        # Push sentinel to unblock consumers
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        logger.info("AudioCapture stopped")

    def on_audio_frame(
        self,
        user_id: int,
        audio_data: bytes,
        sample_rate: int = _SDK_SAMPLE_RATE,
        speaker_name: str | None = None,
    ) -> None:
        """Callback for raw audio from the Zoom SDK.

        This is called from the SDK audio thread. It resamples the
        audio if needed and pushes it onto the async queue.

        Args:
            user_id: Zoom user ID of the speaker.
            audio_data: Raw 16-bit mono PCM bytes at *sample_rate*.
            sample_rate: Sample rate of the incoming audio.
            speaker_name: Optional display name of the speaker.
        """
        if not self._active:
            return

        if speaker_name:
            self._speaker_names[user_id] = speaker_name
        name = self._speaker_names.get(user_id, f"User_{user_id}")

        # Resample to target rate
        resampled = resample_pcm(
            audio_data, sample_rate, self._target_sample_rate
        )

        frame = AudioFrame(
            speaker_id=user_id,
            speaker_name=name,
            pcm_data=resampled,
            sample_rate=self._target_sample_rate,
            timestamp_ms=int(time.time() * 1000),
        )

        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop the oldest frame to make room
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("Audio frame dropped (queue full)")

    def set_speaker_name(self, user_id: int, name: str) -> None:
        """Update the display name mapping for a speaker.

        Args:
            user_id: Zoom user ID.
            name: Display name.
        """
        self._speaker_names[user_id] = name

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield audio frames as they arrive from participants.

        Blocks until a frame is available. Stops when ``stop()``
        is called (sentinel received) or capture becomes inactive.

        Yields:
            AudioFrame objects with per-speaker PCM data.
        """
        while self._active:
            frame = await self._queue.get()
            if frame is None:
                # Sentinel — stop iteration
                break
            yield frame

    @property
    def is_active(self) -> bool:
        """Whether audio capture is currently active."""
        return self._active

    @property
    def queue_size(self) -> int:
        """Current number of frames buffered in the queue."""
        return self._queue.qsize()
