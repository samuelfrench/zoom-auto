"""Audio frame routing between components.

Routes audio frames from Zoom capture through VAD, STT, and back
through TTS to the Zoom audio sender.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from zoom_auto.config import Settings
from zoom_auto.pipeline.vad import VADProcessor
from zoom_auto.stt.base import STTEngine
from zoom_auto.tts.base import TTSEngine
from zoom_auto.zoom.audio_capture import AudioCapture
from zoom_auto.zoom.audio_sender import AudioSender

logger = logging.getLogger(__name__)

# Type alias for the transcript callback
TranscriptCallback = Callable[[str, str], Coroutine[Any, Any, None]]


class AudioPipeline:
    """Routes audio frames between Zoom SDK and processing components.

    Flow: Zoom Capture -> VAD -> STT -> [LLM] -> TTS -> Zoom Send

    The pipeline handles buffering, format conversion, and async
    coordination between components.
    """

    def __init__(
        self,
        settings: Settings,
        capture: AudioCapture,
        sender: AudioSender,
        vad: VADProcessor,
        stt: STTEngine,
        tts: TTSEngine,
    ) -> None:
        self.settings = settings
        self.capture = capture
        self.sender = sender
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self._running = False
        self._bot_speaking = False
        self._capture_task: asyncio.Task[None] | None = None
        self._tts_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        # Callback invoked when a complete utterance is transcribed.
        # Signature: async callback(speaker_name: str, text: str) -> None
        self._on_transcript: TranscriptCallback | None = None

        # Voice sample path for TTS cloning
        self._voice_sample: Path | None = None
        voice_dir = Path(settings.tts.voice_sample_dir)
        if voice_dir.is_dir():
            # Pick first .wav file found
            wavs = sorted(voice_dir.glob("*.wav"))
            if wavs:
                self._voice_sample = wavs[0]
                logger.info("Using voice sample: %s", self._voice_sample)

    def set_transcript_callback(self, callback: TranscriptCallback) -> None:
        """Register a callback for completed transcriptions.

        Args:
            callback: Async function(speaker_name, text) called when
                      a full utterance is transcribed.
        """
        self._on_transcript = callback

    async def start(self) -> None:
        """Start the audio pipeline processing loop."""
        if self._running:
            logger.warning("AudioPipeline already running")
            return

        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())
        logger.info("AudioPipeline started")

    async def stop(self) -> None:
        """Stop the audio pipeline."""
        if not self._running:
            return

        self._running = False

        # Cancel the capture loop
        if self._capture_task is not None:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        # Cancel any in-progress TTS send
        await self.stop_speaking()

        logger.info("AudioPipeline stopped")

    async def send_response(self, text: str) -> None:
        """Synthesize text to speech and send to the meeting.

        Args:
            text: The response text to speak.
        """
        if not text.strip():
            return

        async with self._lock:
            self._bot_speaking = True

        try:
            logger.info("Synthesizing response: %s", text[:80])
            result = await self.tts.synthesize(text, self._voice_sample)

            # Check if we were interrupted during synthesis
            if not self._bot_speaking:
                logger.info("Interrupted during TTS synthesis, discarding")
                return

            logger.info(
                "Sending TTS audio (%.1fs, %d bytes)",
                result.duration_seconds,
                len(result.audio_data),
            )
            await self.sender.send_audio(
                result.audio_data, result.sample_rate
            )
        except asyncio.CancelledError:
            logger.info("TTS send cancelled (interruption)")
            raise
        except Exception:
            logger.exception("Error in send_response")
        finally:
            async with self._lock:
                self._bot_speaking = False

    async def stop_speaking(self) -> None:
        """Stop any in-progress TTS playback (interruption support)."""
        async with self._lock:
            self._bot_speaking = False

        if self._tts_task is not None:
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
            self._tts_task = None

        # Drain the sender queue to stop audio output
        while not self.sender._queue.empty():
            try:
                self.sender._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.debug("Stopped speaking (TTS interrupted)")

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return self._running

    @property
    def is_bot_speaking(self) -> bool:
        """Whether the bot is currently sending TTS audio."""
        return self._bot_speaking

    async def _capture_loop(self) -> None:
        """Main capture loop: consume frames -> VAD -> STT -> callback."""
        logger.debug("Capture loop started")
        # Track the last speaker for each VAD segment
        current_speaker = "Unknown"

        try:
            async for frame in self.capture.frames():
                if not self._running:
                    break

                current_speaker = frame.speaker_name

                # Feed audio to VAD
                event = await self.vad.process_chunk(frame.pcm_data)

                if event is None:
                    continue

                if event.is_speech_start:
                    logger.debug(
                        "Speech start detected from %s",
                        frame.speaker_name,
                    )

                if event.is_speech_end and event.audio_buffer:
                    # Complete utterance -- transcribe it
                    logger.debug(
                        "Speech end -- transcribing %d bytes from %s",
                        len(event.audio_buffer),
                        current_speaker,
                    )
                    await self._handle_speech_segment(
                        current_speaker,
                        event.audio_buffer,
                        frame.sample_rate,
                    )
        except asyncio.CancelledError:
            logger.debug("Capture loop cancelled")
        except Exception:
            logger.exception("Error in capture loop")
        finally:
            logger.debug("Capture loop exited")

    async def _handle_speech_segment(
        self,
        speaker: str,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> None:
        """Transcribe a complete speech segment and invoke callback.

        Args:
            speaker: The speaker's display name.
            audio_buffer: Complete speech audio bytes.
            sample_rate: Audio sample rate.
        """
        try:
            result = await self.stt.transcribe(audio_buffer, sample_rate)

            if not result.text.strip():
                logger.debug("Empty transcription, skipping")
                return

            logger.info(
                "Transcribed [%s]: %s (confidence=%.2f)",
                speaker,
                result.text[:80],
                result.confidence,
            )

            if self._on_transcript is not None:
                await self._on_transcript(speaker, result.text)
        except Exception:
            logger.exception("Error transcribing speech segment")
