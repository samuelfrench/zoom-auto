"""Silero VAD integration for voice activity detection.

Uses Silero VAD to detect speech boundaries in audio streams,
enabling accurate segmentation for STT processing.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from zoom_auto.config import VADConfig

logger = logging.getLogger(__name__)

# Thread pool for blocking model operations
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad")

# Silero VAD requires specific chunk sizes at 16kHz:
# 512, 1024, or 1536 samples per chunk.
_SILERO_CHUNK_SAMPLES = 512
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit PCM


@dataclass
class VADEvent:
    """Event emitted by VAD on speech state transitions.

    Attributes:
        is_speech_start: True when speech begins.
        is_speech_end: True when speech ends.
        audio_buffer: Complete speech audio bytes (only on speech_end).
    """

    is_speech_start: bool = False
    is_speech_end: bool = False
    audio_buffer: bytes | None = None


@dataclass
class VADResult:
    """Result from voice activity detection.

    Attributes:
        is_speech: Whether speech was detected.
        confidence: Confidence score (0-1).
        speech_start: Start of speech segment (seconds), if detected.
        speech_end: End of speech segment (seconds), if detected.
    """

    is_speech: bool
    confidence: float
    speech_start: float | None = None
    speech_end: float | None = None


class VADProcessor:
    """Voice activity detection using Silero VAD.

    Processes audio frames to detect speech boundaries,
    buffering silence and speech segments appropriately.

    Implements a state machine:
    - IDLE: Not speaking. On speech detected -> SPEAKING.
    - SPEAKING: Buffering audio. On silence detected for
      min_silence_duration -> emit speech_end with buffer -> IDLE.

    Args:
        config: VAD configuration (threshold, min speech/silence durations).
        sample_rate: Audio sample rate (must be 16000 for Silero VAD).
    """

    def __init__(
        self,
        config: VADConfig | None = None,
        sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        self._config = config or VADConfig()
        self._sample_rate = sample_rate
        self._model = None
        self._model_utils = None  # (get_speech_timestamps, etc.)

        # State machine
        self._in_speech = False
        self._audio_buffer = bytearray()

        # Timing counters (in samples)
        self._speech_samples = 0
        self._silence_samples = 0
        self._total_samples_processed = 0

        # Pre-compute sample thresholds from config durations
        self._min_speech_samples = int(
            self._config.min_speech_duration * self._sample_rate
        )
        self._min_silence_samples = int(
            self._config.min_silence_duration * self._sample_rate
        )

        # Internal chunk buffer for reframing to Silero's expected size
        self._chunk_buffer = bytearray()

    async def load_model(self) -> None:
        """Load the Silero VAD model.

        Downloads from torch.hub on first use, then loads from cache.
        """
        if self._model is not None:
            logger.debug("Silero VAD model already loaded, skipping")
            return

        loop = asyncio.get_running_loop()

        def _load() -> tuple:
            import torch

            logger.info("Loading Silero VAD model")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            model.eval()
            logger.info("Silero VAD model loaded successfully")
            return model, utils

        self._model, self._model_utils = await loop.run_in_executor(
            _executor, _load
        )

    async def unload_model(self) -> None:
        """Unload the VAD model from memory."""
        if self._model is None:
            return

        logger.info("Unloading Silero VAD model")
        self._model = None
        self._model_utils = None
        self.reset()

    def is_loaded(self) -> bool:
        """Whether the VAD model is loaded."""
        return self._model is not None

    def reset(self) -> None:
        """Reset VAD state for a new audio stream."""
        self._in_speech = False
        self._audio_buffer.clear()
        self._chunk_buffer.clear()
        self._speech_samples = 0
        self._silence_samples = 0
        self._total_samples_processed = 0

        # Reset the model's internal state if loaded
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass

    async def process_frame(
        self,
        pcm_data: bytes,
        sample_rate: int = _SAMPLE_RATE,
    ) -> VADResult:
        """Process a single audio frame for voice activity.

        This is a simpler interface that returns a VADResult with
        speech presence info. For utterance-level segmentation,
        use process_chunk() which returns VADEvents on transitions.

        Args:
            pcm_data: Raw 16-bit PCM audio frame.
            sample_rate: Sample rate of the audio.

        Returns:
            VADResult indicating speech presence and boundaries.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "Silero VAD model not loaded. Call load_model() first."
            )

        confidence = await self._get_speech_prob(pcm_data)
        is_speech = confidence >= self._config.threshold

        current_time = self._total_samples_processed / self._sample_rate

        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            speech_start=current_time if is_speech and not self._in_speech else None,
            speech_end=current_time if not is_speech and self._in_speech else None,
        )

    async def process_chunk(self, audio_chunk: bytes) -> VADEvent | None:
        """Process an audio chunk and detect speech boundaries.

        Implements a state machine that tracks speech start/end events.
        Audio is buffered during speech and flushed as a complete
        utterance when silence is detected.

        The chunk is internally reframed to Silero's required 512-sample
        blocks, so callers can pass any chunk size.

        Args:
            audio_chunk: Raw 16-bit PCM audio bytes (mono, 16kHz).

        Returns:
            VADEvent on state transitions (speech start or end),
            None if no transition occurred.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "Silero VAD model not loaded. Call load_model() first."
            )

        # Add incoming data to our reframing buffer
        self._chunk_buffer.extend(audio_chunk)

        silero_chunk_bytes = _SILERO_CHUNK_SAMPLES * _BYTES_PER_SAMPLE
        event: VADEvent | None = None

        # Process all complete Silero-sized chunks
        while len(self._chunk_buffer) >= silero_chunk_bytes:
            chunk = bytes(self._chunk_buffer[:silero_chunk_bytes])
            del self._chunk_buffer[:silero_chunk_bytes]

            result = await self._process_silero_chunk(chunk)
            if result is not None:
                # Return the most recent event (speech_end takes priority
                # since it carries the buffer)
                event = result

        return event

    async def get_speech_segment(self) -> bytes | None:
        """Get a complete speech segment if one has been detected.

        Returns:
            Complete speech audio bytes, or None if no segment is ready.
        """
        # This is handled via process_chunk returning VADEvent with
        # audio_buffer on speech_end. But if we're currently in speech
        # and have buffered data, we can force-flush.
        if self._in_speech and len(self._audio_buffer) > 0:
            segment = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            self._in_speech = False
            self._speech_samples = 0
            self._silence_samples = 0
            return segment
        return None

    async def _process_silero_chunk(self, chunk: bytes) -> VADEvent | None:
        """Process a single Silero-sized chunk (512 samples).

        Args:
            chunk: Exactly 512 samples of 16-bit PCM bytes.

        Returns:
            VADEvent on state transition, None otherwise.
        """
        num_samples = len(chunk) // _BYTES_PER_SAMPLE
        self._total_samples_processed += num_samples

        confidence = await self._get_speech_prob(chunk)
        is_speech = confidence >= self._config.threshold

        if not self._in_speech:
            # Currently in silence
            if is_speech:
                self._speech_samples += num_samples
                # Buffer audio from potential speech start
                self._audio_buffer.extend(chunk)

                if self._speech_samples >= self._min_speech_samples:
                    # Confirmed speech start
                    self._in_speech = True
                    self._silence_samples = 0
                    logger.debug(
                        "Speech start at %.2fs (confidence=%.3f)",
                        self._total_samples_processed / self._sample_rate,
                        confidence,
                    )
                    return VADEvent(is_speech_start=True)
            else:
                # Still silence — discard any tentative speech buffer
                if self._speech_samples > 0:
                    self._speech_samples = 0
                    self._audio_buffer.clear()
        else:
            # Currently in speech
            self._audio_buffer.extend(chunk)

            if not is_speech:
                self._silence_samples += num_samples

                if self._silence_samples >= self._min_silence_samples:
                    # Confirmed speech end — flush buffer
                    speech_audio = bytes(self._audio_buffer)
                    self._audio_buffer.clear()
                    self._in_speech = False
                    self._speech_samples = 0
                    self._silence_samples = 0
                    logger.debug(
                        "Speech end at %.2fs (buffer=%d bytes)",
                        self._total_samples_processed / self._sample_rate,
                        len(speech_audio),
                    )
                    return VADEvent(
                        is_speech_end=True,
                        audio_buffer=speech_audio,
                    )
            else:
                # Still speaking — reset silence counter
                self._silence_samples = 0

        return None

    async def _get_speech_prob(self, pcm_chunk: bytes) -> float:
        """Run the Silero VAD model on a PCM chunk and return speech probability.

        Converts PCM bytes to a torch tensor and runs inference.
        Uses run_in_executor for the model forward pass.

        Args:
            pcm_chunk: Raw 16-bit PCM bytes.

        Returns:
            Speech probability between 0 and 1.
        """
        import torch

        # Convert 16-bit PCM to float32 tensor
        audio_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)

        loop = asyncio.get_running_loop()

        def _infer() -> float:
            with torch.no_grad():
                prob = self._model(tensor, self._sample_rate)
                if isinstance(prob, torch.Tensor):
                    return prob.item()
                return float(prob)

        return await loop.run_in_executor(_executor, _infer)
