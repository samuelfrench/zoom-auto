"""Faster Whisper STT engine implementation.

Uses faster-whisper with large-v3-turbo model and int8 quantization
for low-latency, GPU-accelerated transcription.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from zoom_auto.config import STTConfig
from zoom_auto.stt.base import STTEngine, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

# Thread pool for blocking model operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")


class FasterWhisperEngine(STTEngine):
    """Speech-to-text using faster-whisper (CTranslate2 backend).

    Optimized for real-time meeting transcription with int8 quantization.
    Uses ~6GB VRAM on RTX 4090, leaving room for TTS model.

    Args:
        config: STT configuration (model size, compute type, language, beam size).
        device: Device to run on ("cuda" or "cpu").
    """

    def __init__(
        self,
        config: STTConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self._config = config or STTConfig()
        self._device = device
        self._model = None  # WhisperModel instance

    async def load_model(self) -> None:
        """Load the faster-whisper model into GPU memory.

        Downloads the model on first use, then loads from cache.
        Runs in a thread pool since model loading is blocking.
        """
        if self._model is not None:
            logger.debug("Faster-whisper model already loaded, skipping")
            return

        loop = asyncio.get_running_loop()

        def _load() -> object:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                self._config.model,
                self._device,
                self._config.compute_type,
            )
            model = WhisperModel(
                self._config.model,
                device=self._device,
                compute_type=self._config.compute_type,
            )
            logger.info("Faster-whisper model loaded successfully")
            return model

        self._model = await loop.run_in_executor(_executor, _load)

    async def unload_model(self) -> None:
        """Unload the model from memory and free GPU resources."""
        if self._model is None:
            return

        logger.info("Unloading faster-whisper model")
        self._model = None

        # Attempt to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        """Whether the Whisper model is loaded."""
        return self._model is not None

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Converts raw 16-bit PCM bytes to a float32 numpy array, then runs
        faster-whisper transcription in a thread pool to avoid blocking
        the event loop.

        Args:
            audio_data: Raw 16-bit PCM audio bytes (little-endian, mono).
            sample_rate: Sample rate of the audio data.

        Returns:
            TranscriptionResult with text, confidence, language, and
            timestamped segments.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If audio_data is empty or has invalid length.
        """
        if self._model is None:
            raise RuntimeError(
                "Faster-whisper model not loaded. Call load_model() first."
            )

        if not audio_data:
            raise ValueError("audio_data is empty")

        if len(audio_data) % 2 != 0:
            raise ValueError(
                f"audio_data has odd byte count ({len(audio_data)}); "
                "expected 16-bit PCM (2 bytes per sample)"
            )

        # Convert 16-bit PCM bytes to float32 numpy array in [-1, 1]
        audio_array = _pcm_bytes_to_float32(audio_data)
        duration_seconds = len(audio_array) / sample_rate

        loop = asyncio.get_running_loop()
        transcribe_fn = partial(
            self._transcribe_sync,
            audio_array=audio_array,
            sample_rate=sample_rate,
        )
        result = await loop.run_in_executor(_executor, transcribe_fn)

        # Attach duration
        result.duration_seconds = duration_seconds
        return result

    def _transcribe_sync(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Synchronous transcription (runs in thread pool).

        Args:
            audio_array: Float32 audio array normalized to [-1, 1].
            sample_rate: Sample rate of the audio.

        Returns:
            TranscriptionResult with segments.
        """
        segments_iter, info = self._model.transcribe(
            audio_array,
            beam_size=self._config.beam_size,
            language=self._config.language,
            vad_filter=True,
        )

        # Collect all segments
        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        total_confidence = 0.0

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            # faster-whisper provides avg_logprob; convert to probability
            confidence = _logprob_to_confidence(seg.avg_logprob)

            segments.append(
                TranscriptionSegment(
                    text=text,
                    start=seg.start,
                    end=seg.end,
                    confidence=confidence,
                )
            )
            full_text_parts.append(text)
            total_confidence += confidence

        full_text = " ".join(full_text_parts)
        avg_confidence = total_confidence / len(segments) if segments else 0.0

        return TranscriptionResult(
            text=full_text,
            language=info.language if info.language else self._config.language,
            confidence=avg_confidence,
            segments=segments,
        )


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes (little-endian) to float32 in [-1, 1].

    Args:
        pcm_bytes: Raw 16-bit signed PCM bytes.

    Returns:
        Numpy float32 array normalized to [-1.0, 1.0].
    """
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


def _logprob_to_confidence(avg_logprob: float) -> float:
    """Convert faster-whisper's avg_logprob to a 0-1 confidence score.

    Uses exponential mapping: confidence = exp(avg_logprob).
    Clamped to [0, 1] range.

    Args:
        avg_logprob: Average log probability from faster-whisper segment.

    Returns:
        Confidence score between 0 and 1.
    """
    import math

    confidence = math.exp(avg_logprob)
    return max(0.0, min(1.0, confidence))
