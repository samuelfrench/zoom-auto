"""Faster Whisper STT engine implementation.

Uses faster-whisper with large-v3-turbo model and int8 quantization
for low-latency, GPU-accelerated transcription.
"""

from __future__ import annotations

import logging

from zoom_auto.config import STTConfig
from zoom_auto.stt.base import STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class FasterWhisperEngine(STTEngine):
    """Speech-to-text using faster-whisper (CTranslate2 backend).

    Optimized for real-time meeting transcription with int8 quantization.
    """

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model = None

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Args:
            audio_data: Raw 16-bit PCM audio bytes.
            sample_rate: Sample rate of the audio data.

        Returns:
            TranscriptionResult with text and timing segments.
        """
        raise NotImplementedError("Faster Whisper transcription not yet implemented")

    async def load_model(self) -> None:
        """Load the faster-whisper model into GPU memory."""
        raise NotImplementedError("Model loading not yet implemented")

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        self._model = None

    def is_loaded(self) -> bool:
        """Whether the Whisper model is loaded."""
        return self._model is not None
