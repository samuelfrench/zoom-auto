"""Abstract base class for speech-to-text engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech with timing info.

    Attributes:
        text: The transcribed text for this segment.
        start: Start time in seconds.
        end: End time in seconds.
        confidence: Confidence score for this segment (0-1).
    """

    text: str
    start: float
    end: float
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Result from a speech-to-text transcription.

    Attributes:
        text: The transcribed text.
        language: Detected or specified language code.
        confidence: Overall confidence score (0-1).
        segments: Individual word/phrase segments with timestamps.
        duration_seconds: Duration of the transcribed audio.
    """

    text: str
    language: str = "en"
    confidence: float = 0.0
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration_seconds: float = 0.0


class STTEngine(ABC):
    """Abstract base class for speech-to-text engines.

    Implementations must provide methods for transcribing audio data
    from PCM bytes or numpy arrays.
    """

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw 16-bit PCM audio bytes.
            sample_rate: Sample rate of the audio data.

        Returns:
            TranscriptionResult with text, confidence, and segments.
        """
        ...

    @abstractmethod
    async def load_model(self) -> None:
        """Load the STT model into memory (may download if needed)."""
        ...

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the STT model from memory."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded and ready."""
        ...
