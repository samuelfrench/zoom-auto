"""Abstract base class for text-to-speech engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSResult:
    """Result from a text-to-speech synthesis.

    Attributes:
        audio_data: Raw PCM audio bytes (16-bit).
        sample_rate: Sample rate of the generated audio.
        duration_seconds: Duration of the generated audio.
    """

    audio_data: bytes
    sample_rate: int
    duration_seconds: float


class TTSEngine(ABC):
    """Abstract base class for text-to-speech engines.

    Implementations must support voice cloning from reference samples
    and both batch and streaming synthesis.
    """

    @abstractmethod
    async def synthesize(self, text: str, voice_sample: Path | None = None) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: The text to convert to speech.
            voice_sample: Optional path to a voice reference sample for cloning.

        Returns:
            TTSResult with PCM audio data.
        """
        ...

    @abstractmethod
    async def synthesize_stream(
        self, text: str, voice_sample: Path | None = None
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech as PCM audio chunks.

        Args:
            text: The text to convert to speech.
            voice_sample: Optional path to a voice reference sample for cloning.

        Yields:
            PCM audio data chunks.
        """
        ...
        yield b""  # pragma: no cover

    @abstractmethod
    async def load_model(self) -> None:
        """Load the TTS model into memory."""
        ...

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the TTS model from memory."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        ...
