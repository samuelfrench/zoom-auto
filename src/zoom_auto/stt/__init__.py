"""Speech-to-text engines for transcribing meeting audio."""

from zoom_auto.stt.base import STTEngine, TranscriptionResult
from zoom_auto.stt.faster_whisper import FasterWhisperEngine

__all__ = ["STTEngine", "TranscriptionResult", "FasterWhisperEngine"]
