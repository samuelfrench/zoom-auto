"""Text-to-speech engines with voice cloning support."""

from zoom_auto.tts.audio_validation import AudioQualityReport, validate_audio_data, validate_audio_file
from zoom_auto.tts.base import TTSEngine, TTSResult
from zoom_auto.tts.chatterbox import ChatterboxEngine
from zoom_auto.tts.prompts import RECORDING_PROMPTS
from zoom_auto.tts.voice_store import SegmentMetadata, UserVoiceMetadata, VoiceProfile, VoiceStore

__all__ = [
    "AudioQualityReport",
    "ChatterboxEngine",
    "RECORDING_PROMPTS",
    "SegmentMetadata",
    "TTSEngine",
    "TTSResult",
    "UserVoiceMetadata",
    "VoiceProfile",
    "VoiceStore",
    "validate_audio_data",
    "validate_audio_file",
]
