"""Tests for the Faster Whisper STT engine.

All tests mock the faster-whisper model since we cannot load it in CI.
Tests cover:
- Model loading and unloading
- PCM bytes to float32 conversion
- Transcription with segments
- Error handling (not loaded, empty audio, invalid audio)
- Confidence score conversion
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zoom_auto.config import STTConfig
from zoom_auto.stt.base import TranscriptionResult, TranscriptionSegment
from zoom_auto.stt.faster_whisper import (
    FasterWhisperEngine,
    _logprob_to_confidence,
    _pcm_bytes_to_float32,
)

# --- Fixtures ---


@pytest.fixture
def config() -> STTConfig:
    """Create a default STT config."""
    return STTConfig()


@pytest.fixture
def engine(config: STTConfig) -> FasterWhisperEngine:
    """Create a FasterWhisperEngine with default config (no model loaded)."""
    return FasterWhisperEngine(config=config, device="cpu")


@pytest.fixture
def mock_segment() -> MagicMock:
    """Create a mock faster-whisper segment."""
    seg = MagicMock()
    seg.text = " Hello, how are you doing today?"
    seg.start = 0.0
    seg.end = 2.5
    seg.avg_logprob = -0.3
    return seg


@pytest.fixture
def mock_segment_2() -> MagicMock:
    """Create a second mock faster-whisper segment."""
    seg = MagicMock()
    seg.text = " I am doing well, thank you."
    seg.start = 2.8
    seg.end = 4.5
    seg.avg_logprob = -0.25
    return seg


@pytest.fixture
def mock_info() -> MagicMock:
    """Create a mock faster-whisper TranscriptionInfo."""
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.98
    return info


@pytest.fixture
def mock_whisper_model(
    mock_segment: MagicMock,
    mock_segment_2: MagicMock,
    mock_info: MagicMock,
) -> MagicMock:
    """Create a mock WhisperModel that returns segments."""
    model = MagicMock()
    model.transcribe.return_value = (
        iter([mock_segment, mock_segment_2]),
        mock_info,
    )
    return model


@pytest.fixture
def loaded_engine(
    engine: FasterWhisperEngine,
    mock_whisper_model: MagicMock,
) -> FasterWhisperEngine:
    """Create an engine with a pre-loaded mock model."""
    engine._model = mock_whisper_model
    return engine


@pytest.fixture
def sample_pcm_audio() -> bytes:
    """Generate 1 second of 16-bit PCM silence at 16kHz."""
    samples = np.zeros(16000, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def sample_pcm_sine() -> bytes:
    """Generate 1 second of 440Hz sine wave as 16-bit PCM at 16kHz."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return audio.tobytes()


# --- Constructor and Properties ---


class TestFasterWhisperInit:
    """Tests for FasterWhisperEngine initialization."""

    def test_default_init(self) -> None:
        """Default constructor should set expected defaults."""
        engine = FasterWhisperEngine()
        assert engine._device == "cuda"
        assert engine._config.model == "large-v3-turbo"
        assert engine._config.compute_type == "int8"
        assert engine._config.beam_size == 5
        assert engine._config.language == "en"
        assert not engine.is_loaded()

    def test_custom_config(self) -> None:
        """Custom config should be stored correctly."""
        config = STTConfig(model="base", compute_type="float16", beam_size=3)
        engine = FasterWhisperEngine(config=config, device="cpu")
        assert engine._config.model == "base"
        assert engine._config.compute_type == "float16"
        assert engine._config.beam_size == 3
        assert engine._device == "cpu"

    def test_is_loaded_false_initially(self, engine: FasterWhisperEngine) -> None:
        """Engine should not be loaded initially."""
        assert engine.is_loaded() is False


# --- Model Loading ---


class TestModelLoading:
    """Tests for model load and unload."""

    @pytest.mark.asyncio
    async def test_load_model(self, engine: FasterWhisperEngine) -> None:
        """Loading model should set _model."""
        mock_model = MagicMock()
        mock_fw_module = MagicMock()
        mock_fw_module.WhisperModel.return_value = mock_model

        with patch.dict(
            "sys.modules", {"faster_whisper": mock_fw_module}
        ):
            await engine.load_model()

        assert engine.is_loaded()
        assert engine._model is mock_model

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(
        self, loaded_engine: FasterWhisperEngine
    ) -> None:
        """Loading an already-loaded model should be a no-op."""
        original_model = loaded_engine._model
        await loaded_engine.load_model()
        assert loaded_engine._model is original_model

    @pytest.mark.asyncio
    async def test_unload_model(self, loaded_engine: FasterWhisperEngine) -> None:
        """Unloading should clear the model."""
        assert loaded_engine.is_loaded()
        await loaded_engine.unload_model()
        assert loaded_engine._model is None
        assert not loaded_engine.is_loaded()

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(
        self, engine: FasterWhisperEngine
    ) -> None:
        """Unloading when not loaded should be a no-op."""
        await engine.unload_model()
        assert not engine.is_loaded()


# --- Transcription ---


class TestTranscription:
    """Tests for audio transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_not_loaded_raises(
        self, engine: FasterWhisperEngine, sample_pcm_audio: bytes
    ) -> None:
        """Transcribing without loading should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not loaded"):
            await engine.transcribe(sample_pcm_audio)

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_raises(
        self, loaded_engine: FasterWhisperEngine
    ) -> None:
        """Transcribing empty audio should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await loaded_engine.transcribe(b"")

    @pytest.mark.asyncio
    async def test_transcribe_odd_bytes_raises(
        self, loaded_engine: FasterWhisperEngine
    ) -> None:
        """Transcribing audio with odd byte count should raise ValueError."""
        with pytest.raises(ValueError, match="odd byte count"):
            await loaded_engine.transcribe(b"\x00\x01\x02")

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(
        self, loaded_engine: FasterWhisperEngine, sample_pcm_audio: bytes
    ) -> None:
        """Transcribe should return a TranscriptionResult."""
        result = await loaded_engine.transcribe(sample_pcm_audio)

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert result.duration_seconds == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_transcribe_has_segments(
        self, loaded_engine: FasterWhisperEngine, sample_pcm_audio: bytes
    ) -> None:
        """Transcription should include segments with timestamps."""
        result = await loaded_engine.transcribe(sample_pcm_audio)

        assert len(result.segments) == 2
        seg1 = result.segments[0]
        assert isinstance(seg1, TranscriptionSegment)
        assert seg1.text == "Hello, how are you doing today?"
        assert seg1.start == 0.0
        assert seg1.end == 2.5
        assert 0.0 <= seg1.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_transcribe_full_text(
        self, loaded_engine: FasterWhisperEngine, sample_pcm_audio: bytes
    ) -> None:
        """Full text should join all segment texts."""
        result = await loaded_engine.transcribe(sample_pcm_audio)

        assert "Hello, how are you doing today?" in result.text
        assert "I am doing well, thank you." in result.text

    @pytest.mark.asyncio
    async def test_transcribe_confidence(
        self, loaded_engine: FasterWhisperEngine, sample_pcm_audio: bytes
    ) -> None:
        """Overall confidence should be average of segment confidences."""
        result = await loaded_engine.transcribe(sample_pcm_audio)

        expected_conf1 = math.exp(-0.3)
        expected_conf2 = math.exp(-0.25)
        expected_avg = (expected_conf1 + expected_conf2) / 2
        assert result.confidence == pytest.approx(expected_avg, abs=0.01)

    @pytest.mark.asyncio
    async def test_transcribe_passes_config_to_model(
        self,
        loaded_engine: FasterWhisperEngine,
        mock_whisper_model: MagicMock,
        sample_pcm_audio: bytes,
    ) -> None:
        """Transcribe should pass beam_size, language, vad_filter to model."""
        await loaded_engine.transcribe(sample_pcm_audio)

        mock_whisper_model.transcribe.assert_called_once()
        call_kwargs = mock_whisper_model.transcribe.call_args.kwargs
        assert call_kwargs["beam_size"] == 5
        assert call_kwargs["language"] == "en"
        assert call_kwargs["vad_filter"] is True

    @pytest.mark.asyncio
    async def test_transcribe_with_custom_sample_rate(
        self, loaded_engine: FasterWhisperEngine
    ) -> None:
        """Transcribe should compute duration from sample rate."""
        # 0.5 seconds at 8kHz = 4000 samples = 8000 bytes
        audio = np.zeros(4000, dtype=np.int16).tobytes()
        result = await loaded_engine.transcribe(audio, sample_rate=8000)

        assert result.duration_seconds == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_transcribe_empty_segments(
        self,
        engine: FasterWhisperEngine,
        mock_info: MagicMock,
    ) -> None:
        """Transcribing audio with no speech should return empty result."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)
        engine._model = mock_model

        audio = np.zeros(16000, dtype=np.int16).tobytes()
        result = await engine.transcribe(audio)

        assert result.text == ""
        assert len(result.segments) == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_whitespace_only_segments_skipped(
        self,
        engine: FasterWhisperEngine,
        mock_info: MagicMock,
    ) -> None:
        """Segments with only whitespace should be skipped."""
        empty_seg = MagicMock()
        empty_seg.text = "   "
        empty_seg.start = 0.0
        empty_seg.end = 0.5
        empty_seg.avg_logprob = -0.5

        real_seg = MagicMock()
        real_seg.text = " Actual text"
        real_seg.start = 0.5
        real_seg.end = 1.5
        real_seg.avg_logprob = -0.2

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([empty_seg, real_seg]),
            mock_info,
        )
        engine._model = mock_model

        audio = np.zeros(16000, dtype=np.int16).tobytes()
        result = await engine.transcribe(audio)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Actual text"


# --- PCM Conversion ---


class TestPCMConversion:
    """Tests for PCM bytes to float32 conversion."""

    def test_silence(self) -> None:
        """Silence (zeros) should convert to zeros."""
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        result = _pcm_bytes_to_float32(pcm)
        np.testing.assert_array_equal(result, np.zeros(100, dtype=np.float32))

    def test_max_positive(self) -> None:
        """Max positive int16 should map to ~1.0."""
        pcm = np.array([32767], dtype=np.int16).tobytes()
        result = _pcm_bytes_to_float32(pcm)
        assert result[0] == pytest.approx(32767 / 32768.0, abs=1e-6)

    def test_max_negative(self) -> None:
        """Min negative int16 should map to -1.0."""
        pcm = np.array([-32768], dtype=np.int16).tobytes()
        result = _pcm_bytes_to_float32(pcm)
        assert result[0] == pytest.approx(-1.0, abs=1e-6)

    def test_shape_preserved(self) -> None:
        """Output array should have same number of samples as input."""
        pcm = np.zeros(500, dtype=np.int16).tobytes()
        result = _pcm_bytes_to_float32(pcm)
        assert result.shape == (500,)
        assert result.dtype == np.float32

    def test_known_values(self) -> None:
        """Known int16 values should convert correctly."""
        values = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        result = _pcm_bytes_to_float32(values.tobytes())

        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5, abs=0.001)
        assert result[2] == pytest.approx(-0.5, abs=0.001)


# --- Confidence Conversion ---


class TestConfidenceConversion:
    """Tests for log probability to confidence conversion."""

    def test_zero_logprob(self) -> None:
        """Log prob of 0 should give confidence of 1.0."""
        assert _logprob_to_confidence(0.0) == pytest.approx(1.0)

    def test_negative_logprob(self) -> None:
        """Negative log prob should give confidence < 1.0."""
        conf = _logprob_to_confidence(-0.5)
        assert 0.0 < conf < 1.0
        assert conf == pytest.approx(math.exp(-0.5))

    def test_very_negative_logprob(self) -> None:
        """Very negative log prob should give confidence near 0."""
        conf = _logprob_to_confidence(-10.0)
        assert conf < 0.001

    def test_positive_logprob_clamped(self) -> None:
        """Positive log prob should be clamped to 1.0."""
        conf = _logprob_to_confidence(1.0)
        assert conf == 1.0

    def test_extremely_negative_logprob(self) -> None:
        """Extremely negative log prob should clamp to 0.0."""
        conf = _logprob_to_confidence(-1000.0)
        assert conf == 0.0
