"""Tests for the Chatterbox TTS engine.

All tests mock the chatterbox model since we cannot load it in CI.
Tests cover:
- Model loading and unloading
- Synthesis (batch and streaming)
- Voice reference loading
- Tensor to PCM conversion
- Exaggeration property
- Error handling
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from zoom_auto.tts.base import TTSResult
from zoom_auto.tts.chatterbox import ChatterboxEngine, _DEFAULT_SAMPLE_RATE


# --- Fixtures ---


@pytest.fixture
def engine() -> ChatterboxEngine:
    """Create a ChatterboxEngine with default config (no model loaded)."""
    return ChatterboxEngine(device="cpu", exaggeration=0.5)


@pytest.fixture
def mock_torch_tensor() -> torch.Tensor:
    """Create a torch tensor that mimics Chatterbox output."""
    # 1 second of audio at 24kHz
    samples = 24000
    t = np.linspace(0, 1.0, samples, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return torch.from_numpy(audio).unsqueeze(0)  # shape [1, N]


@pytest.fixture
def mock_chatterbox_model(mock_torch_tensor: torch.Tensor) -> MagicMock:
    """Create a mock ChatterboxTTS model."""
    model = MagicMock()
    model.sr = 24000
    model.generate.return_value = mock_torch_tensor
    return model


@pytest.fixture
def loaded_engine(engine: ChatterboxEngine, mock_chatterbox_model: MagicMock) -> ChatterboxEngine:
    """Create an engine with a pre-loaded mock model."""
    engine._model = mock_chatterbox_model
    engine._sample_rate = 24000
    return engine


@pytest.fixture
def voice_ref_wav(tmp_path: Path) -> Path:
    """Create a temporary voice reference WAV file."""
    path = tmp_path / "reference.wav"
    duration = 3.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


# --- Constructor and Properties ---


class TestChatterboxEngineInit:
    """Tests for ChatterboxEngine initialization and properties."""

    def test_default_init(self) -> None:
        """Default constructor should set expected defaults."""
        engine = ChatterboxEngine()
        assert engine._device == "cuda"
        assert engine.exaggeration == 0.5
        assert engine.sample_rate == _DEFAULT_SAMPLE_RATE
        assert not engine.is_loaded()

    def test_custom_init(self) -> None:
        """Custom parameters should be stored correctly."""
        engine = ChatterboxEngine(device="cpu", exaggeration=0.8)
        assert engine._device == "cpu"
        assert engine.exaggeration == 0.8

    def test_exaggeration_clamping_high(self) -> None:
        """Exaggeration above 1.0 should be clamped."""
        engine = ChatterboxEngine(exaggeration=1.5)
        assert engine.exaggeration == 1.0

    def test_exaggeration_clamping_low(self) -> None:
        """Exaggeration below 0.0 should be clamped."""
        engine = ChatterboxEngine(exaggeration=-0.5)
        assert engine.exaggeration == 0.0

    def test_exaggeration_setter(self, engine: ChatterboxEngine) -> None:
        """Setting exaggeration should clamp to valid range."""
        engine.exaggeration = 0.9
        assert engine.exaggeration == 0.9

        engine.exaggeration = 2.0
        assert engine.exaggeration == 1.0

        engine.exaggeration = -1.0
        assert engine.exaggeration == 0.0

    def test_is_loaded_false_initially(self, engine: ChatterboxEngine) -> None:
        """Engine should not be loaded initially."""
        assert engine.is_loaded() is False


# --- Model Loading ---


class TestModelLoading:
    """Tests for model load and unload."""

    @pytest.mark.asyncio
    async def test_load_model(
        self, engine: ChatterboxEngine, mock_chatterbox_model: MagicMock
    ) -> None:
        """Loading model should set _model and sample_rate."""
        mock_module = MagicMock()
        mock_module.ChatterboxTTS.from_pretrained.return_value = mock_chatterbox_model

        with patch.dict("sys.modules", {"chatterbox": MagicMock(), "chatterbox.tts": mock_module}):
            await engine.load_model()

        assert engine.is_loaded()
        assert engine.sample_rate == 24000

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(
        self, loaded_engine: ChatterboxEngine
    ) -> None:
        """Loading an already-loaded model should be a no-op."""
        await loaded_engine.load_model()
        assert loaded_engine.is_loaded()

    @pytest.mark.asyncio
    async def test_unload_model(self, loaded_engine: ChatterboxEngine) -> None:
        """Unloading should clear the model and free resources."""
        await loaded_engine.unload_model()

        assert loaded_engine._model is None
        assert not loaded_engine.is_loaded()

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self, engine: ChatterboxEngine) -> None:
        """Unloading when not loaded should be a no-op."""
        await engine.unload_model()
        assert not engine.is_loaded()


# --- Synthesis ---


class TestSynthesis:
    """Tests for text synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_not_loaded_raises(self, engine: ChatterboxEngine) -> None:
        """Synthesizing without loading should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not loaded"):
            await engine.synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_synthesize_returns_tts_result(
        self, loaded_engine: ChatterboxEngine
    ) -> None:
        """Synthesize should return a TTSResult with PCM data."""
        result = await loaded_engine.synthesize("Hello world")

        assert isinstance(result, TTSResult)
        assert result.sample_rate == 24000
        assert result.duration_seconds > 0
        assert len(result.audio_data) > 0

        # Verify PCM 16-bit: data length should be even (2 bytes per sample)
        assert len(result.audio_data) % 2 == 0

    @pytest.mark.asyncio
    async def test_synthesize_calls_model_generate(
        self,
        loaded_engine: ChatterboxEngine,
        mock_chatterbox_model: MagicMock,
    ) -> None:
        """Synthesize should call model.generate with correct args."""
        await loaded_engine.synthesize("Hello world")

        mock_chatterbox_model.generate.assert_called_once()
        call_kwargs = mock_chatterbox_model.generate.call_args.kwargs
        assert call_kwargs["text"] == "Hello world"
        assert call_kwargs["exaggeration"] == 0.5

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_reference(
        self,
        loaded_engine: ChatterboxEngine,
        mock_chatterbox_model: MagicMock,
        voice_ref_wav: Path,
    ) -> None:
        """Synthesize with voice reference should load and pass it to model."""
        result = await loaded_engine.synthesize("Hello world", voice_sample=voice_ref_wav)

        assert isinstance(result, TTSResult)
        assert len(result.audio_data) > 0

        # Verify audio_prompt was passed to generate
        call_kwargs = mock_chatterbox_model.generate.call_args.kwargs
        assert "audio_prompt" in call_kwargs
        assert call_kwargs["audio_prompt"] is not None

    @pytest.mark.asyncio
    async def test_synthesize_passes_exaggeration(
        self,
        loaded_engine: ChatterboxEngine,
        mock_chatterbox_model: MagicMock,
    ) -> None:
        """Synthesize should pass exaggeration to the model."""
        loaded_engine.exaggeration = 0.7

        await loaded_engine.synthesize("Test")

        call_kwargs = mock_chatterbox_model.generate.call_args.kwargs
        assert call_kwargs["exaggeration"] == 0.7

    @pytest.mark.asyncio
    async def test_synthesize_without_voice_reference(
        self,
        loaded_engine: ChatterboxEngine,
        mock_chatterbox_model: MagicMock,
    ) -> None:
        """Synthesize without voice_sample should not pass audio_prompt."""
        await loaded_engine.synthesize("Test")

        call_kwargs = mock_chatterbox_model.generate.call_args.kwargs
        # audio_prompt should not be in kwargs (or be None)
        assert call_kwargs.get("audio_prompt") is None

    @pytest.mark.asyncio
    async def test_synthesize_duration_correct(
        self, loaded_engine: ChatterboxEngine
    ) -> None:
        """Duration should match the number of samples / sample rate."""
        result = await loaded_engine.synthesize("Test")

        num_samples = len(result.audio_data) // 2
        expected_duration = num_samples / result.sample_rate
        assert abs(result.duration_seconds - expected_duration) < 0.001


# --- Streaming ---


class TestSynthesisStream:
    """Tests for streaming synthesis."""

    @pytest.mark.asyncio
    async def test_stream_not_loaded_raises(self, engine: ChatterboxEngine) -> None:
        """Streaming without loading should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not loaded"):
            async for _ in engine.synthesize_stream("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(
        self, loaded_engine: ChatterboxEngine
    ) -> None:
        """Streaming should yield multiple chunks of PCM data."""
        chunks: list[bytes] = []
        async for chunk in loaded_engine.synthesize_stream("Hello world"):
            chunks.append(chunk)

        assert len(chunks) > 0
        # All chunks should contain bytes
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0

        # Combined chunks should have valid PCM alignment
        combined = b"".join(chunks)
        assert len(combined) % 2 == 0  # 16-bit PCM

    @pytest.mark.asyncio
    async def test_stream_total_matches_batch(
        self,
        loaded_engine: ChatterboxEngine,
        mock_chatterbox_model: MagicMock,
    ) -> None:
        """Streaming total data should match batch synthesis output."""
        batch_result = await loaded_engine.synthesize("Test phrase")

        # Reset mock for second call (to return same data)
        mock_chatterbox_model.generate.reset_mock()

        chunks: list[bytes] = []
        async for chunk in loaded_engine.synthesize_stream("Test phrase"):
            chunks.append(chunk)

        stream_data = b"".join(chunks)
        assert len(stream_data) == len(batch_result.audio_data)

    @pytest.mark.asyncio
    async def test_stream_with_voice_reference(
        self, loaded_engine: ChatterboxEngine, voice_ref_wav: Path
    ) -> None:
        """Streaming with voice reference should work."""
        chunks: list[bytes] = []
        async for chunk in loaded_engine.synthesize_stream("Test", voice_sample=voice_ref_wav):
            chunks.append(chunk)

        assert len(chunks) > 0


# --- Tensor Conversion ---


class TestTensorConversion:
    """Tests for tensor to PCM bytes conversion."""

    def test_tensor_to_pcm_bytes(self, mock_torch_tensor: torch.Tensor) -> None:
        """Conversion should produce correct length PCM bytes."""
        pcm = ChatterboxEngine._tensor_to_pcm_bytes(mock_torch_tensor)

        # 24000 samples * 2 bytes per sample = 48000 bytes
        assert len(pcm) == 24000 * 2
        assert isinstance(pcm, bytes)

    def test_tensor_to_pcm_bytes_values(self) -> None:
        """Conversion should produce correct 16-bit values."""
        # Simple test: tensor with known values
        audio = torch.tensor([[0.5, -0.5, 0.0, 1.0, -1.0]])
        pcm = ChatterboxEngine._tensor_to_pcm_bytes(audio)

        # Unpack 16-bit signed integers
        values = np.frombuffer(pcm, dtype=np.int16)
        assert len(values) == 5

        # 0.5 * 32767 = 16383
        assert values[0] == 16383
        # -0.5 * 32767 = -16383
        assert values[1] == -16383
        # 0.0 = 0
        assert values[2] == 0
        # 1.0 * 32767 = 32767
        assert values[3] == 32767
        # -1.0 * 32767 = -32767
        assert values[4] == -32767

    def test_tensor_to_pcm_bytes_clipping(self) -> None:
        """Values exceeding [-1, 1] should be clamped."""
        audio = torch.tensor([[2.0, -2.0]])
        pcm = ChatterboxEngine._tensor_to_pcm_bytes(audio)
        values = np.frombuffer(pcm, dtype=np.int16)

        # Should be clamped to max/min int16 (from 1.0/-1.0 after clamp)
        assert values[0] == 32767
        assert values[1] == -32767

    def test_tensor_to_pcm_non_tensor_raises(self) -> None:
        """Passing a non-tensor should raise TypeError."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            ChatterboxEngine._tensor_to_pcm_bytes("not a tensor")


# --- Voice Reference Loading ---


class TestVoiceReferenceLoading:
    """Tests for loading voice reference files."""

    def test_load_missing_reference_raises(self, engine: ChatterboxEngine) -> None:
        """Loading a nonexistent reference should raise FileNotFoundError."""
        engine._sample_rate = 24000
        with pytest.raises(FileNotFoundError, match="not found"):
            engine._load_voice_reference(Path("/nonexistent/reference.wav"))

    def test_load_valid_reference(
        self, engine: ChatterboxEngine, voice_ref_wav: Path
    ) -> None:
        """Loading a valid reference should return a tensor."""
        engine._device = "cpu"
        engine._sample_rate = 24000

        result = engine._load_voice_reference(voice_ref_wav)

        # Should be a torch tensor
        assert isinstance(result, torch.Tensor)
        # Should be mono (1 channel)
        assert result.shape[0] == 1
        # Should have samples
        assert result.shape[1] > 0

    def test_load_reference_resamples(
        self, engine: ChatterboxEngine, voice_ref_wav: Path
    ) -> None:
        """Loading a reference at different sample rate should resample."""
        engine._device = "cpu"
        engine._sample_rate = 24000  # ref file is at 22050

        result = engine._load_voice_reference(voice_ref_wav)

        # The result should be resampled to 24kHz
        expected_samples = int(3.0 * 24000)  # ~3 seconds at 24kHz
        actual_samples = result.shape[1]
        # Allow some tolerance for resampling
        assert abs(actual_samples - expected_samples) < 100


# --- Save WAV ---


class TestSaveWav:
    """Tests for saving PCM bytes as WAV files."""

    def test_save_wav(self, engine: ChatterboxEngine, tmp_path: Path) -> None:
        """Saving PCM bytes should create a valid WAV file."""
        engine._sample_rate = 24000

        # Generate some test PCM data
        samples = 24000  # 1 second
        t = np.linspace(0, 1.0, samples, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        pcm_bytes = audio.tobytes()

        output_path = tmp_path / "output.wav"
        engine.save_wav(pcm_bytes, output_path)

        assert output_path.exists()

        # Verify the WAV is readable
        data, sr = sf.read(str(output_path))
        assert sr == 24000
        assert len(data) == 24000

    def test_save_wav_creates_dirs(
        self, engine: ChatterboxEngine, tmp_path: Path
    ) -> None:
        """save_wav should create parent directories if needed."""
        engine._sample_rate = 24000

        audio = np.zeros(1000, dtype=np.int16)
        output_path = tmp_path / "deep" / "nested" / "dir" / "output.wav"

        engine.save_wav(audio.tobytes(), output_path)
        assert output_path.exists()
