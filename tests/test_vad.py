"""Tests for the Silero VAD processor.

All tests mock the Silero VAD model since we cannot load it in CI.
Tests cover:
- Model loading and unloading
- Speech detection (process_frame)
- Utterance boundary detection (process_chunk)
- State machine transitions (silence -> speech -> silence)
- Audio buffering and flushing
- Reset behavior
- Reframing to Silero chunk size
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from zoom_auto.config import VADConfig
from zoom_auto.pipeline.vad import (
    _BYTES_PER_SAMPLE,
    _SILERO_CHUNK_SAMPLES,
    VADEvent,
    VADProcessor,
    VADResult,
)

# --- Fixtures ---


@pytest.fixture
def config() -> VADConfig:
    """Create a default VAD config."""
    return VADConfig(threshold=0.5, min_speech_duration=0.25, min_silence_duration=0.3)


@pytest.fixture
def processor(config: VADConfig) -> VADProcessor:
    """Create a VADProcessor with default config (no model loaded)."""
    return VADProcessor(config=config)


@pytest.fixture
def mock_vad_model() -> MagicMock:
    """Create a mock Silero VAD model."""
    model = MagicMock()
    model.eval.return_value = model
    model.reset_states.return_value = None
    # Default: return low confidence (no speech)
    model.return_value = torch.tensor(0.1)
    return model


@pytest.fixture
def loaded_processor(
    processor: VADProcessor, mock_vad_model: MagicMock
) -> VADProcessor:
    """Create a processor with a pre-loaded mock model."""
    processor._model = mock_vad_model
    processor._model_utils = MagicMock()
    processor._torch = torch
    return processor


def make_pcm_chunk(num_samples: int = _SILERO_CHUNK_SAMPLES) -> bytes:
    """Generate a PCM chunk of silence with the given number of samples."""
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def make_silero_chunk() -> bytes:
    """Generate exactly one Silero-sized chunk (512 samples)."""
    return make_pcm_chunk(_SILERO_CHUNK_SAMPLES)


# --- Constructor and Properties ---


class TestVADProcessorInit:
    """Tests for VADProcessor initialization."""

    def test_default_init(self) -> None:
        """Default constructor should set expected defaults."""
        proc = VADProcessor()
        assert proc._config.threshold == 0.5
        assert proc._config.min_speech_duration == 0.25
        assert proc._config.min_silence_duration == 0.3
        assert not proc.is_loaded()
        assert not proc._in_speech

    def test_custom_config(self) -> None:
        """Custom config should be stored correctly."""
        config = VADConfig(threshold=0.7, min_speech_duration=0.5, min_silence_duration=0.4)
        proc = VADProcessor(config=config)
        assert proc._config.threshold == 0.7
        assert proc._config.min_speech_duration == 0.5

    def test_is_loaded_false_initially(self, processor: VADProcessor) -> None:
        """Processor should not be loaded initially."""
        assert processor.is_loaded() is False


# --- Model Loading ---


class TestModelLoading:
    """Tests for model load and unload."""

    @pytest.mark.asyncio
    async def test_load_model(self, processor: VADProcessor) -> None:
        """Loading model should set _model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_utils = MagicMock()

        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            await processor.load_model()

        assert processor.is_loaded()
        assert processor._model is mock_model

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(
        self, loaded_processor: VADProcessor
    ) -> None:
        """Loading an already-loaded model should be a no-op."""
        original_model = loaded_processor._model
        await loaded_processor.load_model()
        assert loaded_processor._model is original_model

    @pytest.mark.asyncio
    async def test_unload_model(self, loaded_processor: VADProcessor) -> None:
        """Unloading should clear the model and reset state."""
        assert loaded_processor.is_loaded()
        await loaded_processor.unload_model()
        assert loaded_processor._model is None
        assert not loaded_processor.is_loaded()

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(
        self, processor: VADProcessor
    ) -> None:
        """Unloading when not loaded should be a no-op."""
        await processor.unload_model()
        assert not processor.is_loaded()


# --- Reset ---


class TestReset:
    """Tests for reset behavior."""

    def test_reset_clears_state(self, loaded_processor: VADProcessor) -> None:
        """Reset should clear all internal state."""
        # Simulate some state
        loaded_processor._in_speech = True
        loaded_processor._audio_buffer.extend(b"\x00" * 1000)
        loaded_processor._speech_samples = 500
        loaded_processor._silence_samples = 100
        loaded_processor._total_samples_processed = 5000

        loaded_processor.reset()

        assert not loaded_processor._in_speech
        assert len(loaded_processor._audio_buffer) == 0
        assert loaded_processor._speech_samples == 0
        assert loaded_processor._silence_samples == 0
        assert loaded_processor._total_samples_processed == 0

    def test_reset_calls_model_reset_states(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Reset should call model.reset_states()."""
        loaded_processor.reset()
        mock_vad_model.reset_states.assert_called_once()


# --- process_frame ---


class TestProcessFrame:
    """Tests for single-frame processing."""

    @pytest.mark.asyncio
    async def test_process_frame_not_loaded_raises(
        self, processor: VADProcessor
    ) -> None:
        """Processing without loading should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not loaded"):
            await processor.process_frame(make_silero_chunk())

    @pytest.mark.asyncio
    async def test_process_frame_no_speech(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Frame with low confidence should report no speech."""
        mock_vad_model.return_value = torch.tensor(0.1)

        result = await loaded_processor.process_frame(make_silero_chunk())

        assert isinstance(result, VADResult)
        assert result.is_speech is False
        assert result.confidence == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_process_frame_speech_detected(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Frame with high confidence should report speech."""
        mock_vad_model.return_value = torch.tensor(0.9)

        result = await loaded_processor.process_frame(make_silero_chunk())

        assert result.is_speech is True
        assert result.confidence == pytest.approx(0.9)


# --- process_chunk State Machine ---


class TestProcessChunkStateMachine:
    """Tests for the speech start/end state machine via process_chunk."""

    @pytest.mark.asyncio
    async def test_process_chunk_not_loaded_raises(
        self, processor: VADProcessor
    ) -> None:
        """Processing without loading should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not loaded"):
            await processor.process_chunk(make_silero_chunk())

    @pytest.mark.asyncio
    async def test_silence_returns_none(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Continued silence should not emit any event."""
        mock_vad_model.return_value = torch.tensor(0.1)

        event = await loaded_processor.process_chunk(make_silero_chunk())
        assert event is None

    @pytest.mark.asyncio
    async def test_speech_start_event(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Speech should trigger a speech_start event after min_speech_duration.

        With min_speech_duration=0.25s at 16kHz, we need 4000 samples.
        Each Silero chunk is 512 samples, so we need ceil(4000/512) = 8 chunks.
        """
        mock_vad_model.return_value = torch.tensor(0.9)

        chunks_needed = int(
            loaded_processor._min_speech_samples / _SILERO_CHUNK_SAMPLES
        ) + 1

        event = None
        for _ in range(chunks_needed):
            result = await loaded_processor.process_chunk(make_silero_chunk())
            if result is not None:
                event = result
                break

        assert event is not None
        assert event.is_speech_start is True
        assert event.is_speech_end is False

    @pytest.mark.asyncio
    async def test_speech_end_event_with_buffer(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """After speech, silence should trigger speech_end with audio buffer.

        Flow: speech chunks until start -> silence chunks until end.
        """
        # Phase 1: Send speech chunks to trigger speech_start
        mock_vad_model.return_value = torch.tensor(0.9)

        speech_chunks = int(
            loaded_processor._min_speech_samples / _SILERO_CHUNK_SAMPLES
        ) + 2

        speech_started = False
        for _ in range(speech_chunks):
            result = await loaded_processor.process_chunk(make_silero_chunk())
            if result is not None and result.is_speech_start:
                speech_started = True

        assert speech_started, "Speech should have started"
        assert loaded_processor._in_speech

        # Phase 2: Send silence chunks to trigger speech_end
        mock_vad_model.return_value = torch.tensor(0.1)

        silence_chunks = int(
            loaded_processor._min_silence_samples / _SILERO_CHUNK_SAMPLES
        ) + 2

        end_event = None
        for _ in range(silence_chunks):
            result = await loaded_processor.process_chunk(make_silero_chunk())
            if result is not None and result.is_speech_end:
                end_event = result
                break

        assert end_event is not None
        assert end_event.is_speech_end is True
        assert end_event.audio_buffer is not None
        assert len(end_event.audio_buffer) > 0
        assert not loaded_processor._in_speech

    @pytest.mark.asyncio
    async def test_tentative_speech_discarded_on_quick_silence(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Brief speech below min_speech_duration should be discarded."""
        # Send 1-2 speech chunks (not enough to trigger start)
        mock_vad_model.return_value = torch.tensor(0.9)
        await loaded_processor.process_chunk(make_silero_chunk())

        assert loaded_processor._speech_samples > 0
        assert not loaded_processor._in_speech

        # Then silence — should discard the tentative buffer
        mock_vad_model.return_value = torch.tensor(0.1)
        event = await loaded_processor.process_chunk(make_silero_chunk())

        assert event is None
        assert loaded_processor._speech_samples == 0
        assert len(loaded_processor._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_silence_counter_resets_on_continued_speech(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """Brief silence during speech should not trigger end."""
        # Start speech
        mock_vad_model.return_value = torch.tensor(0.9)
        chunks_for_start = int(
            loaded_processor._min_speech_samples / _SILERO_CHUNK_SAMPLES
        ) + 2

        for _ in range(chunks_for_start):
            await loaded_processor.process_chunk(make_silero_chunk())

        assert loaded_processor._in_speech

        # Brief silence (1 chunk - not enough for end)
        mock_vad_model.return_value = torch.tensor(0.1)
        event = await loaded_processor.process_chunk(make_silero_chunk())
        assert event is None  # No end event yet
        assert loaded_processor._silence_samples > 0

        # Resume speech — silence counter should reset
        mock_vad_model.return_value = torch.tensor(0.9)
        event = await loaded_processor.process_chunk(make_silero_chunk())
        assert event is None
        assert loaded_processor._silence_samples == 0
        assert loaded_processor._in_speech


# --- Reframing ---


class TestReframing:
    """Tests for chunk reframing to Silero's expected size."""

    @pytest.mark.asyncio
    async def test_large_chunk_processes_multiple_silero_chunks(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """A chunk larger than 512 samples should be split into Silero chunks."""
        mock_vad_model.return_value = torch.tensor(0.1)

        # Send 1024 samples (2x Silero chunk)
        large_chunk = make_pcm_chunk(1024)
        await loaded_processor.process_chunk(large_chunk)

        # Model should have been called twice (once per 512-sample chunk)
        assert mock_vad_model.call_count == 2

    @pytest.mark.asyncio
    async def test_small_chunk_buffers_until_full(
        self, loaded_processor: VADProcessor, mock_vad_model: MagicMock
    ) -> None:
        """A chunk smaller than 512 samples should be buffered."""
        mock_vad_model.return_value = torch.tensor(0.1)

        # Send 256 samples (half a Silero chunk)
        small_chunk = make_pcm_chunk(256)
        await loaded_processor.process_chunk(small_chunk)

        # Model should not have been called yet
        mock_vad_model.assert_not_called()
        assert len(loaded_processor._chunk_buffer) == 256 * _BYTES_PER_SAMPLE

        # Send another 256 samples — now we have a full chunk
        await loaded_processor.process_chunk(small_chunk)
        mock_vad_model.assert_called_once()


# --- get_speech_segment ---


class TestGetSpeechSegment:
    """Tests for forced speech segment extraction."""

    @pytest.mark.asyncio
    async def test_no_segment_when_not_speaking(
        self, loaded_processor: VADProcessor
    ) -> None:
        """Should return None when not in speech."""
        result = await loaded_processor.get_speech_segment()
        assert result is None

    @pytest.mark.asyncio
    async def test_force_flush_during_speech(
        self, loaded_processor: VADProcessor
    ) -> None:
        """Should return buffered audio and reset state."""
        loaded_processor._in_speech = True
        loaded_processor._audio_buffer.extend(b"\x01\x02" * 500)

        segment = await loaded_processor.get_speech_segment()

        assert segment is not None
        assert len(segment) == 1000
        assert not loaded_processor._in_speech
        assert len(loaded_processor._audio_buffer) == 0


# --- VADEvent and VADResult dataclasses ---


class TestDataclasses:
    """Tests for VADEvent and VADResult dataclasses."""

    def test_vad_event_defaults(self) -> None:
        """VADEvent should have sensible defaults."""
        event = VADEvent()
        assert event.is_speech_start is False
        assert event.is_speech_end is False
        assert event.audio_buffer is None

    def test_vad_event_speech_start(self) -> None:
        """VADEvent for speech start."""
        event = VADEvent(is_speech_start=True)
        assert event.is_speech_start is True
        assert event.is_speech_end is False

    def test_vad_event_speech_end_with_buffer(self) -> None:
        """VADEvent for speech end with audio buffer."""
        audio = b"\x00" * 1000
        event = VADEvent(is_speech_end=True, audio_buffer=audio)
        assert event.is_speech_end is True
        assert event.audio_buffer == audio

    def test_vad_result_defaults(self) -> None:
        """VADResult should have sensible defaults."""
        result = VADResult(is_speech=False, confidence=0.3)
        assert result.is_speech is False
        assert result.confidence == 0.3
        assert result.speech_start is None
        assert result.speech_end is None
