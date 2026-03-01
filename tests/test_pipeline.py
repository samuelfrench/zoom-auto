"""Tests for audio pipeline, conversation loop, and main app orchestrator.

All tests use mocks for external dependencies (no real Zoom SDK,
no real ML models, no real LLM APIs).

Covers:
- AudioPipeline: start/stop lifecycle, frame processing (capture -> VAD -> STT),
  send_response (text -> TTS -> sender), interruption handling
- ConversationLoop: process_utterance (transcript + trigger + response),
  start/stop lifecycle, main loop integration
- ZoomAutoApp: initialization, provider selection, event wiring, lifecycle
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from zoom_auto.config import (
    ContextConfig,
    LLMConfig,
    ResponseConfig,
    Settings,
    TTSConfig,
    VADConfig,
    ZoomConfig,
)
from zoom_auto.context.manager import ContextManager
from zoom_auto.main import ZoomAutoApp
from zoom_auto.pipeline.audio_pipeline import AudioPipeline
from zoom_auto.pipeline.conversation import ConversationLoop
from zoom_auto.pipeline.vad import VADEvent
from zoom_auto.response.decision import ResponseDecision, TriggerDetector, TriggerReason
from zoom_auto.response.generator import GeneratedResponse, ResponseGenerator
from zoom_auto.response.turn_manager import TurnManager
from zoom_auto.stt.base import TranscriptionResult
from zoom_auto.tts.base import TTSResult
from zoom_auto.zoom.audio_capture import AudioCapture, AudioFrame
from zoom_auto.zoom.audio_sender import AudioSender
from zoom_auto.zoom.events import MeetingEvent, ParticipantInfo

# ===================================================================== #
#  Fixtures                                                              #
# ===================================================================== #


@pytest.fixture
def settings() -> Settings:
    """Settings with dummy credentials and safe defaults."""
    return Settings(
        zoom_meeting_sdk_key="test_key",
        zoom_meeting_sdk_secret="test_secret",
        anthropic_api_key="test_api_key",
        zoom=ZoomConfig(bot_name="TestBot"),
        tts=TTSConfig(voice_sample_dir="/tmp/nonexistent_voice_dir"),
        vad=VADConfig(),
        response=ResponseConfig(cooldown_seconds=0.1, max_consecutive=5),
        llm=LLMConfig(provider="claude"),
        context=ContextConfig(),
    )


@pytest.fixture
def mock_capture() -> AsyncMock:
    """Mock AudioCapture."""
    capture = AsyncMock(spec=AudioCapture)
    capture.is_active = True
    capture._queue = asyncio.Queue()
    return capture


@pytest.fixture
def mock_sender() -> AsyncMock:
    """Mock AudioSender."""
    sender = AsyncMock(spec=AudioSender)
    sender.is_active = True
    sender.clear_pending = MagicMock()
    return sender


@pytest.fixture
def mock_vad() -> AsyncMock:
    """Mock VADProcessor."""
    vad = AsyncMock(spec=MagicMock)
    vad.process_chunk = AsyncMock(return_value=None)
    vad.load_model = AsyncMock()
    vad.unload_model = AsyncMock()
    vad.is_loaded = MagicMock(return_value=True)
    return vad


@pytest.fixture
def mock_stt() -> AsyncMock:
    """Mock STTEngine."""
    stt = AsyncMock()
    stt.transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="Hello, how are you?",
            language="en",
            confidence=0.95,
        )
    )
    stt.load_model = AsyncMock()
    stt.unload_model = AsyncMock()
    stt.is_loaded = MagicMock(return_value=True)
    return stt


@pytest.fixture
def mock_tts() -> AsyncMock:
    """Mock TTSEngine."""
    tts = AsyncMock()
    tts.synthesize = AsyncMock(
        return_value=TTSResult(
            audio_data=b"\x00" * 3200,
            sample_rate=22050,
            duration_seconds=0.5,
        )
    )
    tts.load_model = AsyncMock()
    tts.unload_model = AsyncMock()
    tts.is_loaded = MagicMock(return_value=True)
    return tts


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock LLMProvider."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=MagicMock(
            text="Sure, I can help with that.",
            usage_output_tokens=10,
        )
    )
    llm.decide = AsyncMock(return_value=(True, 0.9))
    llm.is_available = AsyncMock(return_value=True)
    return llm


@pytest.fixture
def mock_context_manager() -> AsyncMock:
    """Mock ContextManager."""
    ctx = AsyncMock(spec=ContextManager)
    ctx.add_transcript = AsyncMock()
    ctx.get_context = AsyncMock(
        return_value=MagicMock(
            recent_transcript=["Alice: Let's start the meeting."],
            summary="",
            meeting_context="",
        )
    )
    ctx.meeting_state = MagicMock()
    ctx.meeting_state.add_participant = MagicMock()
    ctx.reset = AsyncMock()
    return ctx


@pytest.fixture
def mock_trigger_detector() -> AsyncMock:
    """Mock TriggerDetector."""
    detector = AsyncMock(spec=TriggerDetector)
    detector.should_respond = AsyncMock(
        return_value=ResponseDecision(
            should_respond=True,
            confidence=0.95,
            reason=TriggerReason.DIRECT_ADDRESS,
            context_snippet="Hey TestBot, what do you think?",
        )
    )
    return detector


@pytest.fixture
def mock_response_generator() -> AsyncMock:
    """Mock ResponseGenerator."""
    gen = AsyncMock(spec=ResponseGenerator)
    gen.generate = AsyncMock(
        return_value=GeneratedResponse(
            text="I think we should proceed with the plan.",
            persona_applied=True,
            token_usage=15,
        )
    )
    return gen


@pytest.fixture
def turn_manager(settings: Settings) -> TurnManager:
    """Real TurnManager with test config."""
    return TurnManager(config=settings.response)


@pytest.fixture
def audio_pipeline(
    settings: Settings,
    mock_capture: AsyncMock,
    mock_sender: AsyncMock,
    mock_vad: AsyncMock,
    mock_stt: AsyncMock,
    mock_tts: AsyncMock,
) -> AudioPipeline:
    """AudioPipeline with all dependencies mocked."""
    return AudioPipeline(
        settings=settings,
        capture=mock_capture,
        sender=mock_sender,
        vad=mock_vad,
        stt=mock_stt,
        tts=mock_tts,
    )


@pytest.fixture
def conversation_loop(
    settings: Settings,
    audio_pipeline: AudioPipeline,
    mock_context_manager: AsyncMock,
    mock_trigger_detector: AsyncMock,
    mock_response_generator: AsyncMock,
    turn_manager: TurnManager,
) -> ConversationLoop:
    """ConversationLoop with mocked dependencies."""
    return ConversationLoop(
        settings=settings,
        audio_pipeline=audio_pipeline,
        context_manager=mock_context_manager,
        trigger_detector=mock_trigger_detector,
        response_generator=mock_response_generator,
        turn_manager=turn_manager,
    )


# ===================================================================== #
#  AudioPipeline tests                                                   #
# ===================================================================== #


class TestAudioPipelineLifecycle:
    """Tests for AudioPipeline start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(
        self, audio_pipeline: AudioPipeline, mock_capture: AsyncMock
    ) -> None:
        """Pipeline should be running after start."""
        # Make frames() return an empty async iterator
        async def empty_frames():
            return
            yield  # type: ignore[misc]  # noqa: B901

        mock_capture.frames = empty_frames

        await audio_pipeline.start()
        assert audio_pipeline.is_running is True
        await audio_pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(
        self, audio_pipeline: AudioPipeline, mock_capture: AsyncMock
    ) -> None:
        """Pipeline should not be running after stop."""
        async def empty_frames():
            return
            yield  # type: ignore[misc]  # noqa: B901

        mock_capture.frames = empty_frames

        await audio_pipeline.start()
        await audio_pipeline.stop()
        assert audio_pipeline.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_warns(
        self, audio_pipeline: AudioPipeline, mock_capture: AsyncMock
    ) -> None:
        """Starting an already running pipeline should be idempotent."""
        async def empty_frames():
            return
            yield  # type: ignore[misc]  # noqa: B901

        mock_capture.frames = empty_frames

        await audio_pipeline.start()
        await audio_pipeline.start()  # Should warn but not crash
        assert audio_pipeline.is_running is True
        await audio_pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(
        self, audio_pipeline: AudioPipeline
    ) -> None:
        """Stopping a pipeline that was never started should be safe."""
        await audio_pipeline.stop()  # Should not raise
        assert audio_pipeline.is_running is False


class TestAudioPipelineFrameProcessing:
    """Tests for frame routing: capture -> VAD -> STT."""

    @pytest.mark.asyncio
    async def test_frame_routed_through_vad(
        self,
        audio_pipeline: AudioPipeline,
        mock_capture: AsyncMock,
        mock_vad: AsyncMock,
    ) -> None:
        """Frames from capture should be fed to VAD."""
        frame = AudioFrame(
            speaker_id=1,
            speaker_name="Alice",
            pcm_data=b"\x00" * 640,
            sample_rate=16000,
            timestamp_ms=1000,
        )

        frames_yielded = False

        async def mock_frames():
            nonlocal frames_yielded
            yield frame
            frames_yielded = True

        mock_capture.frames = mock_frames
        mock_vad.process_chunk.return_value = None

        await audio_pipeline.start()
        # Give the capture loop time to process
        await asyncio.sleep(0.1)
        await audio_pipeline.stop()

        assert frames_yielded
        mock_vad.process_chunk.assert_called_with(frame.pcm_data)

    @pytest.mark.asyncio
    async def test_speech_end_triggers_stt(
        self,
        audio_pipeline: AudioPipeline,
        mock_capture: AsyncMock,
        mock_vad: AsyncMock,
        mock_stt: AsyncMock,
    ) -> None:
        """When VAD returns speech_end with buffer, STT should be called."""
        frame = AudioFrame(
            speaker_id=1,
            speaker_name="Alice",
            pcm_data=b"\x00" * 640,
            sample_rate=16000,
            timestamp_ms=1000,
        )

        speech_buffer = b"\x00" * 16000  # 0.5s of 16kHz audio

        async def mock_frames():
            yield frame

        mock_capture.frames = mock_frames
        mock_vad.process_chunk.return_value = VADEvent(
            is_speech_end=True,
            audio_buffer=speech_buffer,
        )

        await audio_pipeline.start()
        await asyncio.sleep(0.1)
        await audio_pipeline.stop()

        mock_stt.transcribe.assert_called_once_with(
            speech_buffer, frame.sample_rate
        )

    @pytest.mark.asyncio
    async def test_transcript_callback_invoked(
        self,
        audio_pipeline: AudioPipeline,
        mock_capture: AsyncMock,
        mock_vad: AsyncMock,
        mock_stt: AsyncMock,
    ) -> None:
        """Transcript callback should be invoked with speaker and text."""
        frame = AudioFrame(
            speaker_id=1,
            speaker_name="Alice",
            pcm_data=b"\x00" * 640,
            sample_rate=16000,
            timestamp_ms=1000,
        )

        callback = AsyncMock()
        audio_pipeline.set_transcript_callback(callback)

        async def mock_frames():
            yield frame

        mock_capture.frames = mock_frames
        mock_vad.process_chunk.return_value = VADEvent(
            is_speech_end=True,
            audio_buffer=b"\x00" * 16000,
        )
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.9,
        )

        await audio_pipeline.start()
        await asyncio.sleep(0.1)
        await audio_pipeline.stop()

        callback.assert_called_once_with("Alice", "Hello world")

    @pytest.mark.asyncio
    async def test_empty_transcription_skipped(
        self,
        audio_pipeline: AudioPipeline,
        mock_capture: AsyncMock,
        mock_vad: AsyncMock,
        mock_stt: AsyncMock,
    ) -> None:
        """Empty transcriptions should not invoke callback."""
        frame = AudioFrame(
            speaker_id=1,
            speaker_name="Alice",
            pcm_data=b"\x00" * 640,
            sample_rate=16000,
            timestamp_ms=1000,
        )

        callback = AsyncMock()
        audio_pipeline.set_transcript_callback(callback)

        async def mock_frames():
            yield frame

        mock_capture.frames = mock_frames
        mock_vad.process_chunk.return_value = VADEvent(
            is_speech_end=True,
            audio_buffer=b"\x00" * 16000,
        )
        mock_stt.transcribe.return_value = TranscriptionResult(
            text="   ",
            language="en",
            confidence=0.0,
        )

        await audio_pipeline.start()
        await asyncio.sleep(0.1)
        await audio_pipeline.stop()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_speech_start_event_no_stt(
        self,
        audio_pipeline: AudioPipeline,
        mock_capture: AsyncMock,
        mock_vad: AsyncMock,
        mock_stt: AsyncMock,
    ) -> None:
        """Speech start events should not trigger STT."""
        frame = AudioFrame(
            speaker_id=1,
            speaker_name="Alice",
            pcm_data=b"\x00" * 640,
            sample_rate=16000,
            timestamp_ms=1000,
        )

        async def mock_frames():
            yield frame

        mock_capture.frames = mock_frames
        mock_vad.process_chunk.return_value = VADEvent(is_speech_start=True)

        await audio_pipeline.start()
        await asyncio.sleep(0.1)
        await audio_pipeline.stop()

        mock_stt.transcribe.assert_not_called()


class TestAudioPipelineSendResponse:
    """Tests for send_response: text -> TTS -> sender."""

    @pytest.mark.asyncio
    async def test_send_response_synthesizes_and_sends(
        self,
        audio_pipeline: AudioPipeline,
        mock_tts: AsyncMock,
        mock_sender: AsyncMock,
    ) -> None:
        """send_response should synthesize via TTS and send via AudioSender."""
        await audio_pipeline.send_response("Hello everyone!")

        mock_tts.synthesize.assert_called_once()
        mock_sender.send_audio.assert_called_once_with(
            b"\x00" * 3200, 22050
        )

    @pytest.mark.asyncio
    async def test_send_response_empty_text_skipped(
        self,
        audio_pipeline: AudioPipeline,
        mock_tts: AsyncMock,
        mock_sender: AsyncMock,
    ) -> None:
        """Empty text should not trigger TTS."""
        await audio_pipeline.send_response("   ")

        mock_tts.synthesize.assert_not_called()
        mock_sender.send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_response_sets_bot_speaking(
        self,
        audio_pipeline: AudioPipeline,
        mock_tts: AsyncMock,
    ) -> None:
        """Bot should be marked speaking during send_response."""
        speaking_during_call = False

        original_synthesize = mock_tts.synthesize

        async def check_speaking(*args, **kwargs):
            nonlocal speaking_during_call
            speaking_during_call = audio_pipeline.is_bot_speaking
            return await original_synthesize(*args, **kwargs)

        mock_tts.synthesize = check_speaking

        await audio_pipeline.send_response("Testing speech state")

        assert speaking_during_call is True
        # After completion, should no longer be speaking
        assert audio_pipeline.is_bot_speaking is False

    @pytest.mark.asyncio
    async def test_send_response_tts_error_handled(
        self,
        audio_pipeline: AudioPipeline,
        mock_tts: AsyncMock,
        mock_sender: AsyncMock,
    ) -> None:
        """TTS errors should be caught and not crash the pipeline."""
        mock_tts.synthesize.side_effect = RuntimeError("TTS failed")

        # Should not raise
        await audio_pipeline.send_response("This will fail")

        mock_sender.send_audio.assert_not_called()
        assert audio_pipeline.is_bot_speaking is False


class TestAudioPipelineInterruption:
    """Tests for interruption handling."""

    @pytest.mark.asyncio
    async def test_stop_speaking_clears_state(
        self, audio_pipeline: AudioPipeline
    ) -> None:
        """stop_speaking should clear the bot_speaking flag."""
        audio_pipeline._bot_speaking = True
        await audio_pipeline.stop_speaking()
        assert audio_pipeline.is_bot_speaking is False

    @pytest.mark.asyncio
    async def test_stop_speaking_drains_sender_queue(
        self,
        audio_pipeline: AudioPipeline,
        mock_sender: AsyncMock,
    ) -> None:
        """stop_speaking should call clear_pending on sender."""
        await audio_pipeline.stop_speaking()

        mock_sender.clear_pending.assert_called_once()


# ===================================================================== #
#  ConversationLoop tests                                                #
# ===================================================================== #


class TestConversationLoopProcessUtterance:
    """Tests for process_utterance logic."""

    @pytest.mark.asyncio
    async def test_process_utterance_adds_transcript(
        self,
        conversation_loop: ConversationLoop,
        mock_context_manager: AsyncMock,
    ) -> None:
        """process_utterance should add transcript to context."""
        await conversation_loop.process_utterance(
            "Alice", "Let's discuss the roadmap"
        )

        mock_context_manager.add_transcript.assert_called()
        call_kwargs = mock_context_manager.add_transcript.call_args
        assert call_kwargs[1]["speaker"] == "Alice"
        assert call_kwargs[1]["text"] == "Let's discuss the roadmap"

    @pytest.mark.asyncio
    async def test_process_utterance_triggers_response(
        self,
        conversation_loop: ConversationLoop,
        mock_trigger_detector: AsyncMock,
        mock_response_generator: AsyncMock,
    ) -> None:
        """When trigger detector says respond, generator should be called."""
        result = await conversation_loop.process_utterance(
            "Alice", "Hey TestBot, what do you think?"
        )

        mock_trigger_detector.should_respond.assert_called_once()
        mock_response_generator.generate.assert_called_once()
        assert result == "I think we should proceed with the plan."

    @pytest.mark.asyncio
    async def test_process_utterance_no_response_when_not_triggered(
        self,
        conversation_loop: ConversationLoop,
        mock_trigger_detector: AsyncMock,
        mock_response_generator: AsyncMock,
    ) -> None:
        """When trigger detector says no, generator should not be called."""
        mock_trigger_detector.should_respond.return_value = ResponseDecision(
            should_respond=False,
            confidence=0.9,
            reason=TriggerReason.NONE,
        )

        result = await conversation_loop.process_utterance(
            "Alice", "I was thinking about lunch"
        )

        assert result is None
        mock_response_generator.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_utterance_empty_text_returns_none(
        self,
        conversation_loop: ConversationLoop,
        mock_context_manager: AsyncMock,
    ) -> None:
        """Empty text should return None without processing."""
        result = await conversation_loop.process_utterance("Alice", "   ")

        assert result is None
        mock_context_manager.add_transcript.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_utterance_records_other_speaker(
        self,
        conversation_loop: ConversationLoop,
        turn_manager: TurnManager,
    ) -> None:
        """process_utterance should reset consecutive count."""
        turn_manager._consecutive_count = 3
        await conversation_loop.process_utterance(
            "Alice", "Something interesting"
        )
        assert turn_manager._consecutive_count == 0

    @pytest.mark.asyncio
    async def test_process_utterance_blocked_by_turn_manager(
        self,
        conversation_loop: ConversationLoop,
        turn_manager: TurnManager,
        mock_response_generator: AsyncMock,
    ) -> None:
        """Turn manager blocking should prevent response generation.

        If turn_manager.can_speak() returns False even after trigger says
        respond, the response should be discarded.
        """
        # Simulate: bot is already speaking so can_speak() returns False
        turn_manager.mark_bot_speaking()

        await conversation_loop.process_utterance(
            "Alice", "Hey TestBot, help me"
        )

        # can_speak() returns False because bot_speaking is True,
        # so the response is blocked even though trigger says respond.

    @pytest.mark.asyncio
    async def test_process_utterance_interruption_stops_bot(
        self,
        conversation_loop: ConversationLoop,
        turn_manager: TurnManager,
        audio_pipeline: AudioPipeline,
    ) -> None:
        """If bot is speaking and someone else talks, bot should stop."""
        turn_manager.mark_bot_speaking()
        audio_pipeline._bot_speaking = True
        assert turn_manager.bot_speaking is True

        await conversation_loop.process_utterance(
            "Alice", "Actually, wait."
        )

        # Bot should have been stopped via stop_speaking
        assert turn_manager.bot_speaking is False

    @pytest.mark.asyncio
    async def test_process_utterance_empty_response_returns_none(
        self,
        conversation_loop: ConversationLoop,
        mock_response_generator: AsyncMock,
    ) -> None:
        """If the generated response is empty, return None."""
        mock_response_generator.generate.return_value = GeneratedResponse(
            text="   ",
            persona_applied=False,
            token_usage=0,
        )

        result = await conversation_loop.process_utterance(
            "Alice", "Hey TestBot"
        )

        assert result is None


class TestConversationLoopLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(
        self,
        conversation_loop: ConversationLoop,
        mock_capture: AsyncMock,
    ) -> None:
        """Conversation loop should be running after start."""
        async def empty_frames():
            return
            yield  # type: ignore[misc]  # noqa: B901

        mock_capture.frames = empty_frames

        await conversation_loop.start()
        assert conversation_loop.is_running is True
        await conversation_loop.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(
        self,
        conversation_loop: ConversationLoop,
        mock_capture: AsyncMock,
    ) -> None:
        """Conversation loop should not be running after stop."""
        async def empty_frames():
            return
            yield  # type: ignore[misc]  # noqa: B901

        mock_capture.frames = empty_frames

        await conversation_loop.start()
        await conversation_loop.stop()
        assert conversation_loop.is_running is False

    @pytest.mark.asyncio
    async def test_stop_without_start(
        self, conversation_loop: ConversationLoop
    ) -> None:
        """Stopping without starting should be safe."""
        await conversation_loop.stop()
        assert conversation_loop.is_running is False

    @pytest.mark.asyncio
    async def test_main_loop_processes_queued_utterance(
        self,
        conversation_loop: ConversationLoop,
        mock_capture: AsyncMock,
        mock_trigger_detector: AsyncMock,
        mock_response_generator: AsyncMock,
        mock_context_manager: AsyncMock,
    ) -> None:
        """The main loop should process utterances from the queue."""
        async def empty_frames():
            return
            yield  # type: ignore[misc]  # noqa: B901

        mock_capture.frames = empty_frames

        await conversation_loop.start()

        # Push an utterance directly into the queue
        await conversation_loop._utterance_queue.put(
            ("Alice", "Hey TestBot, respond please")
        )

        # Wait for processing
        await asyncio.sleep(0.3)
        await conversation_loop.stop()

        mock_context_manager.add_transcript.assert_called()


# ===================================================================== #
#  ZoomAutoApp tests                                                     #
# ===================================================================== #


class TestZoomAutoAppInit:
    """Tests for ZoomAutoApp initialization."""

    def test_creates_all_components(self, settings: Settings) -> None:
        """App should create all required components."""
        app = ZoomAutoApp(settings)

        assert app.event_handler is not None
        assert app.zoom_client is not None
        assert app.audio_capture is not None
        assert app.audio_sender is not None
        assert app.vad is not None
        assert app.stt is not None
        assert app.tts is not None
        assert app.llm is not None
        assert app.context_manager is not None
        assert app.trigger_detector is not None
        assert app.response_generator is not None
        assert app.turn_manager is not None
        assert app.audio_pipeline is not None
        assert app.conversation_loop is not None

    def test_claude_provider_selected(self, settings: Settings) -> None:
        """With API key and provider=claude, ClaudeProvider should be used."""
        from zoom_auto.llm.claude import ClaudeProvider

        app = ZoomAutoApp(settings)
        assert isinstance(app.llm, ClaudeProvider)

    def test_ollama_provider_selected(self) -> None:
        """With provider=ollama, OllamaProvider should be used."""
        from zoom_auto.llm.ollama import OllamaProvider

        settings = Settings(
            zoom_meeting_sdk_key="k",
            zoom_meeting_sdk_secret="s",
            llm=LLMConfig(provider="ollama"),
        )
        app = ZoomAutoApp(settings)
        assert isinstance(app.llm, OllamaProvider)

    def test_claude_fallback_to_ollama_without_key(self) -> None:
        """Without API key, Claude should fall back to Ollama."""
        from zoom_auto.llm.ollama import OllamaProvider

        settings = Settings(
            zoom_meeting_sdk_key="k",
            zoom_meeting_sdk_secret="s",
            anthropic_api_key="",
            llm=LLMConfig(provider="claude"),
        )
        app = ZoomAutoApp(settings)
        assert isinstance(app.llm, OllamaProvider)

    def test_is_running_initially_false(self, settings: Settings) -> None:
        """App should not be running initially."""
        app = ZoomAutoApp(settings)
        assert app.is_running is False


class TestZoomAutoAppLifecycle:
    """Tests for ZoomAutoApp start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_loads_models_and_starts_pipeline(
        self, settings: Settings
    ) -> None:
        """start() should load models and start the pipeline."""
        app = ZoomAutoApp(settings)

        # Mock all the ML model loading
        app.stt.load_model = AsyncMock()
        app.tts.load_model = AsyncMock()
        app.vad.load_model = AsyncMock()
        app.audio_capture.start = AsyncMock()
        app.audio_sender.start = AsyncMock()
        app.conversation_loop.start = AsyncMock()

        # Start in background and stop shortly after
        async def stop_after_delay():
            await asyncio.sleep(0.2)
            app._running = False

        task = asyncio.create_task(app.start())
        stop_task = asyncio.create_task(stop_after_delay())

        await asyncio.gather(task, stop_task)

        app.stt.load_model.assert_called_once()
        app.tts.load_model.assert_called_once()
        app.vad.load_model.assert_called_once()
        app.audio_capture.start.assert_called_once()
        app.audio_sender.start.assert_called_once()
        app.conversation_loop.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_components(
        self, settings: Settings
    ) -> None:
        """stop() should shut down all components."""
        app = ZoomAutoApp(settings)

        app.conversation_loop.stop = AsyncMock()
        app.audio_capture.stop = AsyncMock()
        app.audio_sender.stop = AsyncMock()
        app.stt.unload_model = AsyncMock()
        app.tts.unload_model = AsyncMock()
        app.vad.unload_model = AsyncMock()
        app.zoom_client._connected = False

        await app.stop()

        app.conversation_loop.stop.assert_called_once()
        app.audio_capture.stop.assert_called_once()
        app.audio_sender.stop.assert_called_once()
        app.stt.unload_model.assert_called_once()
        app.tts.unload_model.assert_called_once()
        app.vad.unload_model.assert_called_once()
        assert app.is_running is False

    @pytest.mark.asyncio
    async def test_stop_leaves_meeting_if_connected(
        self, settings: Settings
    ) -> None:
        """stop() should leave the meeting if connected."""
        app = ZoomAutoApp(settings)

        app.conversation_loop.stop = AsyncMock()
        app.audio_capture.stop = AsyncMock()
        app.audio_sender.stop = AsyncMock()
        app.stt.unload_model = AsyncMock()
        app.tts.unload_model = AsyncMock()
        app.vad.unload_model = AsyncMock()
        app.zoom_client._connected = True
        app.zoom_client.leave = AsyncMock()

        await app.stop()

        app.zoom_client.leave.assert_called_once()


class TestZoomAutoAppEventWiring:
    """Tests for event handler wiring."""

    def test_participant_joined_event_updates_context(
        self, settings: Settings
    ) -> None:
        """Participant joined should update context and audio capture."""
        app = ZoomAutoApp(settings)

        # Simulate participant join event
        participant = ParticipantInfo(
            user_id=42, display_name="Bob", is_host=False
        )

        app.event_handler.emit(
            MeetingEvent.PARTICIPANT_JOINED,
            {"participant": participant},
        )

        # Verify the speaker name was registered
        assert app.audio_capture._speaker_names.get(42) == "Bob"

    def test_speaker_changed_event_triggers_turn_manager(
        self, settings: Settings
    ) -> None:
        """Speaker changed should notify turn manager."""
        app = ZoomAutoApp(settings)

        assert app.turn_manager.someone_speaking is False

        app.event_handler.emit(
            MeetingEvent.SPEAKER_CHANGED,
            {"user_id": 42},
        )

        assert app.turn_manager.someone_speaking is True
