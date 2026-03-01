"""Tests for the response engine: decision, generator, turn manager, and LLM providers.

Tests cover:
- TriggerDetector: direct address, standup turn, cooldown, LLM fallback
- ResponseGenerator: persona-aware generation, response cleaning
- TurnManager: cooldown, speaking state, interruption, natural pause
- ClaudeProvider: message conversion, decision parsing
- OllamaProvider: message conversion, decision parsing
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from zoom_auto.config import ContextConfig, LLMConfig, ResponseConfig
from zoom_auto.context.manager import ContextManager
from zoom_auto.llm.base import LLMMessage, LLMResponse, LLMRole
from zoom_auto.llm.claude import ClaudeProvider
from zoom_auto.llm.ollama import OllamaProvider
from zoom_auto.persona.builder import PersonaProfile
from zoom_auto.response.decision import (
    ResponseDecision,
    TriggerDetector,
    TriggerReason,
)
from zoom_auto.response.generator import (
    GeneratedResponse,
    ResponseGenerator,
    _clean_response,
)
from zoom_auto.response.turn_manager import TurnManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def response_config() -> ResponseConfig:
    """Create a response config with short cooldown for testing."""
    return ResponseConfig(
        cooldown_seconds=5.0,
        trigger_threshold=0.6,
        max_consecutive=3,
    )


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create default LLM config."""
    return LLMConfig()


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM provider for testing."""
    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            text="I think we should focus on the API changes first.",
            model="test-model",
            usage_input_tokens=100,
            usage_output_tokens=20,
        )
    )
    llm.decide = AsyncMock(return_value=(True, 0.85))
    llm.is_available = AsyncMock(return_value=True)
    return llm


@pytest.fixture
def trigger_detector(
    response_config: ResponseConfig, mock_llm: MagicMock
) -> TriggerDetector:
    """Create a trigger detector with mock LLM."""
    return TriggerDetector(config=response_config, llm=mock_llm)


@pytest.fixture
def context_manager() -> ContextManager:
    """Create a context manager for testing."""
    return ContextManager(config=ContextConfig())


@pytest.fixture
def persona() -> PersonaProfile:
    """Create a test persona profile."""
    return PersonaProfile(
        name="Sam",
        formality=0.4,
        verbosity=0.5,
        technical_depth=0.6,
        assertiveness=0.6,
        greeting_style="Hey everyone",
        agreement_style="Sounds good to me",
        avg_response_words=40,
    )


@pytest.fixture
def response_generator(
    mock_llm: MagicMock,
    context_manager: ContextManager,
    persona: PersonaProfile,
) -> ResponseGenerator:
    """Create a response generator with mock LLM and persona."""
    return ResponseGenerator(
        llm=mock_llm,
        context_manager=context_manager,
        persona=persona,
    )


@pytest.fixture
def turn_manager(response_config: ResponseConfig) -> TurnManager:
    """Create a turn manager for testing."""
    return TurnManager(config=response_config)


# ---------------------------------------------------------------------------
# TriggerDetector tests
# ---------------------------------------------------------------------------


class TestTriggerDetector:
    """Tests for the trigger detection engine."""

    @pytest.mark.asyncio
    async def test_empty_transcript(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Empty transcript should not trigger a response."""
        result = await trigger_detector.should_respond("", "Sam")
        assert result.should_respond is False
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_direct_address_full_name(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Bot should respond when directly addressed by full name."""
        transcript = "Alice: Hey AI Assistant, what do you think?"
        result = await trigger_detector.should_respond(
            transcript, "AI Assistant"
        )
        assert result.should_respond is True
        assert result.reason == TriggerReason.DIRECT_ADDRESS
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_direct_address_case_insensitive(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Direct address should be case-insensitive."""
        transcript = "Alice: hey sam, thoughts?"
        result = await trigger_detector.should_respond(
            transcript, "Sam"
        )
        assert result.should_respond is True
        assert result.reason == TriggerReason.DIRECT_ADDRESS

    @pytest.mark.asyncio
    async def test_direct_address_first_name(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Bot should respond when addressed by first name only."""
        transcript = "Alice: Sam, can you update us?"
        result = await trigger_detector.should_respond(
            transcript, "Sam French"
        )
        assert result.should_respond is True
        assert result.reason == TriggerReason.DIRECT_ADDRESS

    @pytest.mark.asyncio
    async def test_someone_speaking_blocks(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Should not respond if someone is currently speaking."""
        transcript = "Alice: Sam, what do you think?"
        result = await trigger_detector.should_respond(
            transcript, "Sam", someone_speaking=True
        )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_direct_address_overrides_cooldown(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Direct address should trigger even during cooldown."""
        transcript = "Alice: Hey Sam, can you answer this?"
        result = await trigger_detector.should_respond(
            transcript, "Sam", is_cooldown_active=True
        )
        assert result.should_respond is True
        assert result.reason == TriggerReason.DIRECT_ADDRESS

    @pytest.mark.asyncio
    async def test_cooldown_blocks_ambiguous(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Cooldown should block non-direct-address triggers."""
        transcript = "Alice: What does everyone think about the API?"
        result = await trigger_detector.should_respond(
            transcript, "Bot", is_cooldown_active=True
        )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_standup_turn(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Should respond when it is the bot's standup turn."""
        transcript = "Alice: Sam, your turn. What's your update?"
        result = await trigger_detector.should_respond(
            transcript, "Sam"
        )
        assert result.should_respond is True
        assert result.reason in (
            TriggerReason.STANDUP_TURN,
            TriggerReason.DIRECT_ADDRESS,
        )

    @pytest.mark.asyncio
    async def test_llm_fallback_yes(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """LLM should be called for ambiguous situations."""
        # mock_llm.decide returns (True, 0.85) by default
        transcript = "Alice: Has anyone looked into the caching issue?"
        result = await trigger_detector.should_respond(
            transcript, "Bot"
        )
        assert result.should_respond is True
        assert trigger_detector.llm.decide.called

    @pytest.mark.asyncio
    async def test_llm_fallback_no(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """LLM returning NO should prevent response."""
        trigger_detector.llm.decide = AsyncMock(
            return_value=(False, 0.9)
        )
        transcript = "Alice: Great, let's move on to the next item."
        result = await trigger_detector.should_respond(
            transcript, "Bot"
        )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_llm_low_confidence(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """LLM decision below threshold should not trigger."""
        trigger_detector.llm.decide = AsyncMock(
            return_value=(True, 0.3)
        )
        transcript = "Alice: I think we should do X."
        result = await trigger_detector.should_respond(
            transcript, "Bot"
        )
        assert result.should_respond is False

    @pytest.mark.asyncio
    async def test_llm_error_graceful(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """LLM error should default to no response."""
        trigger_detector.llm.decide = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        transcript = "Alice: What about the database?"
        result = await trigger_detector.should_respond(
            transcript, "Bot"
        )
        assert result.should_respond is False
        assert result.confidence == 0.0


class TestCheckDirectAddress:
    """Tests for the direct address checker."""

    @pytest.mark.asyncio
    async def test_name_at_start(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Name at start of utterance should be detected."""
        assert await trigger_detector.check_direct_address(
            "Sam, what do you think?", "Sam"
        )

    @pytest.mark.asyncio
    async def test_name_in_middle(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Name in middle of utterance should be detected."""
        assert await trigger_detector.check_direct_address(
            "What do you think, Sam?", "Sam"
        )

    @pytest.mark.asyncio
    async def test_no_match(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Should return False when name is not present."""
        assert not await trigger_detector.check_direct_address(
            "Let's discuss the API", "Sam"
        )

    @pytest.mark.asyncio
    async def test_empty_text(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Should return False for empty text."""
        assert not await trigger_detector.check_direct_address("", "Sam")

    @pytest.mark.asyncio
    async def test_empty_name(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Should return False for empty name."""
        assert not await trigger_detector.check_direct_address(
            "Hello", ""
        )

    @pytest.mark.asyncio
    async def test_partial_name_no_match(
        self, trigger_detector: TriggerDetector
    ) -> None:
        """Partial name embedded in word should not match (word boundary)."""
        assert not await trigger_detector.check_direct_address(
            "Let's sample the data", "Sam"
        )


# ---------------------------------------------------------------------------
# ResponseGenerator tests
# ---------------------------------------------------------------------------


class TestCleanResponse:
    """Tests for the response cleaning function."""

    def test_empty_string(self) -> None:
        assert _clean_response("") == ""

    def test_strips_markdown_bold(self) -> None:
        assert _clean_response("This is **bold** text") == "This is bold text"

    def test_strips_markdown_italic(self) -> None:
        assert _clean_response("This is *italic* text") == "This is italic text"

    def test_strips_headers(self) -> None:
        assert _clean_response("## Header\nContent") == "Header Content"

    def test_strips_bullets(self) -> None:
        result = _clean_response("- item one\n- item two")
        assert "item one" in result
        assert "-" not in result

    def test_strips_numbered_list(self) -> None:
        result = _clean_response("1. first\n2. second")
        assert "first" in result
        assert "1." not in result

    def test_strips_backticks(self) -> None:
        assert _clean_response("Use `git commit`") == "Use git commit"

    def test_strips_code_blocks(self) -> None:
        result = _clean_response("Before ```code here``` after")
        assert "code here" not in result
        assert "Before" in result

    def test_strips_surrounding_quotes(self) -> None:
        assert _clean_response('"Hello there"') == "Hello there"

    def test_collapses_whitespace(self) -> None:
        result = _clean_response("Hello   \n\n  world")
        assert result == "Hello world"


class TestResponseGenerator:
    """Tests for the response generator."""

    @pytest.mark.asyncio
    async def test_generate_basic(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Should generate a response from context."""
        result = await response_generator.generate()
        assert isinstance(result, GeneratedResponse)
        assert len(result.text) > 0
        assert result.persona_applied is True

    @pytest.mark.asyncio
    async def test_generate_with_trigger_context(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Should include trigger context in the prompt."""
        result = await response_generator.generate(
            trigger_context="Alice asked about the API deadline."
        )
        assert isinstance(result, GeneratedResponse)
        # Verify LLM was called with messages that include trigger
        call_args = response_generator.llm.generate.call_args
        messages = call_args.kwargs.get(
            "messages", call_args.args[0] if call_args.args else []
        )
        all_content = " ".join(m.content for m in messages)
        assert "API deadline" in all_content

    @pytest.mark.asyncio
    async def test_generate_without_persona(
        self, mock_llm: MagicMock, context_manager: ContextManager
    ) -> None:
        """Should work without a persona profile."""
        gen = ResponseGenerator(
            llm=mock_llm,
            context_manager=context_manager,
            persona=None,
        )
        result = await gen.generate()
        assert result.persona_applied is False
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_generate_cleans_markdown(
        self, mock_llm: MagicMock, context_manager: ContextManager
    ) -> None:
        """Should clean markdown from LLM output."""
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                text="**I think** we should:\n- Do X\n- Do Y",
                model="test",
                usage_output_tokens=15,
            )
        )
        gen = ResponseGenerator(
            llm=mock_llm, context_manager=context_manager
        )
        result = await gen.generate()
        assert "**" not in result.text
        assert "- " not in result.text

    @pytest.mark.asyncio
    async def test_generate_token_limit(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Should call LLM with max 200 tokens."""
        await response_generator.generate()
        call_kwargs = response_generator.llm.generate.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 200

    @pytest.mark.asyncio
    async def test_set_persona(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Should update the persona profile."""
        new_persona = PersonaProfile(name="Alice", formality=0.8)
        await response_generator.set_persona(new_persona)
        assert response_generator.persona.name == "Alice"

    def test_llm_temperature_normal(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Normal verbosity should use default temperature."""
        assert response_generator.llm_temperature == 0.7

    def test_llm_temperature_high_verbosity(
        self,
        mock_llm: MagicMock,
        context_manager: ContextManager,
    ) -> None:
        """High verbosity persona should use higher temperature."""
        persona = PersonaProfile(verbosity=0.8)
        gen = ResponseGenerator(
            llm=mock_llm,
            context_manager=context_manager,
            persona=persona,
        )
        assert gen.llm_temperature == 0.8


# ---------------------------------------------------------------------------
# TurnManager tests
# ---------------------------------------------------------------------------


class TestTurnManager:
    """Tests for the turn-taking manager."""

    def test_can_speak_initially(self, turn_manager: TurnManager) -> None:
        """Bot should be able to speak when freshly initialized."""
        assert turn_manager.can_speak() is True

    def test_cooldown_blocks(self, turn_manager: TurnManager) -> None:
        """After recording a response, cooldown should block."""
        turn_manager.record_response()
        assert turn_manager.can_speak() is False
        assert turn_manager.is_cooldown_active is True

    def test_cooldown_expires(self, turn_manager: TurnManager) -> None:
        """Cooldown should expire after the configured time."""
        turn_manager._last_response_time = time.time() - 10.0
        # Config has 5s cooldown, so 10s ago should be expired
        assert turn_manager.is_cooldown_active is False
        assert turn_manager.can_speak() is True

    def test_cooldown_remaining(self, turn_manager: TurnManager) -> None:
        """cooldown_remaining should decrease over time."""
        turn_manager.record_response()
        remaining = turn_manager.cooldown_remaining
        assert remaining > 0
        assert remaining <= turn_manager.config.cooldown_seconds

    def test_cooldown_remaining_zero_initially(
        self, turn_manager: TurnManager
    ) -> None:
        """cooldown_remaining should be 0 when no response recorded."""
        assert turn_manager.cooldown_remaining == 0.0

    def test_consecutive_limit(self, turn_manager: TurnManager) -> None:
        """Should block after max consecutive responses."""
        for _ in range(3):
            turn_manager._last_response_time = 0.0  # bypass cooldown
            turn_manager.record_response()
        turn_manager._last_response_time = 0.0  # bypass cooldown
        assert turn_manager.can_speak() is False

    def test_record_other_resets_consecutive(
        self, turn_manager: TurnManager
    ) -> None:
        """Other speaker resets the consecutive counter."""
        turn_manager.record_response()
        turn_manager.record_response()
        turn_manager.record_other_speaker()
        assert turn_manager._consecutive_count == 0

    def test_someone_speaking_blocks(
        self, turn_manager: TurnManager
    ) -> None:
        """Should not speak when someone else is talking."""
        turn_manager.on_speech_detected()
        assert turn_manager.can_speak() is False
        assert turn_manager.someone_speaking is True

    def test_silence_unblocks(self, turn_manager: TurnManager) -> None:
        """Silence detection should unblock speaking."""
        turn_manager.on_speech_detected()
        assert turn_manager.can_speak() is False
        turn_manager.on_silence_detected()
        assert turn_manager.someone_speaking is False
        assert turn_manager.can_speak() is True

    def test_bot_speaking_blocks(
        self, turn_manager: TurnManager
    ) -> None:
        """Should not speak when bot is already speaking."""
        turn_manager.mark_bot_speaking()
        assert turn_manager.can_speak() is False
        assert turn_manager.bot_speaking is True

    def test_mark_bot_done(self, turn_manager: TurnManager) -> None:
        """Marking bot done should allow new responses."""
        turn_manager.mark_bot_speaking()
        turn_manager.mark_bot_done()
        assert turn_manager.bot_speaking is False

    def test_should_interrupt(self, turn_manager: TurnManager) -> None:
        """Should detect interruption when bot is speaking and someone talks."""
        turn_manager.mark_bot_speaking()
        assert turn_manager.should_interrupt() is False
        turn_manager.on_speech_detected()
        assert turn_manager.should_interrupt() is True

    def test_should_not_interrupt_when_silent(
        self, turn_manager: TurnManager
    ) -> None:
        """Should not interrupt when nobody is talking."""
        turn_manager.mark_bot_speaking()
        assert turn_manager.should_interrupt() is False

    def test_should_not_interrupt_when_not_speaking(
        self, turn_manager: TurnManager
    ) -> None:
        """Should not interrupt when bot is not speaking."""
        turn_manager.on_speech_detected()
        assert turn_manager.should_interrupt() is False

    def test_natural_pause_range(
        self, turn_manager: TurnManager
    ) -> None:
        """Natural pause should be between 0.3 and 0.8 seconds."""
        for _ in range(50):
            pause = turn_manager.get_natural_pause()
            assert 0.3 <= pause <= 0.8

    def test_override_cooldown(
        self, turn_manager: TurnManager
    ) -> None:
        """Overriding cooldown should allow immediate speaking."""
        turn_manager.record_response()
        assert turn_manager.is_cooldown_active is True
        turn_manager.override_cooldown()
        assert turn_manager.is_cooldown_active is False
        assert turn_manager.can_speak() is True

    def test_reset(self, turn_manager: TurnManager) -> None:
        """Reset should clear all state."""
        turn_manager.record_response()
        turn_manager.on_speech_detected()
        turn_manager.mark_bot_speaking()
        turn_manager.reset()
        assert turn_manager.can_speak() is True
        assert turn_manager.someone_speaking is False
        assert turn_manager.bot_speaking is False
        assert turn_manager.cooldown_remaining == 0.0


# ---------------------------------------------------------------------------
# ClaudeProvider tests
# ---------------------------------------------------------------------------


class TestClaudeProvider:
    """Tests for the Claude LLM provider."""

    def test_convert_messages_system(self, llm_config: LLMConfig) -> None:
        """Should extract system message from message list."""
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="You are helpful."),
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
        system, api_msgs = ClaudeProvider._convert_messages(messages)
        assert system == "You are helpful."
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"

    def test_convert_messages_no_system(
        self, llm_config: LLMConfig
    ) -> None:
        """Should handle messages without system prompt."""
        messages = [
            LLMMessage(role=LLMRole.USER, content="Hello"),
            LLMMessage(role=LLMRole.ASSISTANT, content="Hi"),
        ]
        system, api_msgs = ClaudeProvider._convert_messages(messages)
        assert system == ""
        assert len(api_msgs) == 2

    def test_convert_messages_empty(
        self, llm_config: LLMConfig
    ) -> None:
        """Empty messages should add a default user message."""
        system, api_msgs = ClaudeProvider._convert_messages([])
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"

    def test_parse_decision_yes(self) -> None:
        """Should parse YES responses correctly."""
        decision, confidence = ClaudeProvider._parse_decision(
            "YES 0.85 directly addressed by name"
        )
        assert decision is True
        assert confidence == 0.85

    def test_parse_decision_no(self) -> None:
        """Should parse NO responses correctly."""
        decision, confidence = ClaudeProvider._parse_decision(
            "NO 0.92 someone else is talking"
        )
        assert decision is False
        assert confidence == 0.92

    def test_parse_decision_malformed(self) -> None:
        """Should handle malformed responses gracefully."""
        decision, confidence = ClaudeProvider._parse_decision(
            "maybe I should respond"
        )
        assert decision is False
        assert confidence == 0.5

    def test_parse_decision_yes_uppercase(self) -> None:
        """Should handle mixed case."""
        decision, _ = ClaudeProvider._parse_decision("Yes 0.7 good time")
        assert decision is True

    def test_is_available_no_key(self, llm_config: LLMConfig) -> None:
        """Should report unavailable without API key."""
        provider = ClaudeProvider(config=llm_config, api_key="")
        import asyncio
        assert asyncio.get_event_loop().run_until_complete(
            provider.is_available()
        ) is False

    @pytest.mark.asyncio
    async def test_generate_calls_api(
        self, llm_config: LLMConfig
    ) -> None:
        """Should call Anthropic API with correct parameters."""
        provider = ClaudeProvider(
            config=llm_config, api_key="test-key"
        )

        # Mock the anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.model = "claude-sonnet"
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 10
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=mock_response
        )
        provider._client = mock_client

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="Be helpful"),
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
        result = await provider.generate(messages, max_tokens=200)

        assert result.text == "Test response"
        assert result.usage_input_tokens == 50
        assert result.usage_output_tokens == 10
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_decide_calls_api(
        self, llm_config: LLMConfig
    ) -> None:
        """Should call Haiku for quick decisions."""
        provider = ClaudeProvider(
            config=llm_config, api_key="test-key"
        )

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="YES 0.90 question asked to group")
        ]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=mock_response
        )
        provider._client = mock_client

        decision, confidence = await provider.decide(
            "Should I speak?", context="Alice asked a question"
        )
        assert decision is True
        assert confidence == 0.90
        # Should use the decision model (Haiku)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == llm_config.decision_model


# ---------------------------------------------------------------------------
# OllamaProvider tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    """Tests for the Ollama LLM provider."""

    def test_convert_messages(self, llm_config: LLMConfig) -> None:
        """Should convert messages to Ollama format."""
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content="Be helpful"),
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
        result = OllamaProvider._convert_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_parse_decision_yes(self) -> None:
        """Should parse YES responses."""
        decision, confidence = OllamaProvider._parse_decision(
            "YES 0.80 question needs answer"
        )
        assert decision is True
        assert confidence == 0.80

    def test_parse_decision_no(self) -> None:
        """Should parse NO responses."""
        decision, confidence = OllamaProvider._parse_decision(
            "NO 0.95 not relevant"
        )
        assert decision is False
        assert confidence == 0.95

    @pytest.mark.asyncio
    async def test_generate_calls_api(
        self, llm_config: LLMConfig
    ) -> None:
        """Should call Ollama chat API."""
        provider = OllamaProvider(config=llm_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "message": {"content": "Ollama response text"},
            "model": "llama3.1:8b",
            "prompt_eval_count": 30,
            "eval_count": 15,
            "done_reason": "stop",
        })

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        messages = [
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
        result = await provider.generate(messages, max_tokens=100)

        assert result.text == "Ollama response text"
        assert result.model == "llama3.1:8b"
        assert result.usage_input_tokens == 30
        assert result.usage_output_tokens == 15
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_available_with_model(
        self, llm_config: LLMConfig
    ) -> None:
        """Should check if the configured model is available."""
        provider = OllamaProvider(config=llm_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral:7b"},
            ]
        })

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_without_model(
        self, llm_config: LLMConfig
    ) -> None:
        """Should return False if model is not available."""
        provider = OllamaProvider(config=llm_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "models": [{"name": "mistral:7b"}]
        })

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_connection_error(
        self, llm_config: LLMConfig
    ) -> None:
        """Should return False when Ollama is not reachable."""
        provider = OllamaProvider(config=llm_config)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(
            side_effect=ConnectionError("not running")
        )
        provider._client = mock_client

        assert await provider.is_available() is False


# ---------------------------------------------------------------------------
# ResponseDecision dataclass tests
# ---------------------------------------------------------------------------


class TestResponseDecision:
    """Tests for the ResponseDecision dataclass."""

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        d = ResponseDecision(should_respond=True, confidence=0.9)
        assert d.reason == TriggerReason.NONE
        assert d.context_snippet == ""

    def test_all_fields(self) -> None:
        """Should store all fields correctly."""
        d = ResponseDecision(
            should_respond=True,
            confidence=0.95,
            reason=TriggerReason.DIRECT_ADDRESS,
            context_snippet="Hey Sam",
        )
        assert d.should_respond is True
        assert d.confidence == 0.95
        assert d.reason == TriggerReason.DIRECT_ADDRESS
        assert d.context_snippet == "Hey Sam"


# ---------------------------------------------------------------------------
# TriggerReason enum tests
# ---------------------------------------------------------------------------


class TestTriggerReason:
    """Tests for the TriggerReason enum."""

    def test_values(self) -> None:
        """All expected trigger reasons should exist."""
        assert TriggerReason.DIRECT_ADDRESS == "direct_address"
        assert TriggerReason.QUESTION_ASKED == "question_asked"
        assert TriggerReason.TOPIC_EXPERTISE == "topic_expertise"
        assert TriggerReason.LONG_SILENCE == "long_silence"
        assert TriggerReason.STANDUP_TURN == "standup_turn"
        assert TriggerReason.NONE == "none"

    def test_string_comparison(self) -> None:
        """StrEnum values should compare to strings."""
        assert TriggerReason.DIRECT_ADDRESS == "direct_address"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestResponseEngineIntegration:
    """Integration tests for the response engine components."""

    @pytest.mark.asyncio
    async def test_full_trigger_to_response_flow(
        self,
        mock_llm: MagicMock,
        response_config: ResponseConfig,
    ) -> None:
        """Test the complete flow from trigger to response."""
        # Setup
        context_manager = ContextManager()
        await context_manager.add_transcript(
            "Alice", "Hey Sam, what do you think about the API?"
        )

        detector = TriggerDetector(config=response_config, llm=mock_llm)
        turn_mgr = TurnManager(config=response_config)
        generator = ResponseGenerator(
            llm=mock_llm, context_manager=context_manager
        )

        # Step 1: Check if should respond
        transcript = context_manager.transcript.format_recent(10)
        decision = await detector.should_respond(
            transcript,
            bot_name="Sam",
            is_cooldown_active=turn_mgr.is_cooldown_active,
            someone_speaking=turn_mgr.someone_speaking,
        )

        assert decision.should_respond is True
        assert decision.reason == TriggerReason.DIRECT_ADDRESS

        # Step 2: Check turn manager
        assert turn_mgr.can_speak() is True

        # Step 3: Generate response
        response = await generator.generate(
            trigger_context=decision.context_snippet
        )
        assert len(response.text) > 0

        # Step 4: Record that we spoke
        turn_mgr.mark_bot_speaking()
        assert turn_mgr.bot_speaking is True
        turn_mgr.mark_bot_done()
        turn_mgr.record_response()
        assert turn_mgr.is_cooldown_active is True

    @pytest.mark.asyncio
    async def test_cooldown_prevents_double_response(
        self,
        mock_llm: MagicMock,
        response_config: ResponseConfig,
    ) -> None:
        """Bot should not respond twice in a row during cooldown."""
        detector = TriggerDetector(config=response_config, llm=mock_llm)
        turn_mgr = TurnManager(config=response_config)

        # First response
        turn_mgr.record_response()

        # Second attempt should be blocked by cooldown
        transcript = "Alice: What do others think?"
        decision = await detector.should_respond(
            transcript,
            bot_name="Bot",
            is_cooldown_active=turn_mgr.is_cooldown_active,
        )
        assert decision.should_respond is False

    @pytest.mark.asyncio
    async def test_interruption_handling(
        self,
        response_config: ResponseConfig,
    ) -> None:
        """Bot should detect interruption and stop."""
        turn_mgr = TurnManager(config=response_config)

        # Bot starts speaking
        turn_mgr.mark_bot_speaking()
        assert turn_mgr.should_interrupt() is False

        # Someone starts talking
        turn_mgr.on_speech_detected()
        assert turn_mgr.should_interrupt() is True

        # Bot stops
        turn_mgr.mark_bot_done()
        assert turn_mgr.bot_speaking is False
