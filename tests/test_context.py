"""Tests for the context engine: transcript, speaker tracker, meeting state, and context manager.

Tests cover:
- TranscriptAccumulator: adding entries, windowing, formatting, clearing
- SpeakerTracker: registration, active speaker, utterance recording, lookup
- MeetingState: participants, topics, decisions, action items, serialization
- ContextManager: transcript ingestion, prompt building, summarization, reset
- estimate_tokens: token estimation heuristic
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from zoom_auto.config import ContextConfig
from zoom_auto.context.manager import ContextManager, ContextWindow, estimate_tokens
from zoom_auto.context.meeting_state import MeetingState
from zoom_auto.context.speaker_tracker import SpeakerTracker
from zoom_auto.context.transcript import TranscriptAccumulator
from zoom_auto.llm.base import LLMResponse, LLMRole

# --- Fixtures ---


@pytest.fixture
def config() -> ContextConfig:
    """Create a default ContextConfig."""
    return ContextConfig()


@pytest.fixture
def small_config() -> ContextConfig:
    """Create a config with small windows for testing."""
    return ContextConfig(
        max_window_tokens=500,
        summarize_at=300,
        max_action_items=5,
        verbatim_window_seconds=60,
        summary_interval_seconds=30,
        max_history_tokens=200,
    )


@pytest.fixture
def transcript() -> TranscriptAccumulator:
    """Create an empty transcript accumulator."""
    return TranscriptAccumulator()


@pytest.fixture
def speaker_tracker() -> SpeakerTracker:
    """Create an empty speaker tracker."""
    return SpeakerTracker()


@pytest.fixture
def meeting_state(config: ContextConfig) -> MeetingState:
    """Create a meeting state instance."""
    return MeetingState(config=config)


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            text="Summary: The team discussed project updates and deadlines.",
            model="test-model",
            usage_input_tokens=50,
            usage_output_tokens=20,
        )
    )
    llm.decide = AsyncMock(return_value=(True, 0.9))
    llm.is_available = AsyncMock(return_value=True)
    return llm


@pytest.fixture
def context_manager(config: ContextConfig) -> ContextManager:
    """Create a context manager without LLM (for basic testing)."""
    return ContextManager(config=config)


@pytest.fixture
def context_manager_with_llm(config: ContextConfig, mock_llm: MagicMock) -> ContextManager:
    """Create a context manager with a mock LLM."""
    return ContextManager(config=config, llm=mock_llm)


# --- estimate_tokens ---


class TestEstimateTokens:
    """Tests for the token estimation function."""

    def test_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        """Short text should return at least 1 token."""
        assert estimate_tokens("Hi") >= 1

    def test_known_length(self) -> None:
        """100 characters should estimate ~25 tokens."""
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_longer_text(self) -> None:
        """400 characters should estimate ~100 tokens."""
        text = "word " * 80  # 400 chars
        assert estimate_tokens(text) == 100

    def test_proportional(self) -> None:
        """Longer text should estimate more tokens."""
        short = estimate_tokens("Hello")
        long = estimate_tokens("Hello, how are you doing today? This is a longer text.")
        assert long > short


# --- TranscriptAccumulator ---


class TestTranscriptAccumulator:
    """Tests for the transcript accumulator."""

    def test_add_entry(self, transcript: TranscriptAccumulator) -> None:
        """Adding an entry should increment the count."""
        entry = transcript.add("Alice", "Hello everyone")
        assert len(transcript) == 1
        assert entry.speaker == "Alice"
        assert entry.text == "Hello everyone"
        assert entry.segment_id == 1

    def test_add_multiple(self, transcript: TranscriptAccumulator) -> None:
        """Adding multiple entries should assign sequential IDs."""
        transcript.add("Alice", "Hello")
        transcript.add("Bob", "Hi there")
        transcript.add("Alice", "How are you?")
        assert len(transcript) == 3
        entries = transcript.entries
        assert entries[0].segment_id == 1
        assert entries[1].segment_id == 2
        assert entries[2].segment_id == 3

    def test_add_with_timestamp(self, transcript: TranscriptAccumulator) -> None:
        """Adding with explicit timestamp should use that timestamp."""
        ts = datetime(2026, 3, 1, 10, 30, 0)
        entry = transcript.add("Alice", "Hello", timestamp=ts)
        assert entry.timestamp == ts

    def test_add_with_confidence(self, transcript: TranscriptAccumulator) -> None:
        """Adding with confidence should store the confidence."""
        entry = transcript.add("Alice", "Hello", confidence=0.95)
        assert entry.confidence == 0.95

    def test_recent(self, transcript: TranscriptAccumulator) -> None:
        """Recent should return the last N entries."""
        for i in range(10):
            transcript.add(f"Speaker{i}", f"Message {i}")
        recent = transcript.recent(3)
        assert len(recent) == 3
        assert recent[0].text == "Message 7"
        assert recent[2].text == "Message 9"

    def test_recent_fewer_than_n(self, transcript: TranscriptAccumulator) -> None:
        """Recent should return all entries if fewer than N exist."""
        transcript.add("Alice", "Only one")
        recent = transcript.recent(10)
        assert len(recent) == 1

    def test_get_window(self, transcript: TranscriptAccumulator) -> None:
        """get_window should return entries within the time window."""
        now = datetime.now()
        old = now - timedelta(seconds=120)
        recent = now - timedelta(seconds=30)

        transcript.add("Alice", "Old message", timestamp=old)
        transcript.add("Bob", "Recent message", timestamp=recent)
        transcript.add("Carol", "Very recent", timestamp=now)

        window = transcript.get_window(60, reference_time=now)
        assert len(window) == 2
        assert window[0].speaker == "Bob"
        assert window[1].speaker == "Carol"

    def test_get_before(self, transcript: TranscriptAccumulator) -> None:
        """get_before should return entries older than the window."""
        now = datetime.now()
        old = now - timedelta(seconds=120)
        recent = now - timedelta(seconds=30)

        transcript.add("Alice", "Old message", timestamp=old)
        transcript.add("Bob", "Recent message", timestamp=recent)

        before = transcript.get_before(60, reference_time=now)
        assert len(before) == 1
        assert before[0].speaker == "Alice"

    def test_remove_before(self, transcript: TranscriptAccumulator) -> None:
        """remove_before should delete old entries."""
        now = datetime.now()
        old = now - timedelta(seconds=120)

        transcript.add("Alice", "Old", timestamp=old)
        transcript.add("Bob", "New", timestamp=now)

        removed = transcript.remove_before(now - timedelta(seconds=60))
        assert removed == 1
        assert len(transcript) == 1
        assert transcript.entries[0].speaker == "Bob"

    def test_format_recent(self, transcript: TranscriptAccumulator) -> None:
        """format_recent should produce 'Speaker (HH:MM): text' format."""
        ts = datetime(2026, 3, 1, 14, 30, 0)
        transcript.add("Alice", "Hello", timestamp=ts)
        formatted = transcript.format_recent(1)
        assert "Alice (14:30): Hello" in formatted

    def test_format_window(self, transcript: TranscriptAccumulator) -> None:
        """format_window should format entries within the time window."""
        now = datetime.now()
        transcript.add("Alice", "Recent talk", timestamp=now)
        formatted = transcript.format_window(60, reference_time=now)
        assert "Alice" in formatted
        assert "Recent talk" in formatted

    def test_get_plain_text(self, transcript: TranscriptAccumulator) -> None:
        """get_plain_text should return 'Speaker: text' lines."""
        transcript.add("Alice", "Hello")
        transcript.add("Bob", "Hi")
        plain = transcript.get_plain_text()
        assert "Alice: Hello" in plain
        assert "Bob: Hi" in plain

    def test_get_plain_text_specific_entries(self, transcript: TranscriptAccumulator) -> None:
        """get_plain_text should accept specific entries."""
        transcript.add("Alice", "Hello")
        transcript.add("Bob", "Hi")
        entries = [transcript.entries[0]]
        plain = transcript.get_plain_text(entries)
        assert "Alice: Hello" in plain
        assert "Bob" not in plain

    def test_clear(self, transcript: TranscriptAccumulator) -> None:
        """Clear should remove all entries and reset segment ID."""
        transcript.add("Alice", "Hello")
        transcript.add("Bob", "Hi")
        transcript.clear()
        assert len(transcript) == 0
        # After clear, next segment_id should reset
        entry = transcript.add("Carol", "Hey")
        assert entry.segment_id == 1

    def test_entries_returns_copy(self, transcript: TranscriptAccumulator) -> None:
        """entries property should return a copy."""
        transcript.add("Alice", "Hello")
        entries = transcript.entries
        entries.clear()
        assert len(transcript) == 1

    def test_entry_count(self, transcript: TranscriptAccumulator) -> None:
        """entry_count should match length."""
        transcript.add("Alice", "Hello")
        transcript.add("Bob", "Hi")
        assert transcript.entry_count == 2


# --- SpeakerTracker ---


class TestSpeakerTracker:
    """Tests for the speaker tracker."""

    def test_register_speaker(self, speaker_tracker: SpeakerTracker) -> None:
        """Registering a speaker should store their info."""
        info = speaker_tracker.register_speaker(1, "Alice")
        assert info.name == "Alice"
        assert info.speaker_id == 1
        assert speaker_tracker.speaker_count == 1

    def test_register_updates_name(self, speaker_tracker: SpeakerTracker) -> None:
        """Re-registering should update the name."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.register_speaker(1, "Alice Smith")
        assert speaker_tracker.get_name(1) == "Alice Smith"
        assert speaker_tracker.speaker_count == 1

    def test_get_name_unknown(self, speaker_tracker: SpeakerTracker) -> None:
        """Getting name for unknown ID should return 'Unknown'."""
        assert speaker_tracker.get_name(999) == "Unknown"

    def test_get_speaker(self, speaker_tracker: SpeakerTracker) -> None:
        """get_speaker should return SpeakerInfo or None."""
        speaker_tracker.register_speaker(1, "Alice")
        info = speaker_tracker.get_speaker(1)
        assert info is not None
        assert info.name == "Alice"
        assert speaker_tracker.get_speaker(999) is None

    def test_set_active(self, speaker_tracker: SpeakerTracker) -> None:
        """Setting active speaker should update is_active."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.register_speaker(2, "Bob")
        speaker_tracker.set_active(1)
        assert speaker_tracker.active_speaker is not None
        assert speaker_tracker.active_speaker.name == "Alice"
        assert speaker_tracker.get_speaker(1).is_active is True

    def test_set_active_deactivates_previous(self, speaker_tracker: SpeakerTracker) -> None:
        """Setting a new active speaker should deactivate the previous one."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.register_speaker(2, "Bob")
        speaker_tracker.set_active(1)
        speaker_tracker.set_active(2)
        assert speaker_tracker.get_speaker(1).is_active is False
        assert speaker_tracker.get_speaker(2).is_active is True
        assert speaker_tracker.active_speaker.name == "Bob"

    def test_clear_active(self, speaker_tracker: SpeakerTracker) -> None:
        """Clearing active should deactivate the current speaker."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.set_active(1)
        speaker_tracker.clear_active()
        assert speaker_tracker.active_speaker is None
        assert speaker_tracker.get_speaker(1).is_active is False

    def test_record_utterance(self, speaker_tracker: SpeakerTracker) -> None:
        """Recording utterance should increment count."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.record_utterance(1)
        speaker_tracker.record_utterance(1)
        info = speaker_tracker.get_speaker(1)
        assert info.utterance_count == 2

    def test_record_utterance_with_duration(self, speaker_tracker: SpeakerTracker) -> None:
        """Recording utterance with duration should add speaking time."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.record_utterance(1, duration=5.0)
        speaker_tracker.record_utterance(1, duration=3.0)
        info = speaker_tracker.get_speaker(1)
        assert info.total_speaking_time == pytest.approx(8.0)

    def test_record_utterance_unregistered(self, speaker_tracker: SpeakerTracker) -> None:
        """Recording utterance for unknown speaker should not crash."""
        speaker_tracker.record_utterance(999)  # Should just log warning

    def test_find_by_name(self, speaker_tracker: SpeakerTracker) -> None:
        """find_by_name should match case-insensitively."""
        speaker_tracker.register_speaker(1, "Alice Smith")
        result = speaker_tracker.find_by_name("alice smith")
        assert result is not None
        assert result.speaker_id == 1

    def test_find_by_name_not_found(self, speaker_tracker: SpeakerTracker) -> None:
        """find_by_name should return None if not found."""
        assert speaker_tracker.find_by_name("Nobody") is None

    def test_all_speakers(self, speaker_tracker: SpeakerTracker) -> None:
        """all_speakers should return all registered speakers."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.register_speaker(2, "Bob")
        speakers = speaker_tracker.all_speakers
        assert len(speakers) == 2

    def test_participant_names(self, speaker_tracker: SpeakerTracker) -> None:
        """participant_names should return name list."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.register_speaker(2, "Bob")
        names = speaker_tracker.participant_names
        assert "Alice" in names
        assert "Bob" in names

    def test_format_speaker_list_empty(self, speaker_tracker: SpeakerTracker) -> None:
        """format_speaker_list should handle empty state."""
        assert "No participants" in speaker_tracker.format_speaker_list()

    def test_format_speaker_list(self, speaker_tracker: SpeakerTracker) -> None:
        """format_speaker_list should list speakers with stats."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.record_utterance(1, duration=10.0)
        formatted = speaker_tracker.format_speaker_list()
        assert "Alice" in formatted
        assert "1 utterances" in formatted

    def test_reset(self, speaker_tracker: SpeakerTracker) -> None:
        """Reset should clear all speakers."""
        speaker_tracker.register_speaker(1, "Alice")
        speaker_tracker.set_active(1)
        speaker_tracker.reset()
        assert speaker_tracker.speaker_count == 0
        assert speaker_tracker.active_speaker is None

    def test_active_speaker_none_initially(self, speaker_tracker: SpeakerTracker) -> None:
        """Active speaker should be None initially."""
        assert speaker_tracker.active_speaker is None


# --- MeetingState ---


class TestMeetingState:
    """Tests for meeting state tracking."""

    def test_init_defaults(self, meeting_state: MeetingState) -> None:
        """Meeting state should have sensible defaults."""
        assert meeting_state.participants == []
        assert meeting_state.current_topic == ""
        assert meeting_state.action_items == []
        assert meeting_state.decisions == []

    def test_init_no_config(self) -> None:
        """MeetingState should work without explicit config."""
        ms = MeetingState()
        assert ms.config is not None

    def test_add_participant(self, meeting_state: MeetingState) -> None:
        """Adding participant should add to list."""
        meeting_state.add_participant("Alice")
        meeting_state.add_participant("Bob")
        assert meeting_state.participants == ["Alice", "Bob"]

    def test_add_participant_duplicate(self, meeting_state: MeetingState) -> None:
        """Adding duplicate participant should be a no-op."""
        meeting_state.add_participant("Alice")
        meeting_state.add_participant("Alice")
        assert len(meeting_state.participants) == 1

    def test_remove_participant(self, meeting_state: MeetingState) -> None:
        """Removing participant should remove from list."""
        meeting_state.add_participant("Alice")
        meeting_state.add_participant("Bob")
        meeting_state.remove_participant("Alice")
        assert meeting_state.participants == ["Bob"]

    def test_remove_nonexistent_participant(self, meeting_state: MeetingState) -> None:
        """Removing nonexistent participant should be safe."""
        meeting_state.remove_participant("Nobody")  # Should not raise

    def test_set_current_topic(self, meeting_state: MeetingState) -> None:
        """Setting current topic should also add to topics list."""
        meeting_state.current_topic = "Q2 Roadmap"
        assert meeting_state.current_topic == "Q2 Roadmap"
        assert "Q2 Roadmap" in meeting_state.topics

    @pytest.mark.asyncio
    async def test_add_action_item(self, meeting_state: MeetingState) -> None:
        """Adding action item should store it."""
        item = await meeting_state.add_action_item("Send report", assignee="Alice")
        assert item is not None
        assert item.description == "Send report"
        assert item.assignee == "Alice"
        assert len(meeting_state.action_items) == 1

    @pytest.mark.asyncio
    async def test_add_action_item_limit(self) -> None:
        """Adding beyond limit should return None."""
        config = ContextConfig(max_action_items=2)
        ms = MeetingState(config=config)
        await ms.add_action_item("Item 1")
        await ms.add_action_item("Item 2")
        result = await ms.add_action_item("Item 3")
        assert result is None
        assert len(ms.action_items) == 2

    @pytest.mark.asyncio
    async def test_add_decision(self, meeting_state: MeetingState) -> None:
        """Adding decision should store it."""
        decision = await meeting_state.add_decision(
            "Use PostgreSQL", context="Need relational queries"
        )
        assert decision.description == "Use PostgreSQL"
        assert decision.context == "Need relational queries"
        assert len(meeting_state.decisions) == 1

    @pytest.mark.asyncio
    async def test_add_topic(self, meeting_state: MeetingState) -> None:
        """Adding topic should store it."""
        await meeting_state.add_topic("Architecture")
        await meeting_state.add_topic("Timeline")
        assert meeting_state.topics == ["Architecture", "Timeline"]

    @pytest.mark.asyncio
    async def test_add_topic_duplicate(self, meeting_state: MeetingState) -> None:
        """Adding duplicate topic should be a no-op."""
        await meeting_state.add_topic("Architecture")
        await meeting_state.add_topic("Architecture")
        assert len(meeting_state.topics) == 1

    def test_set_agenda(self, meeting_state: MeetingState) -> None:
        """Setting agenda should store the items."""
        meeting_state.set_agenda(["Item 1", "Item 2", "Item 3"])
        assert "Item 1" in meeting_state.format_state()
        assert "Item 2" in meeting_state.format_state()

    @pytest.mark.asyncio
    async def test_format_state(self, meeting_state: MeetingState) -> None:
        """format_state should include all tracked info."""
        meeting_state.add_participant("Alice")
        meeting_state.add_participant("Bob")
        meeting_state.current_topic = "Architecture"
        await meeting_state.add_decision("Use React")
        await meeting_state.add_action_item("Create wireframes", assignee="Alice")

        state = meeting_state.format_state()
        assert "Alice" in state
        assert "Bob" in state
        assert "Architecture" in state
        assert "Use React" in state
        assert "Create wireframes" in state

    @pytest.mark.asyncio
    async def test_format_state_with_context(self, meeting_state: MeetingState) -> None:
        """format_state should include decision context if provided."""
        await meeting_state.add_decision("Use React", context="Better ecosystem")
        state = meeting_state.format_state()
        assert "Better ecosystem" in state

    @pytest.mark.asyncio
    async def test_to_dict(self, meeting_state: MeetingState) -> None:
        """to_dict should return complete serialized state."""
        meeting_state.add_participant("Alice")
        meeting_state.current_topic = "Testing"
        await meeting_state.add_decision("Use pytest")
        await meeting_state.add_action_item("Write tests", assignee="Bob")

        d = meeting_state.to_dict()
        assert "Alice" in d["participants"]
        assert d["current_topic"] == "Testing"
        assert len(d["decisions"]) == 1
        assert d["decisions"][0]["description"] == "Use pytest"
        assert len(d["action_items"]) == 1
        assert d["action_items"][0]["assignee"] == "Bob"
        assert "meeting_start_time" in d

    @pytest.mark.asyncio
    async def test_reset(self, meeting_state: MeetingState) -> None:
        """Reset should clear all state."""
        meeting_state.add_participant("Alice")
        meeting_state.current_topic = "Testing"
        await meeting_state.add_decision("Use pytest")
        await meeting_state.add_action_item("Write tests")

        await meeting_state.reset()
        assert meeting_state.participants == []
        assert meeting_state.current_topic == ""
        assert meeting_state.decisions == []
        assert meeting_state.action_items == []
        assert meeting_state.topics == []

    def test_meeting_start_time(self, meeting_state: MeetingState) -> None:
        """Meeting start time should be set on init."""
        assert meeting_state.meeting_start_time is not None

    def test_meeting_start_time_setter(self, meeting_state: MeetingState) -> None:
        """Meeting start time should be settable."""
        ts = datetime(2026, 1, 1, 9, 0, 0)
        meeting_state.meeting_start_time = ts
        assert meeting_state.meeting_start_time == ts

    def test_participants_returns_copy(self, meeting_state: MeetingState) -> None:
        """participants property should return a copy."""
        meeting_state.add_participant("Alice")
        p = meeting_state.participants
        p.clear()
        assert len(meeting_state.participants) == 1


# --- ContextWindow dataclass ---


class TestContextWindow:
    """Tests for the ContextWindow dataclass."""

    def test_defaults(self) -> None:
        """ContextWindow should have sensible defaults."""
        cw = ContextWindow()
        assert cw.summary == ""
        assert cw.recent_transcript == []
        assert cw.meeting_context == ""
        assert cw.total_tokens_estimate == 0


# --- ContextManager ---


class TestContextManager:
    """Tests for the context manager."""

    def test_init_defaults(self) -> None:
        """ContextManager should work with no arguments."""
        cm = ContextManager()
        assert cm.config is not None
        assert cm.transcript is not None
        assert cm.speaker_tracker is not None
        assert cm.meeting_state is not None

    def test_init_with_components(
        self,
        config: ContextConfig,
        transcript: TranscriptAccumulator,
        speaker_tracker: SpeakerTracker,
        meeting_state: MeetingState,
    ) -> None:
        """ContextManager should accept injected components."""
        cm = ContextManager(
            config=config,
            transcript=transcript,
            speaker_tracker=speaker_tracker,
            meeting_state=meeting_state,
        )
        assert cm.transcript is transcript
        assert cm.speaker_tracker is speaker_tracker
        assert cm.meeting_state is meeting_state

    @pytest.mark.asyncio
    async def test_add_transcript(self, context_manager: ContextManager) -> None:
        """add_transcript should add entry to transcript."""
        await context_manager.add_transcript("Alice", "Hello everyone")
        assert len(context_manager.transcript) == 1
        assert context_manager.transcript.entries[0].speaker == "Alice"

    @pytest.mark.asyncio
    async def test_add_transcript_empty_ignored(self, context_manager: ContextManager) -> None:
        """add_transcript should ignore empty text."""
        await context_manager.add_transcript("Alice", "")
        await context_manager.add_transcript("Alice", "   ")
        assert len(context_manager.transcript) == 0

    @pytest.mark.asyncio
    async def test_add_transcript_with_speaker_id(self, context_manager: ContextManager) -> None:
        """add_transcript with speaker_id should register the speaker."""
        await context_manager.add_transcript("Alice", "Hello", speaker_id=1)
        assert context_manager.speaker_tracker.get_name(1) == "Alice"
        info = context_manager.speaker_tracker.get_speaker(1)
        assert info.utterance_count == 1

    @pytest.mark.asyncio
    async def test_add_transcript_adds_participant(self, context_manager: ContextManager) -> None:
        """add_transcript should add speaker as participant."""
        await context_manager.add_transcript("Alice", "Hello")
        assert "Alice" in context_manager.meeting_state.participants

    @pytest.mark.asyncio
    async def test_add_transcript_with_timestamp(self, context_manager: ContextManager) -> None:
        """add_transcript should accept explicit timestamp."""
        ts = datetime(2026, 3, 1, 10, 0, 0)
        await context_manager.add_transcript("Alice", "Hello", timestamp=ts)
        entry = context_manager.transcript.entries[0]
        assert entry.timestamp == ts

    @pytest.mark.asyncio
    async def test_get_context(self, context_manager: ContextManager) -> None:
        """get_context should return a ContextWindow."""
        await context_manager.add_transcript("Alice", "Hello everyone")
        ctx = await context_manager.get_context()
        assert isinstance(ctx, ContextWindow)
        assert len(ctx.recent_transcript) == 1
        assert "Alice" in ctx.recent_transcript[0]
        assert "Hello everyone" in ctx.recent_transcript[0]

    @pytest.mark.asyncio
    async def test_get_context_meeting_state(self, context_manager: ContextManager) -> None:
        """get_context should include meeting state."""
        await context_manager.add_transcript("Alice", "Hello")
        context_manager.meeting_state.current_topic = "Planning"
        ctx = await context_manager.get_context()
        assert "Planning" in ctx.meeting_context

    @pytest.mark.asyncio
    async def test_get_context_token_estimate(self, context_manager: ContextManager) -> None:
        """get_context should estimate tokens."""
        await context_manager.add_transcript("Alice", "Hello " * 50)
        ctx = await context_manager.get_context()
        assert ctx.total_tokens_estimate > 0

    @pytest.mark.asyncio
    async def test_build_prompt(self, context_manager: ContextManager) -> None:
        """build_prompt should return LLM messages."""
        await context_manager.add_transcript("Alice", "Let's discuss the roadmap")
        await context_manager.get_context()

        messages = context_manager.build_prompt(
            system_prompt="You are a helpful meeting assistant.",
            meeting_metadata="Weekly standup meeting",
        )

        assert len(messages) == 2
        assert messages[0].role == LLMRole.SYSTEM
        assert "helpful meeting assistant" in messages[0].content
        assert "Weekly standup" in messages[0].content
        assert messages[1].role == LLMRole.USER
        assert "roadmap" in messages[1].content

    @pytest.mark.asyncio
    async def test_build_prompt_default_system(self, context_manager: ContextManager) -> None:
        """build_prompt without system_prompt should use default."""
        await context_manager.get_context()
        messages = context_manager.build_prompt()
        assert "AI meeting assistant" in messages[0].content

    @pytest.mark.asyncio
    async def test_build_prompt_empty_meeting(self, context_manager: ContextManager) -> None:
        """build_prompt with no transcript should include meeting state."""
        await context_manager.get_context()
        messages = context_manager.build_prompt()
        # Even an empty meeting has state (duration, etc.)
        assert "Meeting State" in messages[1].content
        assert "Meeting duration" in messages[1].content

    @pytest.mark.asyncio
    async def test_build_prompt_with_summary(self, context_manager: ContextManager) -> None:
        """build_prompt should include summaries if present."""
        context_manager._summaries.append("Earlier: discussed budget constraints.")
        await context_manager.add_transcript("Bob", "What about deadlines?")
        await context_manager.get_context()

        messages = context_manager.build_prompt()
        assert "Earlier Discussion" in messages[1].content
        assert "budget constraints" in messages[1].content
        assert "deadlines" in messages[1].content

    @pytest.mark.asyncio
    async def test_summarize_without_llm(self, context_manager: ContextManager) -> None:
        """summarize_if_needed without LLM should return False."""
        await context_manager.add_transcript("Alice", "Hello")
        result = await context_manager.summarize_if_needed()
        assert result is False

    @pytest.mark.asyncio
    async def test_summarize_with_llm(self, context_manager_with_llm: ContextManager) -> None:
        """summarize_if_needed with old entries should summarize them."""
        cm = context_manager_with_llm

        # Add entries that are outside the verbatim window
        old_time = datetime.now() - timedelta(seconds=300)
        cm.transcript.add("Alice", "Let's discuss Q2", timestamp=old_time)
        cm.transcript.add("Bob", "I agree", timestamp=old_time + timedelta(seconds=10))

        # Add a recent entry (inside verbatim window)
        cm.transcript.add("Carol", "What about Q3?", timestamp=datetime.now())

        result = await cm.summarize_if_needed()
        assert result is True
        assert len(cm.summaries) == 1
        assert "discussed" in cm.summaries[0].lower() or "summary" in cm.summaries[0].lower()

    @pytest.mark.asyncio
    async def test_summarize_no_old_entries(self, context_manager_with_llm: ContextManager) -> None:
        """summarize_if_needed with only recent entries should skip."""
        cm = context_manager_with_llm
        cm.transcript.add("Alice", "Hello", timestamp=datetime.now())
        result = await cm.summarize_if_needed()
        assert result is False

    @pytest.mark.asyncio
    async def test_summarize_already_summarized(
        self, context_manager_with_llm: ContextManager,
    ) -> None:
        """summarize_if_needed should not re-summarize already-summarized entries."""
        cm = context_manager_with_llm
        old_time = datetime.now() - timedelta(seconds=300)
        cm.transcript.add("Alice", "Old discussion", timestamp=old_time)
        cm.transcript.add("Bob", "Recent", timestamp=datetime.now())

        await cm.summarize_if_needed()
        initial_summaries = len(cm.summaries)

        # Summarize again -- should not create new summary since nothing new
        result = await cm.summarize_if_needed()
        assert result is False
        assert len(cm.summaries) == initial_summaries

    @pytest.mark.asyncio
    async def test_prune_summaries(self, mock_llm: MagicMock) -> None:
        """Pruning should consolidate old summaries when exceeding token limit."""
        config = ContextConfig(max_history_tokens=50)
        cm = ContextManager(config=config, llm=mock_llm)

        # Add many summaries that exceed the token limit
        for i in range(10):
            cm._summaries.append(f"Summary {i}: " + "discussion about various topics " * 5)

        await cm._prune_summaries()

        # Should have consolidated some summaries
        assert len(cm._summaries) < 10

    @pytest.mark.asyncio
    async def test_prune_summaries_under_limit(
        self, context_manager_with_llm: ContextManager,
    ) -> None:
        """Pruning should not happen when under token limit."""
        cm = context_manager_with_llm
        cm._summaries.append("Short summary")
        original_count = len(cm._summaries)
        await cm._prune_summaries()
        assert len(cm._summaries) == original_count

    @pytest.mark.asyncio
    async def test_reset(self, context_manager: ContextManager) -> None:
        """Reset should clear all state."""
        await context_manager.add_transcript("Alice", "Hello", speaker_id=1)
        context_manager.meeting_state.current_topic = "Testing"
        context_manager._summaries.append("Old summary")

        await context_manager.reset()

        assert len(context_manager.transcript) == 0
        assert context_manager.speaker_tracker.speaker_count == 0
        assert context_manager.meeting_state.current_topic == ""
        assert len(context_manager.summaries) == 0

    @pytest.mark.asyncio
    async def test_summaries_returns_copy(self, context_manager: ContextManager) -> None:
        """summaries property should return a copy."""
        context_manager._summaries.append("Summary 1")
        summaries = context_manager.summaries
        summaries.clear()
        assert len(context_manager._summaries) == 1

    @pytest.mark.asyncio
    async def test_verbatim_window_respected(self, context_manager: ContextManager) -> None:
        """Only entries within verbatim_window_seconds should appear in recent_transcript."""
        now = datetime.now()
        # Add entry outside the window (default 180s)
        old_time = now - timedelta(seconds=300)
        await context_manager.add_transcript("Alice", "Very old", timestamp=old_time)
        # Add entry inside the window
        await context_manager.add_transcript("Bob", "Recent", timestamp=now)

        ctx = await context_manager.get_context()
        texts = " ".join(ctx.recent_transcript)
        assert "Recent" in texts
        assert "Very old" not in texts

    @pytest.mark.asyncio
    async def test_multiple_speakers_prompt(self, context_manager: ContextManager) -> None:
        """Prompt should handle multiple speakers correctly."""
        now = datetime.now()
        await context_manager.add_transcript("Alice", "Hello everyone", timestamp=now, speaker_id=1)
        await context_manager.add_transcript(
            "Bob", "Hi Alice", timestamp=now + timedelta(seconds=1), speaker_id=2
        )
        await context_manager.add_transcript(
            "Carol", "Good morning", timestamp=now + timedelta(seconds=2), speaker_id=3
        )

        ctx = await context_manager.get_context()
        assert len(ctx.recent_transcript) == 3

        # All speakers should be participants
        participants = context_manager.meeting_state.participants
        assert "Alice" in participants
        assert "Bob" in participants
        assert "Carol" in participants

    @pytest.mark.asyncio
    async def test_summarize_llm_error(self, context_manager_with_llm: ContextManager) -> None:
        """Summarization should handle LLM errors gracefully."""
        cm = context_manager_with_llm
        cm._llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

        old_time = datetime.now() - timedelta(seconds=300)
        cm.transcript.add("Alice", "Discussion", timestamp=old_time)
        cm.transcript.add("Bob", "Recent", timestamp=datetime.now())

        result = await cm.summarize_if_needed()
        assert result is False
        assert len(cm.summaries) == 0


# --- Integration tests ---


class TestContextIntegration:
    """Integration tests verifying components work together."""

    @pytest.mark.asyncio
    async def test_full_meeting_flow(self) -> None:
        """Simulate a full meeting flow with all components."""
        config = ContextConfig(
            verbatim_window_seconds=60,
            summary_interval_seconds=10,
            max_history_tokens=500,
        )
        cm = ContextManager(config=config)

        # Start meeting
        now = datetime.now()
        cm.meeting_state.meeting_start_time = now

        # Add participants and transcript
        await cm.add_transcript("Alice", "Welcome to the standup", timestamp=now, speaker_id=1)
        await cm.add_transcript(
            "Bob",
            "I worked on the API yesterday",
            timestamp=now + timedelta(seconds=5),
            speaker_id=2,
        )
        await cm.add_transcript(
            "Alice",
            "Great, what about the database migration?",
            timestamp=now + timedelta(seconds=10),
            speaker_id=1,
        )
        await cm.add_transcript(
            "Carol",
            "I'll handle the migration today",
            timestamp=now + timedelta(seconds=15),
            speaker_id=3,
        )

        # Set topic and add decision
        cm.meeting_state.current_topic = "Sprint progress"
        await cm.meeting_state.add_decision("Carol will handle DB migration")
        await cm.meeting_state.add_action_item(
            "Complete database migration", assignee="Carol"
        )

        # Get context and build prompt
        ctx = await cm.get_context()
        assert len(ctx.recent_transcript) == 4
        assert "Sprint progress" in ctx.meeting_context

        messages = cm.build_prompt(
            system_prompt="You are Sam's AI assistant in meetings.",
        )
        assert len(messages) == 2
        assert "Sam's AI assistant" in messages[0].content

        # Verify all participants are tracked
        assert len(cm.meeting_state.participants) == 3
        assert cm.speaker_tracker.speaker_count == 3

    @pytest.mark.asyncio
    async def test_meeting_with_summarization(self, mock_llm: MagicMock) -> None:
        """Test meeting flow with summarization."""
        config = ContextConfig(
            verbatim_window_seconds=60,
            summary_interval_seconds=10,
            max_history_tokens=500,
        )
        cm = ContextManager(config=config, llm=mock_llm)

        now = datetime.now()

        # Add old entries (outside verbatim window)
        old_time = now - timedelta(seconds=120)
        for i in range(5):
            await cm.add_transcript(
                f"Speaker{i}",
                f"Old discussion point {i}",
                timestamp=old_time + timedelta(seconds=i * 10),
            )

        # Add recent entries
        for i in range(3):
            await cm.add_transcript(
                f"Speaker{i}",
                f"Recent point {i}",
                timestamp=now + timedelta(seconds=i),
            )

        # Force summarization
        result = await cm.summarize_if_needed()
        assert result is True

        # Get context - should have summary + recent
        ctx = await cm.get_context()
        assert ctx.summary != ""
        assert len(ctx.recent_transcript) == 3  # Only recent entries
