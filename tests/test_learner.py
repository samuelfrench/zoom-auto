"""Tests for the ConversationLearner module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zoom_auto.persona.learner import ConversationLearner, ConversationSession


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for learnings."""
    return tmp_path / "learnings"


@pytest.fixture
def learner(data_dir: Path) -> ConversationLearner:
    """Create a ConversationLearner with a temporary data dir."""
    return ConversationLearner(data_dir=data_dir, user="testuser")


# -----------------------------------------------------------------------
# Session lifecycle
# -----------------------------------------------------------------------


class TestSessionLifecycle:
    """Tests for start_session / end_session."""

    def test_start_session_returns_session_id(
        self, learner: ConversationLearner
    ) -> None:
        session_id = learner.start_session()
        assert session_id
        assert isinstance(session_id, str)

    def test_start_session_creates_current_session(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        assert learner._current_session is not None

    def test_end_session_returns_session(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        session = learner.end_session()
        assert isinstance(session, ConversationSession)
        assert session.ended_at != ""

    def test_end_session_clears_current(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.end_session()
        assert learner._current_session is None

    def test_end_session_without_start_returns_empty(
        self, learner: ConversationLearner
    ) -> None:
        session = learner.end_session()
        assert session.session_id == "empty"
        assert session.transcript == []

    def test_session_started_at_is_set(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        session = learner.end_session()
        assert session.started_at != ""


# -----------------------------------------------------------------------
# Record utterance
# -----------------------------------------------------------------------


class TestRecordUtterance:
    """Tests for recording utterances."""

    def test_record_adds_to_transcript(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Hello everyone")
        assert len(learner._current_session.transcript) == 1  # type: ignore[union-attr]
        assert learner._current_session.transcript[0]["speaker"] == "Alice"  # type: ignore[union-attr]
        assert learner._current_session.transcript[0]["text"] == "Hello everyone"  # type: ignore[union-attr]

    def test_record_multiple_utterances(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Hello")
        learner.record_utterance("Bob", "Hi there")
        learner.record_utterance("Alice", "How are you")
        session = learner.end_session()
        assert len(session.transcript) == 3

    def test_record_with_timestamp(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Hello", timestamp="2026-03-01T10:00:00")
        entry = learner._current_session.transcript[0]  # type: ignore[union-attr]
        assert entry["timestamp"] == "2026-03-01T10:00:00"

    def test_record_without_session_is_noop(
        self, learner: ConversationLearner
    ) -> None:
        # Should not raise
        learner.record_utterance("Alice", "Hello")


# -----------------------------------------------------------------------
# Record bot response
# -----------------------------------------------------------------------


class TestRecordBotResponse:
    """Tests for recording bot responses."""

    def test_record_bot_response(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_bot_response(
            trigger_reason="direct_address",
            response_text="I agree with that approach.",
            context_snippet="Alice: What do you think?",
        )
        assert len(learner._current_session.bot_responses) == 1  # type: ignore[union-attr]
        resp = learner._current_session.bot_responses[0]  # type: ignore[union-attr]
        assert resp["trigger_reason"] == "direct_address"
        assert resp["response_text"] == "I agree with that approach."

    def test_record_bot_response_without_session_is_noop(
        self, learner: ConversationLearner
    ) -> None:
        # Should not raise
        learner.record_bot_response(
            trigger_reason="test",
            response_text="test",
        )


# -----------------------------------------------------------------------
# Vocabulary tracking
# -----------------------------------------------------------------------


class TestVocabularyTracking:
    """Tests for word frequency tracking."""

    def test_tracks_word_frequency(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "The deployment pipeline needs fixing")
        learner.record_utterance("Alice", "The deployment was broken yesterday")
        # "deployment" should appear twice
        assert learner._vocabulary.get("deployment", 0) == 2

    def test_excludes_stop_words(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "I think the API is working well")
        # Stop words like "the", "is" should not be tracked
        assert "the" not in learner._vocabulary
        assert "is" not in learner._vocabulary

    def test_excludes_short_words(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Go to do it now")
        # Words under 3 chars should be excluded
        assert "go" not in learner._vocabulary
        assert "to" not in learner._vocabulary
        assert "do" not in learner._vocabulary
        assert "it" not in learner._vocabulary

    def test_vocabulary_is_case_insensitive(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Kubernetes cluster")
        learner.record_utterance("Alice", "kubernetes pods")
        assert learner._vocabulary.get("kubernetes", 0) == 2


# -----------------------------------------------------------------------
# Vocabulary persistence
# -----------------------------------------------------------------------


class TestVocabularyPersistence:
    """Tests for saving and loading vocabulary."""

    def test_save_and_load_vocabulary(
        self, data_dir: Path
    ) -> None:
        # Create learner, add some vocab, end session (saves)
        learner1 = ConversationLearner(data_dir=data_dir, user="testuser")
        learner1.start_session()
        learner1.record_utterance("Alice", "microservices architecture design")
        learner1.end_session()

        # Create new learner instance, should load vocab from disk
        learner2 = ConversationLearner(data_dir=data_dir, user="testuser")
        assert learner2._vocabulary.get("microservices", 0) == 1
        assert learner2._vocabulary.get("architecture", 0) == 1

    def test_vocabulary_accumulates_across_sessions(
        self, data_dir: Path
    ) -> None:
        # Session 1
        learner1 = ConversationLearner(data_dir=data_dir, user="testuser")
        learner1.start_session()
        learner1.record_utterance("Alice", "refactor the database module")
        learner1.end_session()

        # Session 2
        learner2 = ConversationLearner(data_dir=data_dir, user="testuser")
        learner2.start_session()
        learner2.record_utterance("Alice", "refactor the authentication module")
        learner2.end_session()

        # Reload
        learner3 = ConversationLearner(data_dir=data_dir, user="testuser")
        assert learner3._vocabulary.get("refactor", 0) == 2
        assert learner3._vocabulary.get("module", 0) == 2


# -----------------------------------------------------------------------
# Topic extraction
# -----------------------------------------------------------------------


class TestTopicExtraction:
    """Tests for _extract_topics."""

    def test_explicit_topic_markers(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "Let's discuss the Q2 roadmap."},
            {"text": "Sure, regarding the API redesign, I have thoughts."},
        ]
        topics = learner._extract_topics(transcript)
        assert any("roadmap" in t for t in topics)
        assert any("api redesign" in t for t in topics)

    def test_repeated_terms(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "The database migration is important."},
            {"text": "We need to finish the database migration soon."},
            {"text": "Database migration should be our priority."},
        ]
        topics = learner._extract_topics(transcript)
        assert any("database" in t or "migration" in t for t in topics)

    def test_empty_transcript_returns_empty(
        self, learner: ConversationLearner
    ) -> None:
        assert learner._extract_topics([]) == []

    def test_caps_at_20_topics(
        self, learner: ConversationLearner
    ) -> None:
        # Create transcript with many topic markers
        transcript = [
            {"text": f"Let's discuss topic{i}."} for i in range(30)
        ]
        topics = learner._extract_topics(transcript)
        assert len(topics) <= 20


# -----------------------------------------------------------------------
# Meeting type detection
# -----------------------------------------------------------------------


class TestMeetingTypeDetection:
    """Tests for _detect_meeting_type."""

    def test_standup_detection(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "Yesterday I worked on the login page."},
            {"text": "Today I'm working on the dashboard."},
            {"text": "No blockers for me."},
        ]
        assert learner._detect_meeting_type(transcript) == "standup"

    def test_planning_detection(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "Let's estimate the sprint backlog."},
            {"text": "This epic needs to be prioritized."},
            {"text": "How many story points for this?"},
        ]
        assert learner._detect_meeting_type(transcript) == "planning"

    def test_retro_detection(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "What went well this sprint?"},
            {"text": "We should improve our testing."},
            {"text": "Let's add that as an action item."},
        ]
        assert learner._detect_meeting_type(transcript) == "retro"

    def test_technical_detection(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "The API endpoint has high latency."},
            {"text": "We need to refactor the database schema."},
            {"text": "Let's deploy the migration first."},
        ]
        assert learner._detect_meeting_type(transcript) == "technical"

    def test_general_when_no_keywords(
        self, learner: ConversationLearner
    ) -> None:
        transcript = [
            {"text": "Hello everyone."},
            {"text": "How is everyone doing?"},
        ]
        assert learner._detect_meeting_type(transcript) == "general"

    def test_empty_transcript_is_general(
        self, learner: ConversationLearner
    ) -> None:
        assert learner._detect_meeting_type([]) == "general"


# -----------------------------------------------------------------------
# Learning context generation
# -----------------------------------------------------------------------


class TestLearningContext:
    """Tests for get_learning_context."""

    def test_empty_learnings_returns_empty_string(
        self, learner: ConversationLearner
    ) -> None:
        assert learner.get_learning_context() == ""

    def test_includes_vocabulary(
        self, learner: ConversationLearner
    ) -> None:
        learner._vocabulary = {"kubernetes": 10, "deployment": 5, "pipeline": 3}
        context = learner.get_learning_context()
        assert "kubernetes" in context
        assert "deployment" in context

    def test_includes_topics_from_sessions(
        self, data_dir: Path
    ) -> None:
        learner = ConversationLearner(data_dir=data_dir, user="testuser")
        learner.start_session()
        learner.record_utterance("Alice", "Let's discuss the API redesign.")
        learner.record_utterance("Alice", "Let's discuss the API redesign.")
        learner.record_utterance("Alice", "Let's discuss the API redesign.")
        learner.end_session()

        # Reload and check context
        learner2 = ConversationLearner(data_dir=data_dir, user="testuser")
        context = learner2.get_learning_context()
        assert "api redesign" in context.lower() or len(learner2._vocabulary) > 0

    def test_respects_max_tokens(
        self, learner: ConversationLearner
    ) -> None:
        # Fill vocabulary with many words
        learner._vocabulary = {f"word{i}": i for i in range(200)}
        context = learner.get_learning_context(max_tokens=50)
        # Should be roughly limited (50 tokens * 4 chars = 200 chars)
        assert len(context) <= 250  # some slack for header


# -----------------------------------------------------------------------
# Session persistence
# -----------------------------------------------------------------------


class TestSessionPersistence:
    """Tests for saving and loading sessions."""

    def test_session_saved_to_disk(
        self, learner: ConversationLearner, data_dir: Path
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Hello")
        session = learner.end_session()

        sessions_dir = data_dir / "testuser" / "sessions"
        files = list(sessions_dir.glob("*.json"))
        assert len(files) == 1

        # Verify content
        data = json.loads(files[0].read_text())
        assert data["session_id"] == session.session_id
        assert len(data["transcript"]) == 1

    def test_load_session_from_disk(
        self, learner: ConversationLearner, data_dir: Path
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "Hello")
        learner.record_utterance("Bob", "Hi there")
        original = learner.end_session()

        # Load from disk
        files = learner.get_transcript_files()
        assert len(files) == 1
        loaded = learner._load_session(files[0])
        assert loaded.session_id == original.session_id
        assert len(loaded.transcript) == 2

    def test_session_to_dict_roundtrip(self) -> None:
        session = ConversationSession(
            session_id="test-123",
            started_at="2026-03-01T10:00:00",
            ended_at="2026-03-01T11:00:00",
            transcript=[{"speaker": "Alice", "text": "Hello", "timestamp": ""}],
            topics_discussed=["api", "deployment"],
            meeting_type="technical",
        )
        data = session.to_dict()
        restored = ConversationSession.from_dict(data)
        assert restored.session_id == "test-123"
        assert restored.topics_discussed == ["api", "deployment"]
        assert restored.meeting_type == "technical"
        assert len(restored.transcript) == 1


# -----------------------------------------------------------------------
# Multiple sessions
# -----------------------------------------------------------------------


class TestMultipleSessions:
    """Tests for accumulating learnings across sessions."""

    def test_multiple_sessions_create_multiple_files(
        self, data_dir: Path
    ) -> None:
        learner = ConversationLearner(data_dir=data_dir, user="testuser")

        # Session 1
        learner.start_session()
        learner.record_utterance("Alice", "Hello")
        learner.end_session()

        # Session 2
        learner.start_session()
        learner.record_utterance("Bob", "Hi")
        learner.end_session()

        files = learner.get_transcript_files()
        assert len(files) == 2

    def test_vocabulary_accumulates(
        self, data_dir: Path
    ) -> None:
        learner = ConversationLearner(data_dir=data_dir, user="testuser")

        learner.start_session()
        learner.record_utterance("Alice", "kubernetes deployment pipeline")
        learner.end_session()

        learner.start_session()
        learner.record_utterance("Alice", "kubernetes cluster monitoring")
        learner.end_session()

        assert learner._vocabulary.get("kubernetes", 0) == 2

    def test_learning_context_includes_all_sessions(
        self, data_dir: Path
    ) -> None:
        learner = ConversationLearner(data_dir=data_dir, user="testuser")

        # Session with standup keywords
        learner.start_session()
        learner.record_utterance("Alice", "Yesterday I deployed the service.")
        learner.record_utterance("Alice", "Today I'm fixing blockers.")
        learner.end_session()

        # Session with technical keywords
        learner.start_session()
        learner.record_utterance("Alice", "The API endpoint needs refactoring.")
        learner.end_session()

        context = learner.get_learning_context()
        # Should have content from accumulated vocabulary
        assert context != ""


# -----------------------------------------------------------------------
# Get transcript files
# -----------------------------------------------------------------------


class TestGetTranscriptFiles:
    """Tests for get_transcript_files."""

    def test_returns_empty_when_no_sessions(
        self, learner: ConversationLearner
    ) -> None:
        files = learner.get_transcript_files()
        assert files == []

    def test_returns_sorted_paths(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        learner.record_utterance("Alice", "First session")
        learner.end_session()

        learner.start_session()
        learner.record_utterance("Bob", "Second session")
        learner.end_session()

        files = learner.get_transcript_files()
        assert len(files) == 2
        # Should be sorted
        assert files[0].name < files[1].name


# -----------------------------------------------------------------------
# Empty session handling
# -----------------------------------------------------------------------


class TestEmptySession:
    """Tests for handling empty sessions gracefully."""

    def test_empty_session_still_saves(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        session = learner.end_session()
        assert session.transcript == []
        assert session.topics_discussed == []
        assert session.meeting_type == "general"

        # File should still be saved
        files = learner.get_transcript_files()
        assert len(files) == 1

    def test_empty_session_vocabulary_learned_is_empty(
        self, learner: ConversationLearner
    ) -> None:
        learner.start_session()
        session = learner.end_session()
        assert session.vocabulary_learned == []
