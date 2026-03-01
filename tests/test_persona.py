"""Tests for the persona system: style analyzer, builder, and source analyzers.

Tests cover:
- StyleAnalyzer: metrics computation, vocabulary extraction, filler detection
- PersonaProfile: TOML serialization (save/load round-trip)
- PersonaBuilder: sample merging, prompt generation, weighting
- TranscriptAnalyzer: VTT, SRT, and plain text parsing
- SlackAnalyzer: Slack export JSON parsing and pattern extraction
- WritingAnalyzer: email and document parsing
- ConversationAnalyzer: conversation analysis and pattern extraction
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zoom_auto.persona.builder import (
    PersonaBuilder,
    PersonaProfile,
    TextSample,
)
from zoom_auto.persona.sources.conversation import (
    ConversationAnalyzer,
    ConversationTurn,
)
from zoom_auto.persona.sources.slack import SlackAnalyzer
from zoom_auto.persona.sources.transcript import TranscriptAnalyzer
from zoom_auto.persona.sources.writing import WritingAnalyzer
from zoom_auto.persona.style_analyzer import (
    StyleAnalyzer,
    StyleMetrics,
    VocabularyProfile,
)

# --- Fixtures ---


@pytest.fixture
def analyzer() -> StyleAnalyzer:
    """Create a StyleAnalyzer instance."""
    return StyleAnalyzer()


@pytest.fixture
def builder() -> PersonaBuilder:
    """Create a PersonaBuilder instance."""
    return PersonaBuilder()


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample text for style analysis."""
    return [
        "I think we should deploy the API changes today. "
        "The pipeline looks good and all tests pass.",
        "Yeah, um, I basically worked on the database migration "
        "yesterday. You know, the schema changes were kind of tricky.",
        "Let me check the sprint backlog. We have three blockers "
        "that need attention before the release.",
    ]


@pytest.fixture
def formal_text() -> str:
    """Formal writing sample."""
    return (
        "Therefore, I would like to propose that we subsequently "
        "review the aforementioned infrastructure changes. "
        "Furthermore, the implementation should be accordingly "
        "adjusted to meet the requirements. However, we must "
        "consider the implications carefully."
    )


@pytest.fixture
def casual_text() -> str:
    """Casual writing sample."""
    return (
        "Hey yeah so I totally wanna get this stuff done today. "
        "It's gonna be awesome! Cool cool, let's do it. "
        "Btw I dunno if we need that thing anymore lol."
    )


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """Create a temp directory with transcript files."""
    d = tmp_path / "transcripts"
    d.mkdir()

    # Plain text transcript
    plain = d / "meeting1.txt"
    plain.write_text(
        "Alice: Hello everyone, let's get started.\n"
        "Bob: Hi Alice, I have an update on the API.\n"
        "Alice: Great, go ahead.\n"
        "Bob: The deployment went smoothly yesterday.\n"
        "Carol: I have a question about the database.\n",
    )

    # VTT transcript
    vtt = d / "meeting2.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:05.000\n"
        "<v Alice>Welcome to the standup.\n\n"
        "00:00:06.000 --> 00:00:10.000\n"
        "<v Bob>I worked on the migration.\n\n"
        "00:00:11.000 --> 00:00:15.000\n"
        "<v Alice>Any blockers?\n",
    )

    return d


@pytest.fixture
def slack_export_dir(tmp_path: Path) -> Path:
    """Create a temp Slack export directory."""
    d = tmp_path / "slack_export"
    d.mkdir()

    channel = d / "general"
    channel.mkdir()

    messages = [
        {
            "type": "message",
            "user": "U123",
            "text": "Hey team, standup in 5 minutes!",
            "ts": "1234567890.123456",
        },
        {
            "type": "message",
            "user": "U456",
            "text": "Sounds good, I'll be there.",
            "ts": "1234567891.123456",
        },
        {
            "type": "message",
            "user": "U123",
            "text": "Quick update: the API deploy went well :tada:",
            "ts": "1234567892.123456",
        },
        {
            "type": "message",
            "user": "U123",
            "text": "Check this out: https://example.com/docs",
            "ts": "1234567893.123456",
        },
        {
            "type": "message",
            "subtype": "channel_join",
            "user": "U789",
            "text": "has joined the channel",
            "ts": "1234567894.123456",
        },
    ]

    (channel / "2024-01-15.json").write_text(
        json.dumps(messages),
    )

    return d


@pytest.fixture
def email_dir(tmp_path: Path) -> Path:
    """Create a temp directory with email files."""
    d = tmp_path / "emails"
    d.mkdir()

    # Plain text email
    (d / "email1.txt").write_text(
        "Hi team,\n\n"
        "I wanted to follow up on the deployment plan. "
        "The changes look good and I think we can proceed.\n\n"
        "Best regards,\n"
        "Alice"
    )

    # Another email
    (d / "email2.txt").write_text(
        "Hey Bob,\n\n"
        "Thanks for the update. I agree with the approach.\n"
        "Let me know if you need any help.\n\n"
        "--\n"
        "Alice"
    )

    return d


@pytest.fixture
def doc_dir(tmp_path: Path) -> Path:
    """Create a temp directory with document files."""
    d = tmp_path / "docs"
    d.mkdir()

    (d / "notes.txt").write_text(
        "Meeting notes from sprint planning:\n"
        "We discussed the roadmap for Q2.\n"
        "Key decisions: use React for the frontend.\n"
    )

    (d / "readme.md").write_text(
        "# Project Overview\n\n"
        "This project handles data processing.\n"
        "See the docs directory for more details.\n"
    )

    return d


# --- StyleAnalyzer ---


class TestStyleAnalyzer:
    """Tests for the StyleAnalyzer class."""

    def test_analyze_empty(self, analyzer: StyleAnalyzer) -> None:
        """Empty input should return default metrics."""
        metrics = analyzer.analyze([])
        assert metrics == StyleMetrics()

    def test_analyze_basic(
        self, analyzer: StyleAnalyzer, sample_texts: list[str],
    ) -> None:
        """Analyze should return populated metrics."""
        metrics = analyzer.analyze(sample_texts)
        assert metrics.total_words > 0
        assert metrics.avg_sentence_length > 0
        assert 0 <= metrics.formality_score <= 1
        assert 0 <= metrics.vocabulary_richness <= 1
        assert metrics.filler_word_rate >= 0

    def test_analyze_formal(
        self, analyzer: StyleAnalyzer, formal_text: str,
    ) -> None:
        """Formal text should score high on formality."""
        metrics = analyzer.analyze([formal_text])
        assert metrics.formality_score > 0.5

    def test_analyze_casual(
        self, analyzer: StyleAnalyzer, casual_text: str,
    ) -> None:
        """Casual text should score low on formality."""
        metrics = analyzer.analyze([casual_text])
        assert metrics.formality_score < 0.5

    def test_analyze_questions(self, analyzer: StyleAnalyzer) -> None:
        """Text with questions should have question frequency > 0."""
        texts = [
            "What do you think? Should we proceed? "
            "I'm not sure about this."
        ]
        metrics = analyzer.analyze(texts)
        assert metrics.question_frequency > 0

    def test_analyze_filler_words(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Text with fillers should have filler_word_rate > 0."""
        texts = [
            "Um, I basically think, you know, that like "
            "we should um sort of do this thing."
        ]
        metrics = analyzer.analyze(texts)
        assert metrics.filler_word_rate > 0

    def test_analyze_technical(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Technical text should have higher technical_depth."""
        tech = [
            "Deploy the API to the infrastructure pipeline. "
            "The microservice architecture needs optimization. "
            "Check the database schema migration."
        ]
        plain = ["I went to the store and bought some milk."]

        tech_metrics = analyzer.analyze(tech)
        plain_metrics = analyzer.analyze(plain)
        assert tech_metrics.technical_depth > plain_metrics.technical_depth

    def test_analyze_exclamations(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Text with exclamations should have exclamation_rate > 0."""
        texts = ["Great job! Awesome work! This is amazing!"]
        metrics = analyzer.analyze(texts)
        assert metrics.exclamation_rate > 0

    def test_extract_vocabulary_empty(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Empty input should return empty vocabulary."""
        vocab = analyzer.extract_vocabulary([])
        assert vocab == VocabularyProfile()

    def test_extract_vocabulary(
        self, analyzer: StyleAnalyzer, sample_texts: list[str],
    ) -> None:
        """Vocabulary extraction should return top words."""
        vocab = analyzer.extract_vocabulary(sample_texts)
        assert len(vocab.top_words) > 0
        assert vocab.unique_ratio > 0
        assert vocab.avg_word_length > 0

    def test_detect_filler_words_empty(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Empty input should return empty dict."""
        fillers = analyzer.detect_filler_words([])
        assert fillers == {}

    def test_detect_filler_words(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Should detect common filler words."""
        texts = [
            "Um, I basically think, you know, that we should "
            "like do this um sort of thing."
        ]
        fillers = analyzer.detect_filler_words(texts)
        assert "um" in fillers
        assert fillers["um"] > 0

    def test_detect_multi_word_fillers(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Should detect multi-word fillers like 'you know'."""
        texts = [
            "Well, you know, I think we should, you know, "
            "just go ahead and do it."
        ]
        fillers = analyzer.detect_filler_words(texts)
        assert "you know" in fillers

    def test_analyze_single_word(
        self, analyzer: StyleAnalyzer,
    ) -> None:
        """Single word input should not crash."""
        metrics = analyzer.analyze(["Hello"])
        assert metrics.total_words == 1


# --- PersonaProfile TOML serialization ---


class TestPersonaProfileToml:
    """Tests for PersonaProfile TOML save/load."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Round-trip save and load should preserve data."""
        profile = PersonaProfile(
            name="Test User",
            formality=0.7,
            verbosity=0.4,
            technical_depth=0.8,
            assertiveness=0.6,
            filler_words={"um": 2.5, "you know": 1.3},
            common_phrases=["let me check", "sounds good"],
            greeting_style="Hey folks",
            agreement_style="Yeah, makes sense",
            avg_response_words=35,
            preferred_terms=["deploy", "pipeline"],
            avoided_terms=["synergy"],
            standup_format="Did X, doing Y, no blockers",
            vocabulary_richness=0.65,
            question_frequency=0.15,
            exclamation_rate=0.05,
        )

        path = tmp_path / "test_profile.toml"
        profile.to_toml(path)
        assert path.exists()

        loaded = PersonaProfile.from_toml(path)
        assert loaded.name == "Test User"
        assert loaded.formality == 0.7
        assert loaded.verbosity == 0.4
        assert loaded.technical_depth == 0.8
        assert loaded.assertiveness == 0.6
        assert loaded.greeting_style == "Hey folks"
        assert loaded.agreement_style == "Yeah, makes sense"
        assert loaded.avg_response_words == 35
        assert "deploy" in loaded.preferred_terms
        assert "synergy" in loaded.avoided_terms
        assert "let me check" in loaded.common_phrases
        assert loaded.vocabulary_richness == 0.65
        assert loaded.question_frequency == 0.15
        assert loaded.exclamation_rate == 0.05

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Saving should create parent directories."""
        path = tmp_path / "deep" / "nested" / "profile.toml"
        profile = PersonaProfile(name="Nested")
        profile.to_toml(path)
        assert path.exists()

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PersonaProfile.from_toml(tmp_path / "nonexistent.toml")

    def test_default_profile_roundtrip(self, tmp_path: Path) -> None:
        """Default profile should round-trip without errors."""
        path = tmp_path / "default.toml"
        profile = PersonaProfile()
        profile.to_toml(path)
        loaded = PersonaProfile.from_toml(path)
        assert loaded.formality == profile.formality

    def test_filler_words_key_roundtrip(
        self, tmp_path: Path,
    ) -> None:
        """Filler word keys with spaces should survive round-trip."""
        path = tmp_path / "fillers.toml"
        profile = PersonaProfile(
            filler_words={"you know": 3.0, "sort of": 1.5},
        )
        profile.to_toml(path)
        loaded = PersonaProfile.from_toml(path)
        assert "you know" in loaded.filler_words
        assert "sort of" in loaded.filler_words
        assert loaded.filler_words["you know"] == 3.0

    def test_special_chars_in_name(self, tmp_path: Path) -> None:
        """Names with special characters should be escaped."""
        path = tmp_path / "special.toml"
        profile = PersonaProfile(name='Test "User" O\'Brien')
        profile.to_toml(path)
        loaded = PersonaProfile.from_toml(path)
        assert loaded.name == 'Test "User" O\'Brien'


# --- PersonaBuilder ---


class TestPersonaBuilder:
    """Tests for the PersonaBuilder class."""

    def test_build_empty(self, builder: PersonaBuilder) -> None:
        """Building with no samples returns default profile."""
        profile = builder.build_from_samples([], name="Empty")
        assert profile.name == "Empty"

    def test_build_from_samples(
        self, builder: PersonaBuilder, sample_texts: list[str],
    ) -> None:
        """Building from samples should produce a valid profile."""
        samples = [
            TextSample(text=t, source_type="transcript")
            for t in sample_texts
        ]
        profile = builder.build_from_samples(
            samples, name="Test",
        )
        assert profile.name == "Test"
        assert profile.formality >= 0
        assert profile.verbosity >= 0
        assert len(profile.preferred_terms) > 0

    def test_build_from_texts(
        self, builder: PersonaBuilder,
    ) -> None:
        """Convenience method should work."""
        texts = [
            "I deployed the API yesterday. "
            "The pipeline is running smoothly.",
            "Let me check the database migration status.",
        ]
        profile = builder.build_from_texts(
            texts, name="Dev", source_type="transcript",
        )
        assert profile.name == "Dev"
        assert profile.technical_depth > 0

    def test_build_weighting(
        self, builder: PersonaBuilder,
    ) -> None:
        """Transcript samples should have higher weight."""
        formal = TextSample(
            text=(
                "Therefore, I propose we subsequently review "
                "the aforementioned changes. Furthermore, the "
                "implementation is accordingly complete. However, "
                "we must nevertheless proceed carefully."
            ),
            source_type="transcript",
            weight=1.0,
        )
        casual = TextSample(
            text=(
                "Hey yeah gonna wanna do this stuff cool ok. "
                "Totally awesome nah dunno yep nope. "
                "Hey yeah gonna wanna do this stuff cool ok. "
                "Totally awesome nah dunno yep nope."
            ),
            source_type="email",
            weight=1.0,
        )

        # With transcript weight (2.0x) the profile should
        # lean more formal than if weights were equal
        profile = builder.build_from_samples(
            [formal, casual], name="Mixed",
        )
        # Transcript weight is 2.0 vs email 1.0, so formality
        # should be pulled toward the formal sample
        assert profile.formality > 0.3

    def test_generate_system_prompt(
        self, builder: PersonaBuilder,
    ) -> None:
        """System prompt should contain persona instructions."""
        profile = PersonaProfile(
            name="Alice",
            formality=0.8,
            verbosity=0.3,
            technical_depth=0.7,
            assertiveness=0.8,
            filler_words={"um": 2.0},
            greeting_style="Hello everyone",
            agreement_style="I agree",
            avg_response_words=30,
            preferred_terms=["pipeline", "deploy"],
            standup_format="Yesterday X, today Y",
        )
        prompt = builder.generate_system_prompt(profile)
        assert "Alice" in prompt
        assert "formal" in prompt.lower()
        assert "brief" in prompt.lower() or "30" in prompt
        assert "technical" in prompt.lower()
        assert "direct" in prompt.lower()
        assert "um" in prompt
        assert "Hello everyone" in prompt

    def test_generate_prompt_casual(
        self, builder: PersonaBuilder,
    ) -> None:
        """Casual profile should produce casual prompt."""
        profile = PersonaProfile(formality=0.1, verbosity=0.9)
        prompt = builder.generate_system_prompt(profile)
        assert "casual" in prompt.lower()

    def test_save_and_load(
        self, builder: PersonaBuilder, tmp_path: Path,
    ) -> None:
        """Builder save/load should work."""
        profile = PersonaProfile(name="SaveTest", formality=0.9)
        path = tmp_path / "saved.toml"
        builder.save(profile, path)

        loaded = builder.load(path)
        assert loaded.name == "SaveTest"
        assert loaded.formality == 0.9


# --- TranscriptAnalyzer ---


class TestTranscriptAnalyzer:
    """Tests for the TranscriptAnalyzer class."""

    def test_analyze_plain_text(
        self, transcript_dir: Path,
    ) -> None:
        """Should parse plain text transcript."""
        ta = TranscriptAnalyzer()
        excerpts = ta.analyze_file(transcript_dir / "meeting1.txt")
        assert len(excerpts) >= 5
        speakers = {e.speaker for e in excerpts}
        assert "Alice" in speakers
        assert "Bob" in speakers
        assert "Carol" in speakers

    def test_analyze_vtt(self, transcript_dir: Path) -> None:
        """Should parse VTT transcript."""
        ta = TranscriptAnalyzer()
        excerpts = ta.analyze_file(transcript_dir / "meeting2.vtt")
        assert len(excerpts) >= 3
        speakers = {e.speaker for e in excerpts}
        assert "Alice" in speakers
        assert "Bob" in speakers

    def test_analyze_vtt_timestamps(
        self, transcript_dir: Path,
    ) -> None:
        """VTT excerpts should have timestamps."""
        ta = TranscriptAnalyzer()
        excerpts = ta.analyze_file(transcript_dir / "meeting2.vtt")
        assert any(e.timestamp for e in excerpts)

    def test_filter_by_speaker(
        self, transcript_dir: Path,
    ) -> None:
        """Should filter by speaker name."""
        ta = TranscriptAnalyzer()
        excerpts = ta.analyze_file(
            transcript_dir / "meeting1.txt",
            speaker_name="Alice",
        )
        assert all(
            "alice" in e.speaker.lower() for e in excerpts
        )
        assert len(excerpts) >= 2

    def test_analyze_directory(
        self, transcript_dir: Path,
    ) -> None:
        """Should analyze all transcripts in a directory."""
        ta = TranscriptAnalyzer()
        utterances = ta.analyze_directory(
            transcript_dir, "Alice",
        )
        assert len(utterances) >= 3  # from both files

    def test_analyze_missing_file(self, tmp_path: Path) -> None:
        """Missing file should return empty list."""
        ta = TranscriptAnalyzer()
        excerpts = ta.analyze_file(tmp_path / "missing.txt")
        assert excerpts == []

    def test_analyze_missing_directory(
        self, tmp_path: Path,
    ) -> None:
        """Missing directory should return empty list."""
        ta = TranscriptAnalyzer()
        result = ta.analyze_directory(tmp_path / "missing", "Alice")
        assert result == []

    def test_srt_format(self, tmp_path: Path) -> None:
        """Should parse SRT format."""
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:05,000\n"
            "Alice: Hello everyone.\n\n"
            "2\n"
            "00:00:06,000 --> 00:00:10,000\n"
            "Bob: Hi there.\n"
        )
        ta = TranscriptAnalyzer()
        excerpts = ta.analyze_file(srt)
        assert len(excerpts) >= 2


# --- SlackAnalyzer ---


class TestSlackAnalyzer:
    """Tests for the SlackAnalyzer class."""

    def test_analyze_export(
        self, slack_export_dir: Path,
    ) -> None:
        """Should extract messages for a specific user."""
        sa = SlackAnalyzer()
        messages = sa.analyze_export(
            slack_export_dir, user_id="U123",
        )
        assert len(messages) == 3
        assert any("standup" in m for m in messages)

    def test_analyze_export_filters_subtypes(
        self, slack_export_dir: Path,
    ) -> None:
        """Should skip messages with subtypes (joins, etc)."""
        sa = SlackAnalyzer()
        messages = sa.analyze_export(
            slack_export_dir, user_id="U789",
        )
        assert len(messages) == 0

    def test_analyze_export_other_user(
        self, slack_export_dir: Path,
    ) -> None:
        """Should return only the specified user's messages."""
        sa = SlackAnalyzer()
        messages = sa.analyze_export(
            slack_export_dir, user_id="U456",
        )
        assert len(messages) == 1
        assert "Sounds good" in messages[0]

    def test_analyze_missing_dir(self, tmp_path: Path) -> None:
        """Missing directory should return empty list."""
        sa = SlackAnalyzer()
        messages = sa.analyze_export(
            tmp_path / "missing", user_id="U123",
        )
        assert messages == []

    def test_extract_patterns(
        self, slack_export_dir: Path,
    ) -> None:
        """Should extract patterns from messages."""
        sa = SlackAnalyzer()
        messages = sa.analyze_export(
            slack_export_dir, user_id="U123",
        )
        patterns = sa.extract_patterns(messages)
        assert "avg_message_words" in patterns
        assert "emoji_usage_rate" in patterns
        assert "url_sharing_rate" in patterns
        assert patterns["emoji_usage_rate"] > 0
        assert patterns["url_sharing_rate"] > 0

    def test_extract_patterns_empty(self) -> None:
        """Empty messages should return empty patterns."""
        sa = SlackAnalyzer()
        patterns = sa.extract_patterns([])
        assert patterns == {}


# --- WritingAnalyzer ---


class TestWritingAnalyzer:
    """Tests for the WritingAnalyzer class."""

    def test_analyze_emails(self, email_dir: Path) -> None:
        """Should extract email body texts."""
        wa = WritingAnalyzer()
        bodies = wa.analyze_emails(email_dir)
        assert len(bodies) == 2
        assert any("deployment plan" in b for b in bodies)

    def test_email_strips_signature(
        self, email_dir: Path,
    ) -> None:
        """Should strip email signatures after '--'."""
        wa = WritingAnalyzer()
        bodies = wa.analyze_emails(email_dir)
        # email2.txt has "--" signature separator
        for body in bodies:
            assert "Alice" not in body or "--" not in body

    def test_analyze_documents(self, doc_dir: Path) -> None:
        """Should extract document texts."""
        wa = WritingAnalyzer()
        texts = wa.analyze_documents(doc_dir)
        assert len(texts) == 2
        assert any("sprint planning" in t for t in texts)

    def test_analyze_missing_dir(self, tmp_path: Path) -> None:
        """Missing directory should return empty list."""
        wa = WritingAnalyzer()
        assert wa.analyze_emails(tmp_path / "missing") == []
        assert wa.analyze_documents(tmp_path / "missing") == []

    def test_eml_parsing(self, tmp_path: Path) -> None:
        """Should parse .eml files."""
        d = tmp_path / "eml_test"
        d.mkdir()

        eml = d / "test.eml"
        eml.write_text(
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Test\n"
            "Content-Type: text/plain; charset=utf-8\n"
            "\n"
            "Hello Bob,\n\n"
            "This is a test email.\n\n"
            "Best,\nAlice\n"
        )

        wa = WritingAnalyzer()
        bodies = wa.analyze_emails(d)
        assert len(bodies) == 1
        assert "test email" in bodies[0].lower()


# --- ConversationAnalyzer ---


class TestConversationAnalyzer:
    """Tests for the ConversationAnalyzer class."""

    def test_analyze_conversation(self) -> None:
        """Should extract responses and patterns."""
        ca = ConversationAnalyzer()
        turns = [
            ConversationTurn("interviewer", "How do you start meetings?"),
            ConversationTurn("subject", "Hey everyone, let's get going."),
            ConversationTurn("interviewer", "How do you give updates?"),
            ConversationTurn(
                "subject",
                "I usually say what I did yesterday and "
                "what I'm doing today.",
            ),
            ConversationTurn("interviewer", "What about blockers?"),
            ConversationTurn(
                "subject",
                "Yeah, I just mention them at the end.",
            ),
        ]

        result = ca.analyze_conversation(turns)
        assert len(result["responses"]) == 3
        assert any("greetings" in k for k in result)
        assert len(result["greetings"]) >= 1
        assert len(result["agreements"]) >= 1
        assert len(result["patterns"]) >= 1

    def test_analyze_empty(self) -> None:
        """Empty turns should return empty result."""
        ca = ConversationAnalyzer()
        result = ca.analyze_conversation([])
        assert result["responses"] == []

    def test_extract_response_patterns(self) -> None:
        """Should extract response starter patterns."""
        ca = ConversationAnalyzer()
        turns = [
            ConversationTurn("interviewer", "Question 1?"),
            ConversationTurn("subject", "Well I think we should go."),
            ConversationTurn("interviewer", "Question 2?"),
            ConversationTurn("subject", "Well I think it depends."),
            ConversationTurn("interviewer", "Question 3?"),
            ConversationTurn("subject", "That makes sense to me."),
        ]
        patterns = ca.extract_response_patterns(turns)
        assert len(patterns) >= 1

    def test_load_conversation(self, tmp_path: Path) -> None:
        """Should load conversation from JSON."""
        ca = ConversationAnalyzer()
        path = tmp_path / "convo.json"
        data = [
            {"role": "interviewer", "text": "Hi there."},
            {"role": "subject", "text": "Hello!"},
        ]
        path.write_text(json.dumps(data))

        turns = ca.load_conversation(path)
        assert len(turns) == 2
        assert turns[0].role == "interviewer"
        assert turns[1].text == "Hello!"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Missing file should return empty list."""
        ca = ConversationAnalyzer()
        turns = ca.load_conversation(tmp_path / "missing.json")
        assert turns == []

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON should return empty list."""
        ca = ConversationAnalyzer()
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        turns = ca.load_conversation(path)
        assert turns == []


# --- Integration tests ---


class TestPersonaIntegration:
    """Integration tests for the full persona pipeline."""

    def test_full_pipeline(
        self,
        builder: PersonaBuilder,
        transcript_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: analyze transcripts -> build -> save -> load."""
        # Step 1: Extract text from transcripts
        ta = TranscriptAnalyzer()
        utterances = ta.analyze_directory(
            transcript_dir, "Alice",
        )
        assert len(utterances) > 0

        # Step 2: Build persona from extracted text
        samples = [
            TextSample(text=u, source_type="transcript")
            for u in utterances
        ]
        profile = builder.build_from_samples(
            samples, name="Alice",
        )
        assert profile.name == "Alice"

        # Step 3: Save to TOML
        path = tmp_path / "personas" / "alice.toml"
        builder.save(profile, path)
        assert path.exists()

        # Step 4: Load and verify
        loaded = builder.load(path)
        assert loaded.name == "Alice"
        assert loaded.formality == profile.formality

        # Step 5: Generate system prompt
        prompt = builder.generate_system_prompt(loaded)
        assert "Alice" in prompt
        assert len(prompt) > 50

    def test_multi_source_pipeline(
        self,
        builder: PersonaBuilder,
        transcript_dir: Path,
        email_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Build persona from multiple source types."""
        ta = TranscriptAnalyzer()
        wa = WritingAnalyzer()

        # Gather from transcripts
        transcript_texts = ta.analyze_directory(
            transcript_dir, "Alice",
        )
        # Gather from emails
        email_texts = wa.analyze_emails(email_dir)

        # Build combined samples with different weights
        samples: list[TextSample] = []
        for t in transcript_texts:
            samples.append(
                TextSample(text=t, source_type="transcript"),
            )
        for t in email_texts:
            samples.append(
                TextSample(text=t, source_type="email"),
            )

        profile = builder.build_from_samples(
            samples, name="Alice Multi",
        )
        assert profile.name == "Alice Multi"
        assert profile.avg_response_words >= 10

        # Save and verify
        path = tmp_path / "alice_multi.toml"
        builder.save(profile, path)
        loaded = builder.load(path)
        assert loaded.name == "Alice Multi"
