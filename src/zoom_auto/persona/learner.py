"""Real-time conversation learning.

Extracts patterns, vocabulary, topic expertise, and communication
style from live conversations. Every conversation the bot participates
in makes it smarter and more accurate.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words excluded from vocabulary tracking
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "the",
        "a",
        "an",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "can",
        "may",
        "might",
        "must",
        "and",
        "but",
        "or",
        "if",
        "so",
        "yet",
        "not",
        "no",
        "nor",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "to",
        "up",
        "as",
        "with",
        "from",
        "into",
        "that",
        "this",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "how",
        "than",
        "then",
        "just",
        "also",
        "very",
        "too",
        "about",
        "there",
        "here",
        "all",
        "each",
        "some",
        "any",
        "its",
        "like",
        "well",
        "out",
        "over",
        "get",
        "got",
        "go",
        "going",
        "thing",
        "things",
        "really",
        "know",
        "think",
        "yeah",
        "yes",
        "okay",
        "ok",
        "right",
        "so",
        "um",
        "uh",
        "hmm",
        "oh",
    }
)

# ---------------------------------------------------------------------------
# Topic marker patterns
# ---------------------------------------------------------------------------
_TOPIC_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"let'?s\s+(?:talk|discuss|go over)\s+(?:about\s+)?(.+)", re.I),
    re.compile(r"regarding\s+(.+)", re.I),
    re.compile(r"about\s+the\s+(.+?)[\.\,\?\!]", re.I),
    re.compile(r"(?:update|status)\s+on\s+(.+?)[\.\,\?\!]", re.I),
    re.compile(r"working\s+on\s+(.+?)[\.\,\?\!]", re.I),
]

# ---------------------------------------------------------------------------
# Meeting type keyword sets
# ---------------------------------------------------------------------------
_MEETING_TYPE_KEYWORDS: dict[str, set[str]] = {
    "standup": {
        "yesterday",
        "today",
        "blocker",
        "blockers",
        "standup",
        "stand-up",
        "stand up",
        "blocked",
        "working on",
        "done",
        "accomplished",
    },
    "planning": {
        "sprint",
        "planning",
        "estimate",
        "story points",
        "backlog",
        "prioritize",
        "priority",
        "roadmap",
        "milestone",
        "scope",
        "epic",
    },
    "retro": {
        "retro",
        "retrospective",
        "went well",
        "improve",
        "improvement",
        "action item",
        "action items",
        "what worked",
        "what didn't",
        "keep doing",
        "stop doing",
    },
    "technical": {
        "architecture",
        "design",
        "implementation",
        "refactor",
        "deploy",
        "deployment",
        "database",
        "api",
        "endpoint",
        "schema",
        "migration",
        "performance",
        "latency",
        "bug",
        "debug",
    },
    "brainstorm": {
        "brainstorm",
        "idea",
        "ideas",
        "what if",
        "how about",
        "explore",
        "option",
        "options",
        "proposal",
        "propose",
        "creative",
    },
    "one_on_one": {
        "one on one",
        "1-on-1",
        "1:1",
        "how are you",
        "career",
        "growth",
        "feedback",
        "goals",
        "check in",
        "check-in",
    },
}


@dataclass
class ConversationSession:
    """A recorded conversation session with extracted learnings.

    Attributes:
        session_id: Unique identifier for this session.
        started_at: When the conversation started.
        ended_at: When the conversation ended.
        transcript: Full transcript entries.
        topics_discussed: Topics detected in the conversation.
        vocabulary_learned: New words/phrases observed from the user.
        response_patterns: How the user typically responds in various contexts.
        decisions_made: Decisions captured during the meeting.
        action_items: Action items assigned during the meeting.
        bot_responses: What the bot said and how it was received.
        meeting_type: Detected type (standup, planning, retro, technical, etc.).
    """

    session_id: str
    started_at: str
    ended_at: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)
    topics_discussed: list[str] = field(default_factory=list)
    vocabulary_learned: list[str] = field(default_factory=list)
    response_patterns: list[dict[str, str]] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    bot_responses: list[dict[str, str]] = field(default_factory=list)
    meeting_type: str = "unknown"

    def to_dict(self) -> dict:
        """Serialize session to a plain dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "transcript": self.transcript,
            "topics_discussed": self.topics_discussed,
            "vocabulary_learned": self.vocabulary_learned,
            "response_patterns": self.response_patterns,
            "decisions_made": self.decisions_made,
            "action_items": self.action_items,
            "bot_responses": self.bot_responses,
            "meeting_type": self.meeting_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationSession:
        """Deserialize from a plain dict."""
        return cls(
            session_id=data.get("session_id", ""),
            started_at=data.get("started_at", ""),
            ended_at=data.get("ended_at", ""),
            transcript=data.get("transcript", []),
            topics_discussed=data.get("topics_discussed", []),
            vocabulary_learned=data.get("vocabulary_learned", []),
            response_patterns=data.get("response_patterns", []),
            decisions_made=data.get("decisions_made", []),
            action_items=data.get("action_items", []),
            bot_responses=data.get("bot_responses", []),
            meeting_type=data.get("meeting_type", "unknown"),
        )


class ConversationLearner:
    """Learns from conversations to improve bot behavior over time.

    Persists transcripts and extracted patterns to data/learnings/.
    On startup, loads past learnings to inform the persona and
    response generation.

    Learning types:
    1. Transcript storage -- raw conversation logs for persona training
    2. Vocabulary tracking -- words/phrases the user uses frequently
    3. Response pattern analysis -- how the user reacts to different situations
    4. Topic expertise -- what the user knows about and discusses
    5. Meeting type detection -- standup vs planning vs technical discussion
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        user: str = "default",
    ) -> None:
        self.data_dir = data_dir or Path("data/learnings")
        self.user = user
        self._sessions_dir = self.data_dir / user / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: ConversationSession | None = None
        self._vocabulary: dict[str, int] = {}  # word -> frequency
        self._load_vocabulary()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> str:
        """Start a new learning session. Returns session_id."""
        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._current_session = ConversationSession(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
        )
        logger.info("Learning session started: %s", session_id)
        return session_id

    def record_utterance(
        self,
        speaker: str,
        text: str,
        timestamp: str = "",
    ) -> None:
        """Record an utterance from the conversation.

        Args:
            speaker: Who said it.
            text: What they said.
            timestamp: Optional ISO timestamp.
        """
        if self._current_session is None:
            return

        ts = timestamp or datetime.now().isoformat()
        self._current_session.transcript.append(
            {"speaker": speaker, "text": text, "timestamp": ts}
        )

        # Track vocabulary for the cloned user
        self._track_vocabulary(text)

    def record_bot_response(
        self,
        trigger_reason: str,
        response_text: str,
        context_snippet: str = "",
    ) -> None:
        """Record what the bot said and why.

        Args:
            trigger_reason: Why the bot decided to speak.
            response_text: The response text delivered.
            context_snippet: Relevant context that triggered the response.
        """
        if self._current_session is None:
            return

        self._current_session.bot_responses.append(
            {
                "trigger_reason": trigger_reason,
                "response_text": response_text,
                "context_snippet": context_snippet,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def end_session(self) -> ConversationSession:
        """End the current session, extract learnings, and persist.

        Returns:
            The completed ConversationSession with all learnings.
        """
        if self._current_session is None:
            # Return an empty session if none was started
            return ConversationSession(
                session_id="empty",
                started_at="",
                ended_at="",
                meeting_type="unknown",
            )

        session = self._current_session
        session.ended_at = datetime.now().isoformat()

        # Extract topics from conversation
        session.topics_discussed = self._extract_topics(session.transcript)

        # Detect meeting type
        session.meeting_type = self._detect_meeting_type(session.transcript)

        # Identify new vocabulary learned this session
        session.vocabulary_learned = self._get_session_vocabulary(
            session.transcript
        )

        # Persist session
        self._save_session(session)

        # Update and persist vocabulary
        self._save_vocabulary()

        # Clear current session
        self._current_session = None

        logger.info(
            "Learning session ended: %s (%d utterances, %d topics, type=%s)",
            session.session_id,
            len(session.transcript),
            len(session.topics_discussed),
            session.meeting_type,
        )

        return session

    # ------------------------------------------------------------------
    # Learning context for LLM system prompt
    # ------------------------------------------------------------------

    def get_learning_context(self, max_tokens: int = 500) -> str:
        """Build a context string from accumulated learnings.

        Returns a formatted string for the LLM system prompt with:
        - Past topics the user discusses
        - Communication patterns observed
        - Vocabulary preferences
        - Meeting behavior patterns

        Args:
            max_tokens: Approximate token budget (chars / 4).

        Returns:
            Formatted context string, or empty string if no learnings.
        """
        parts: list[str] = []

        # Top vocabulary words
        top_vocab = sorted(
            self._vocabulary.items(), key=lambda x: x[1], reverse=True
        )[:20]
        if top_vocab:
            words = ", ".join(w for w, _ in top_vocab)
            parts.append(f"Frequently used words: {words}.")

        # Recent topics across sessions
        topics = self._load_accumulated_topics()
        if topics:
            topic_list = ", ".join(topics[:10])
            parts.append(f"Topics frequently discussed: {topic_list}.")

        # Meeting type distribution
        type_counts = self._load_meeting_type_counts()
        if type_counts:
            most_common = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
            parts.append(f"Most common meeting type: {most_common}.")

        if not parts:
            return ""

        header = "## Learned patterns from past conversations\n"
        body = " ".join(parts)

        # Rough token budget enforcement
        max_chars = max_tokens * 4
        if len(header) + len(body) > max_chars:
            body = body[: max_chars - len(header) - 3] + "..."

        return header + body

    def get_transcript_files(self) -> list[Path]:
        """Return paths to all saved transcripts for persona rebuilding."""
        if not self._sessions_dir.exists():
            return []
        return sorted(self._sessions_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def _load_vocabulary(self) -> None:
        """Load accumulated vocabulary from disk."""
        vocab_path = self.data_dir / self.user / "vocabulary.json"
        if vocab_path.exists():
            try:
                self._vocabulary = json.loads(vocab_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load vocabulary from %s", vocab_path)
                self._vocabulary = {}

    def _save_vocabulary(self) -> None:
        """Persist vocabulary to disk."""
        vocab_path = self.data_dir / self.user / "vocabulary.json"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(
            json.dumps(self._vocabulary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _track_vocabulary(self, text: str) -> None:
        """Track word frequency for the user's vocabulary.

        Args:
            text: Raw text to extract words from.
        """
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
        for word in words:
            if word in _STOP_WORDS or len(word) < 3:
                continue
            self._vocabulary[word] = self._vocabulary.get(word, 0) + 1

    def _get_session_vocabulary(
        self, transcript: list[dict[str, str]]
    ) -> list[str]:
        """Get new vocabulary words from the current session.

        Returns words that appeared in this session's transcript.
        """
        session_words: Counter[str] = Counter()
        for entry in transcript:
            words = re.findall(
                r"[a-zA-Z][a-zA-Z0-9_-]*", entry.get("text", "").lower()
            )
            for word in words:
                if word not in _STOP_WORDS and len(word) >= 3:
                    session_words[word] += 1

        # Return words that appeared at least twice in the session
        return [w for w, c in session_words.most_common(50) if c >= 2]

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    def _extract_topics(self, transcript: list[dict[str, str]]) -> list[str]:
        """Extract discussion topics from transcript.

        Simple keyword/phrase extraction -- no LLM needed.
        Looks for:
        - Explicit topic markers ("let's talk about...", "regarding...")
        - Proper nouns and technical terms (capitalized words)
        - Repeated multi-word concepts

        Args:
            transcript: List of transcript entries.

        Returns:
            List of detected topic strings.
        """
        topics: list[str] = []
        all_text = " ".join(entry.get("text", "") for entry in transcript)

        # 1. Explicit topic markers
        for pattern in _TOPIC_MARKERS:
            for match in pattern.finditer(all_text):
                topic = match.group(1).strip().rstrip(".,!?")
                if topic and len(topic) > 2:
                    topics.append(topic.lower())

        # 2. Capitalized multi-word phrases (potential proper nouns / project names)
        cap_phrases = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", all_text
        )
        for phrase in cap_phrases:
            normalized = phrase.lower()
            if normalized not in topics:
                topics.append(normalized)

        # 3. Repeated technical terms (appear 3+ times)
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", all_text.lower())
        word_counts = Counter(words)
        for word, count in word_counts.most_common(20):
            if count >= 3 and word not in _STOP_WORDS and word not in topics:
                topics.append(word)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_topics: list[str] = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                unique_topics.append(t)

        return unique_topics[:20]  # Cap at 20 topics per session

    # ------------------------------------------------------------------
    # Meeting type detection
    # ------------------------------------------------------------------

    def _detect_meeting_type(
        self, transcript: list[dict[str, str]]
    ) -> str:
        """Detect meeting type from conversation patterns.

        Categories: standup, planning, retro, technical, brainstorm,
        one_on_one, general

        Args:
            transcript: List of transcript entries.

        Returns:
            Detected meeting type string.
        """
        all_text = " ".join(
            entry.get("text", "") for entry in transcript
        ).lower()

        scores: dict[str, int] = {}
        for meeting_type, keywords in _MEETING_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in all_text)
            if score > 0:
                scores[meeting_type] = score

        if not scores:
            return "general"

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _save_session(self, session: ConversationSession) -> None:
        """Save a completed session to disk as JSON.

        Args:
            session: The session to persist.
        """
        filename = f"{session.session_id}.json"
        filepath = self._sessions_dir / filename
        filepath.write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info("Session saved to %s", filepath)

    def _load_session(self, path: Path) -> ConversationSession:
        """Load a session from disk.

        Args:
            path: Path to the session JSON file.

        Returns:
            The deserialized ConversationSession.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConversationSession.from_dict(data)

    # ------------------------------------------------------------------
    # Accumulated learning helpers
    # ------------------------------------------------------------------

    def _load_accumulated_topics(self) -> list[str]:
        """Load topics from all past sessions and return most common."""
        topic_counter: Counter[str] = Counter()
        for session_file in self.get_transcript_files():
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                for topic in data.get("topics_discussed", []):
                    topic_counter[topic] += 1
            except (json.JSONDecodeError, OSError):
                continue

        return [t for t, _ in topic_counter.most_common(20)]

    def _load_meeting_type_counts(self) -> dict[str, int]:
        """Load meeting type distribution from all past sessions."""
        type_counts: dict[str, int] = {}
        for session_file in self.get_transcript_files():
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                mt = data.get("meeting_type", "general")
                type_counts[mt] = type_counts.get(mt, 0) + 1
            except (json.JSONDecodeError, OSError):
                continue

        return type_counts
