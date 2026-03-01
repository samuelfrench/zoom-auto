"""Test conversation analysis for persona building.

Analyzes interactive test conversations to capture real-time
communication style and response patterns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a test conversation.

    Attributes:
        role: "interviewer" or "subject".
        text: What was said.
        duration_seconds: How long the turn lasted.
    """

    role: str
    text: str
    duration_seconds: float = 0.0


class ConversationAnalyzer:
    """Analyzes test conversations for persona building.

    Uses structured test conversations (interviews, Q&A sessions)
    to capture natural response patterns and style.
    """

    def analyze_conversation(
        self, turns: list[ConversationTurn],
    ) -> dict[str, list[str]]:
        """Analyze a test conversation for communication patterns.

        Args:
            turns: List of conversation turns.

        Returns:
            Dict with keys: "responses", "questions", "greetings",
            "agreements", "patterns".
        """
        result: dict[str, list[str]] = {
            "responses": [],
            "questions": [],
            "greetings": [],
            "agreements": [],
            "patterns": [],
        }

        if not turns:
            return result

        for turn in turns:
            if turn.role == "subject":
                text = turn.text.strip()
                if not text:
                    continue

                result["responses"].append(text)

                # Classify the response
                if text.endswith("?"):
                    result["questions"].append(text)

                lower = text.lower()
                if any(
                    g in lower
                    for g in (
                        "hi", "hello", "hey", "good morning",
                        "good afternoon",
                    )
                ):
                    result["greetings"].append(text)

                if any(
                    a in lower
                    for a in (
                        "sounds good", "agree", "makes sense",
                        "right", "exactly", "absolutely", "sure",
                        "definitely", "yes", "yep", "yeah",
                    )
                ):
                    result["agreements"].append(text)

        # Extract patterns
        responses = result["responses"]
        if responses:
            avg_len = sum(len(r.split()) for r in responses) / len(
                responses
            )
            result["patterns"].append(
                f"avg_response_length: {avg_len:.1f} words"
            )

            question_rate = (
                len(result["questions"]) / len(responses)
            )
            result["patterns"].append(
                f"question_rate: {question_rate:.2f}"
            )

        return result

    def extract_response_patterns(
        self, turns: list[ConversationTurn],
    ) -> list[str]:
        """Extract typical response patterns from a conversation.

        Looks at how the subject responds to different types of
        interviewer prompts (questions, statements, etc.).

        Args:
            turns: List of conversation turns.

        Returns:
            List of response pattern descriptions.
        """
        patterns: list[str] = []
        if not turns:
            return patterns

        # Pair interviewer prompts with subject responses
        prev_interviewer: str | None = None
        response_starters: list[str] = []

        for turn in turns:
            if turn.role == "interviewer":
                prev_interviewer = turn.text.strip()
            elif turn.role == "subject" and prev_interviewer:
                text = turn.text.strip()
                if text:
                    # Get first few words as "starter"
                    words = text.split()
                    starter = " ".join(words[:3]).lower()
                    response_starters.append(starter)
                prev_interviewer = None

        # Find common response starters
        if response_starters:
            from collections import Counter

            starter_counts = Counter(response_starters)
            for starter, count in starter_counts.most_common(5):
                if count >= 2:
                    patterns.append(
                        f"Often starts responses with: "
                        f'"{starter}" ({count} times)'
                    )

        # Analyze response length variation
        subject_turns = [
            t for t in turns if t.role == "subject" and t.text.strip()
        ]
        if subject_turns:
            lengths = [len(t.text.split()) for t in subject_turns]
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            patterns.append(
                f"Response length: {avg_len:.0f} words avg "
                f"(range {min_len}-{max_len})"
            )

        return patterns

    def load_conversation(
        self, path: Path,
    ) -> list[ConversationTurn]:
        """Load a conversation from a JSON file.

        Expected format:
        [
            {"role": "interviewer", "text": "..."},
            {"role": "subject", "text": "..."},
            ...
        ]

        Args:
            path: Path to the JSON file.

        Returns:
            List of ConversationTurn objects.
        """
        if not path.exists():
            logger.warning("Conversation file not found: %s", path)
            return []

        try:
            data = json.loads(
                path.read_text(encoding="utf-8"),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load conversation %s: %s", path, exc,
            )
            return []

        if not isinstance(data, list):
            logger.warning("Conversation file is not a list: %s", path)
            return []

        turns: list[ConversationTurn] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            turns.append(ConversationTurn(
                role=item.get("role", "unknown"),
                text=item.get("text", ""),
                duration_seconds=float(
                    item.get("duration_seconds", 0.0),
                ),
            ))

        return turns
