"""Response trigger detection -- should the bot speak?

Analyzes the current conversation context to determine if the bot
should generate and deliver a response.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from zoom_auto.config import ResponseConfig
from zoom_auto.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class TriggerReason(StrEnum):
    """Reasons why the bot was triggered to speak."""

    DIRECT_ADDRESS = "direct_address"
    QUESTION_ASKED = "question_asked"
    TOPIC_EXPERTISE = "topic_expertise"
    LONG_SILENCE = "long_silence"
    REQUESTED_INPUT = "requested_input"
    STANDUP_TURN = "standup_turn"
    AGREEMENT = "agreement"
    NONE = "none"


@dataclass
class ResponseDecision:
    """The result of trigger detection.

    Attributes:
        should_respond: Whether the bot should speak.
        confidence: Confidence in the decision (0-1).
        reason: Why the bot was triggered.
        context_snippet: Relevant context that triggered the decision.
    """

    should_respond: bool
    confidence: float
    reason: TriggerReason = TriggerReason.NONE
    context_snippet: str = ""


# Patterns that indicate someone is requesting input from others
_INPUT_REQUEST_PATTERNS = [
    r"what do you think",
    r"any thoughts",
    r"anyone else",
    r"does anyone",
    r"what does everyone think",
    r"your (thoughts|opinion|take)",
    r"want to (add|chime in|weigh in)",
    r"go ahead",
]

# Patterns for standup turn detection
_STANDUP_PATTERNS = [
    r"your turn",
    r"you're up",
    r"go ahead",
    r"what('s| is) your update",
    r"how about you",
    r"what have you been working on",
    r"what did you work on",
]


class TriggerDetector:
    """Detects when the bot should generate a response.

    Uses a combination of rule-based checks and LLM-based analysis
    to determine if the bot should speak.

    Args:
        config: Response engine configuration.
        llm: LLM provider for ambiguous decision-making.
    """

    def __init__(self, config: ResponseConfig, llm: LLMProvider) -> None:
        self.config = config
        self.llm = llm

    async def should_respond(
        self,
        transcript: str,
        bot_name: str = "AI Assistant",
        is_cooldown_active: bool = False,
        someone_speaking: bool = False,
    ) -> ResponseDecision:
        """Analyze the conversation to decide if the bot should respond.

        Decision priority:
        1. Never speak if someone is currently talking.
        2. Always speak if directly addressed by name.
        3. Always speak if it is the bot's standup turn.
        4. Never speak if cooldown is active (unless directly addressed).
        5. For ambiguous cases, use the LLM for a quick decision.

        Args:
            transcript: Recent conversation transcript.
            bot_name: The bot's display name in the meeting.
            is_cooldown_active: Whether the response cooldown is active.
            someone_speaking: Whether someone is currently speaking.

        Returns:
            ResponseDecision with the decision and reasoning.
        """
        if not transcript.strip():
            return ResponseDecision(
                should_respond=False,
                confidence=1.0,
                reason=TriggerReason.NONE,
                context_snippet="",
            )

        # Extract the last few lines for analysis
        lines = [
            ln.strip() for ln in transcript.strip().splitlines() if ln.strip()
        ]
        recent_text = "\n".join(lines[-5:]) if lines else ""
        last_line = lines[-1] if lines else ""

        # --- Rule-based: never speak if someone is talking ---
        if someone_speaking:
            return ResponseDecision(
                should_respond=False,
                confidence=1.0,
                reason=TriggerReason.NONE,
                context_snippet="someone is currently speaking",
            )

        # --- Rule-based: direct address (highest priority) ---
        if await self.check_direct_address(last_line, bot_name):
            return ResponseDecision(
                should_respond=True,
                confidence=1.0,
                reason=TriggerReason.DIRECT_ADDRESS,
                context_snippet=last_line,
            )

        # --- Rule-based: standup turn ---
        if self._check_standup_turn(last_line, bot_name):
            return ResponseDecision(
                should_respond=True,
                confidence=0.95,
                reason=TriggerReason.STANDUP_TURN,
                context_snippet=last_line,
            )

        # --- Rule-based: explicit request for input mentioning name ---
        if self._check_input_request(last_line, bot_name):
            return ResponseDecision(
                should_respond=True,
                confidence=0.95,
                reason=TriggerReason.REQUESTED_INPUT,
                context_snippet=last_line,
            )

        # --- Cooldown blocks non-priority triggers ---
        if is_cooldown_active:
            return ResponseDecision(
                should_respond=False,
                confidence=0.9,
                reason=TriggerReason.NONE,
                context_snippet="cooldown active",
            )

        # --- Ambiguous: use LLM for quick decision ---
        return await self._llm_decide(recent_text, bot_name)

    async def check_direct_address(
        self, text: str, bot_name: str
    ) -> bool:
        """Check if the bot was directly addressed by name.

        Does a case-insensitive search for the bot's name in the text,
        ensuring word boundaries to avoid false positives.

        Args:
            text: The most recent utterance.
            bot_name: The bot's display name.

        Returns:
            True if the bot was directly addressed.
        """
        if not text or not bot_name:
            return False

        text_lower = text.lower()
        name_lower = bot_name.lower()

        # Check full name
        pattern = rf"\b{re.escape(name_lower)}\b"
        if re.search(pattern, text_lower):
            return True

        # Check first name only (e.g., "Sam" from "Sam's AI Assistant")
        first_name = name_lower.split()[0] if " " in name_lower else ""
        if first_name and len(first_name) > 2:
            first_pattern = rf"\b{re.escape(first_name)}\b"
            if re.search(first_pattern, text_lower):
                return True

        return False

    def _check_standup_turn(self, text: str, bot_name: str) -> bool:
        """Check if it is the bot's standup turn.

        Args:
            text: Recent utterance text.
            bot_name: The bot's display name.

        Returns:
            True if the bot is being asked for their standup update.
        """
        text_lower = text.lower()
        name_lower = bot_name.lower()
        first_name = name_lower.split()[0] if " " in name_lower else ""

        name_mentioned = name_lower in text_lower
        if first_name and len(first_name) > 2:
            name_mentioned = name_mentioned or first_name in text_lower

        if not name_mentioned:
            return False

        return any(
            re.search(p, text_lower) for p in _STANDUP_PATTERNS
        )

    def _check_input_request(self, text: str, bot_name: str) -> bool:
        """Check if someone is requesting the bot's input.

        Args:
            text: Recent utterance text.
            bot_name: The bot's display name.

        Returns:
            True if input is being requested from the bot.
        """
        text_lower = text.lower()
        name_lower = bot_name.lower()
        first_name = name_lower.split()[0] if " " in name_lower else ""

        name_mentioned = name_lower in text_lower
        if first_name and len(first_name) > 2:
            name_mentioned = name_mentioned or first_name in text_lower

        if not name_mentioned:
            return False

        return any(
            re.search(p, text_lower) for p in _INPUT_REQUEST_PATTERNS
        )

    async def _llm_decide(
        self, recent_text: str, bot_name: str
    ) -> ResponseDecision:
        """Use the LLM for a quick YES/NO decision on ambiguous cases.

        Args:
            recent_text: Recent transcript text.
            bot_name: The bot's display name.

        Returns:
            ResponseDecision from the LLM's judgment.
        """
        prompt = (
            f"You are {bot_name} in a meeting. Based on this recent "
            f"conversation, should {bot_name} speak now?\n"
            f"Consider: Was a question asked to the group? "
            f"Is there an opportunity to add value? "
            f"Is someone waiting for a response?\n\n"
            f"Should {bot_name} speak now? YES/NO + 5-word reason"
        )

        try:
            decision, confidence = await self.llm.decide(
                prompt=prompt,
                context=recent_text,
            )

            if confidence < self.config.trigger_threshold:
                return ResponseDecision(
                    should_respond=False,
                    confidence=confidence,
                    reason=TriggerReason.NONE,
                    context_snippet=recent_text[-200:],
                )

            reason = TriggerReason.QUESTION_ASKED if decision else TriggerReason.NONE
            return ResponseDecision(
                should_respond=decision,
                confidence=confidence,
                reason=reason,
                context_snippet=recent_text[-200:],
            )
        except Exception:
            logger.warning(
                "LLM decision failed, defaulting to no response",
                exc_info=True,
            )
            return ResponseDecision(
                should_respond=False,
                confidence=0.0,
                reason=TriggerReason.NONE,
                context_snippet="llm decision error",
            )
