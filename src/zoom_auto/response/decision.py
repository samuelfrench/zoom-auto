"""Response trigger detection — should the bot speak?

Analyzes the current conversation context to determine if the bot
should generate and deliver a response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from zoom_auto.config import ResponseConfig
from zoom_auto.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class TriggerReason(str, Enum):
    """Reasons why the bot was triggered to speak."""

    DIRECT_ADDRESS = "direct_address"
    QUESTION_ASKED = "question_asked"
    TOPIC_EXPERTISE = "topic_expertise"
    LONG_SILENCE = "long_silence"
    REQUESTED_INPUT = "requested_input"
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


class TriggerDetector:
    """Detects when the bot should generate a response.

    Uses a combination of rule-based checks and LLM-based analysis
    to determine if the bot should speak.
    """

    def __init__(self, config: ResponseConfig, llm: LLMProvider) -> None:
        self.config = config
        self.llm = llm

    async def should_respond(
        self, transcript: str, bot_name: str = "AI Assistant"
    ) -> ResponseDecision:
        """Analyze the conversation to decide if the bot should respond.

        Args:
            transcript: Recent conversation transcript.
            bot_name: The bot's display name in the meeting.

        Returns:
            ResponseDecision with the decision and reasoning.
        """
        raise NotImplementedError("Trigger detection not yet implemented")

    async def check_direct_address(self, text: str, bot_name: str) -> bool:
        """Check if the bot was directly addressed by name.

        Args:
            text: The most recent utterance.
            bot_name: The bot's display name.

        Returns:
            True if the bot was directly addressed.
        """
        raise NotImplementedError("Direct address check not yet implemented")
