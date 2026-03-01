"""Context manager with sliding window and summarization.

Maintains a bounded context window of the meeting conversation,
automatically summarizing older content to stay within token limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from zoom_auto.config import ContextConfig

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """The current context window for LLM consumption.

    Attributes:
        summary: Summary of earlier conversation (before the window).
        recent_transcript: Recent transcript lines within the window.
        meeting_context: Key meeting state (agenda, decisions, etc.).
        total_tokens_estimate: Estimated total token count.
    """

    summary: str = ""
    recent_transcript: list[str] = field(default_factory=list)
    meeting_context: str = ""
    total_tokens_estimate: int = 0


class ContextManager:
    """Manages the sliding context window for LLM prompts.

    Keeps the most recent conversation in full detail while
    summarizing older content to stay within token limits.
    """

    def __init__(self, config: ContextConfig) -> None:
        self.config = config
        self._window = ContextWindow()

    async def add_utterance(self, speaker: str, text: str) -> None:
        """Add a new utterance to the context.

        Args:
            speaker: Name of the speaker.
            text: What they said.
        """
        raise NotImplementedError("Utterance addition not yet implemented")

    async def get_context(self) -> ContextWindow:
        """Get the current context window for LLM consumption.

        Returns:
            ContextWindow with summary, recent transcript, and meeting state.
        """
        return self._window

    async def summarize_if_needed(self) -> bool:
        """Summarize older content if the window exceeds the threshold.

        Returns:
            True if summarization was performed.
        """
        raise NotImplementedError("Summarization not yet implemented")

    async def reset(self) -> None:
        """Reset the context for a new meeting."""
        self._window = ContextWindow()
