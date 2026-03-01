"""Anthropic Claude LLM provider.

Uses Claude Sonnet for response generation and Claude Haiku
for fast decision-making (should I respond?).
"""

from __future__ import annotations

import logging

from zoom_auto.config import LLMConfig
from zoom_auto.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """LLM provider using Anthropic Claude API.

    Uses two models:
    - Sonnet: High-quality response generation
    - Haiku: Fast binary decisions (trigger detection)
    """

    def __init__(self, config: LLMConfig, api_key: str) -> None:
        self.config = config
        self._api_key = api_key
        self._client = None

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response using Claude Sonnet.

        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the generated text.
        """
        raise NotImplementedError("Claude generation not yet implemented")

    async def decide(
        self,
        prompt: str,
        context: str = "",
    ) -> tuple[bool, float]:
        """Make a quick decision using Claude Haiku.

        Args:
            prompt: The decision question.
            context: Meeting context for the decision.

        Returns:
            Tuple of (should_respond, confidence).
        """
        raise NotImplementedError("Claude decision not yet implemented")

    async def is_available(self) -> bool:
        """Check if Claude API is configured and reachable."""
        return bool(self._api_key)
