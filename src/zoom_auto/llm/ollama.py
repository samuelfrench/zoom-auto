"""Ollama local LLM provider.

Fallback provider using locally-hosted models via Ollama
for offline or privacy-sensitive operation.
"""

from __future__ import annotations

import logging

from zoom_auto.config import LLMConfig
from zoom_auto.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama instance.

    Provides a fallback when Claude API is unavailable or
    when local-only operation is desired.
    """

    def __init__(self, config: LLMConfig, host: str = "http://localhost:11434") -> None:
        self.config = config
        self.host = host

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response using a local Ollama model.

        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the generated text.
        """
        raise NotImplementedError("Ollama generation not yet implemented")

    async def decide(
        self,
        prompt: str,
        context: str = "",
    ) -> tuple[bool, float]:
        """Make a quick decision using the local model.

        Args:
            prompt: The decision question.
            context: Meeting context for the decision.

        Returns:
            Tuple of (should_respond, confidence).
        """
        raise NotImplementedError("Ollama decision not yet implemented")

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        raise NotImplementedError("Ollama availability check not yet implemented")
