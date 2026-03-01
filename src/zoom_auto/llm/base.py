"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class LLMRole(StrEnum):
    """Message roles for LLM conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A single message in an LLM conversation.

    Attributes:
        role: The role of the message sender.
        content: The message text content.
    """

    role: LLMRole
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        text: The generated text response.
        model: The model that generated the response.
        usage_input_tokens: Number of input tokens used.
        usage_output_tokens: Number of output tokens generated.
        stop_reason: Why generation stopped.
    """

    text: str
    model: str = ""
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    stop_reason: str = ""


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must support both full response generation
    and quick decision-making (binary yes/no with confidence).
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the generated text.
        """
        ...

    @abstractmethod
    async def decide(
        self,
        prompt: str,
        context: str = "",
    ) -> tuple[bool, float]:
        """Make a quick binary decision with confidence.

        Used for fast decisions like "should I respond to this?"

        Args:
            prompt: The decision question.
            context: Additional context for the decision.

        Returns:
            Tuple of (decision: bool, confidence: float 0-1).
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        ...
