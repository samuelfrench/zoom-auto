"""Anthropic Claude LLM provider.

Uses Claude Sonnet for response generation and Claude Haiku
for fast decision-making (should I respond?).
"""

from __future__ import annotations

import logging
import re

from zoom_auto.config import LLMConfig
from zoom_auto.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMRole

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
        self._client: object | None = None

    def _get_client(self) -> object:
        """Lazily initialize the Anthropic client.

        Returns:
            An anthropic.AsyncAnthropic client instance.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _convert_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict[str, str]]]:
        """Convert LLMMessage list to Anthropic API format.

        Extracts the system prompt (if any) and converts the
        remaining messages to the Anthropic message format.

        Args:
            messages: List of LLMMessage objects.

        Returns:
            Tuple of (system_prompt, api_messages).
        """
        system_prompt = ""
        api_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                system_prompt = msg.content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        # Anthropic requires at least one user message
        if not api_messages:
            api_messages.append({
                "role": "user",
                "content": "Please respond.",
            })

        return system_prompt, api_messages

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
        client = self._get_client()
        system_prompt, api_messages = self._convert_messages(messages)

        kwargs: dict[str, object] = {
            "model": self.config.response_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)  # type: ignore[union-attr]

        text = ""
        if response.content:
            text = response.content[0].text

        return LLMResponse(
            text=text,
            model=response.model,
            usage_input_tokens=response.usage.input_tokens,
            usage_output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "",
        )

    async def decide(
        self,
        prompt: str,
        context: str = "",
    ) -> tuple[bool, float]:
        """Make a quick decision using Claude Haiku.

        Sends a concise prompt to Haiku and parses a YES/NO
        response with confidence.

        Args:
            prompt: The decision question.
            context: Meeting context for the decision.

        Returns:
            Tuple of (should_respond, confidence).
        """
        client = self._get_client()

        system = (
            "You are a fast decision engine. Respond with exactly one line: "
            "YES or NO followed by a confidence score 0.0-1.0 and a "
            "5-word reason.\n"
            "Format: YES 0.85 directly addressed by name\n"
            "Format: NO 0.90 someone else is talking"
        )

        user_content = prompt
        if context:
            user_content = f"{context}\n\n{prompt}"

        response = await client.messages.create(  # type: ignore[union-attr]
            model=self.config.decision_model,
            max_tokens=30,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )

        text = ""
        if response.content:
            text = response.content[0].text.strip()

        return self._parse_decision(text)

    @staticmethod
    def _parse_decision(text: str) -> tuple[bool, float]:
        """Parse a YES/NO decision response from the LLM.

        Expected format: "YES 0.85 reason here" or "NO 0.90 reason"

        Args:
            text: Raw LLM response text.

        Returns:
            Tuple of (decision_bool, confidence_float).
        """
        text_upper = text.upper().strip()
        decision = text_upper.startswith("YES")

        # Extract confidence score
        confidence = 0.5
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            try:
                parsed = float(match.group(1))
                if 0.0 <= parsed <= 1.0:
                    confidence = parsed
            except ValueError:
                pass

        return decision, confidence

    async def is_available(self) -> bool:
        """Check if Claude API is configured and reachable."""
        if not self._api_key:
            return False
        try:
            client = self._get_client()
            # Minimal API call to check availability
            await client.messages.create(  # type: ignore[union-attr]
                model=self.config.decision_model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            logger.debug("Claude API not available", exc_info=True)
            return False
