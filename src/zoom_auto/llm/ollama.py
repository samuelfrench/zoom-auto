"""Ollama local LLM provider.

Fallback provider using locally-hosted models via Ollama
for offline or privacy-sensitive operation.
"""

from __future__ import annotations

import logging
import re

import httpx

from zoom_auto.config import LLMConfig
from zoom_auto.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMRole

logger = logging.getLogger(__name__)

# Timeout for Ollama requests (generation can be slow on CPU)
_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)


class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama instance.

    Provides a fallback when Claude API is unavailable or
    when local-only operation is desired.
    """

    def __init__(
        self,
        config: LLMConfig,
        host: str = "http://localhost:11434",
    ) -> None:
        self.config = config
        self.host = host.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the httpx client.

        Returns:
            An httpx.AsyncClient instance.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=_TIMEOUT,
            )
        return self._client

    @staticmethod
    def _convert_messages(
        messages: list[LLMMessage],
    ) -> list[dict[str, str]]:
        """Convert LLMMessage list to Ollama chat API format.

        Args:
            messages: List of LLMMessage objects.

        Returns:
            List of message dicts for Ollama /api/chat.
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

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
        client = self._get_client()
        ollama_messages = self._convert_messages(messages)

        payload = {
            "model": self.config.ollama_model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "")
        model = data.get("model", self.config.ollama_model)

        # Ollama reports token counts in eval_count / prompt_eval_count
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        done_reason = data.get("done_reason", "")

        return LLMResponse(
            text=text,
            model=model,
            usage_input_tokens=input_tokens,
            usage_output_tokens=output_tokens,
            stop_reason=done_reason,
        )

    async def decide(
        self,
        prompt: str,
        context: str = "",
    ) -> tuple[bool, float]:
        """Make a quick decision using the local model.

        Uses a constrained prompt to get a YES/NO answer.

        Args:
            prompt: The decision question.
            context: Meeting context for the decision.

        Returns:
            Tuple of (should_respond, confidence).
        """
        system_msg = LLMMessage(
            role=LLMRole.SYSTEM,
            content=(
                "You are a fast decision engine. Respond with exactly "
                "one line: YES or NO followed by a confidence score "
                "0.0-1.0 and a 5-word reason.\n"
                "Format: YES 0.85 directly addressed by name\n"
                "Format: NO 0.90 someone else is talking"
            ),
        )

        user_content = prompt
        if context:
            user_content = f"{context}\n\n{prompt}"

        user_msg = LLMMessage(role=LLMRole.USER, content=user_content)

        response = await self.generate(
            messages=[system_msg, user_msg],
            max_tokens=30,
            temperature=0.0,
        )

        return self._parse_decision(response.text)

    @staticmethod
    def _parse_decision(text: str) -> tuple[bool, float]:
        """Parse a YES/NO decision response.

        Args:
            text: Raw LLM response text.

        Returns:
            Tuple of (decision_bool, confidence_float).
        """
        text_upper = text.upper().strip()
        decision = text_upper.startswith("YES")

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
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                m.get("name", "") for m in data.get("models", [])
            ]
            # Check if our configured model is available
            target = self.config.ollama_model
            return any(
                target in m or m.startswith(target.split(":")[0])
                for m in models
            )
        except Exception:
            logger.debug("Ollama not available", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
