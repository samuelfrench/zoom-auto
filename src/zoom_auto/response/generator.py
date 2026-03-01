"""Response generation -- what should the bot say?

Combines LLM generation with persona styling to produce
natural, in-character responses to meeting conversation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from zoom_auto.context.manager import ContextManager
from zoom_auto.llm.base import LLMMessage, LLMProvider, LLMRole
from zoom_auto.persona.builder import PersonaBuilder, PersonaProfile
from zoom_auto.persona.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

# Maximum tokens for meeting responses (no monologues)
_MAX_RESPONSE_TOKENS = 200


@dataclass
class GeneratedResponse:
    """A generated response ready for TTS synthesis.

    Attributes:
        text: The response text to speak.
        persona_applied: Whether persona styling was applied.
        token_usage: Number of tokens used for generation.
    """

    text: str
    persona_applied: bool = False
    token_usage: int = 0


def _clean_response(text: str) -> str:
    """Clean LLM output for spoken delivery.

    Strips markdown formatting, bullet points, numbering, and
    other artifacts that don't work in speech.

    Args:
        text: Raw LLM-generated text.

    Returns:
        Clean text suitable for TTS.
    """
    if not text:
        return ""

    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bullet points and numbered lists
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove code blocks / backticks
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'")

    # Collapse multiple whitespace / newlines into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


class ResponseGenerator:
    """Generates context-aware, persona-styled responses.

    Uses the LLM with the persona's system prompt and current
    meeting context to generate natural responses.

    Args:
        llm: LLM provider for response generation (Sonnet-tier).
        context_manager: The meeting context manager.
        persona: Optional persona profile for styling.
        knowledge_store: Optional knowledge store for project context.
    """

    def __init__(
        self,
        llm: LLMProvider,
        context_manager: ContextManager,
        persona: PersonaProfile | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        self.llm = llm
        self.context_manager = context_manager
        self.persona = persona
        self.knowledge_store = knowledge_store
        self._persona_builder = PersonaBuilder()

    async def generate(
        self, trigger_context: str = ""
    ) -> GeneratedResponse:
        """Generate a response based on current meeting context.

        Builds the full prompt from the context manager and persona,
        then calls the LLM with a strict token limit to avoid
        monologue-style responses.

        Args:
            trigger_context: The specific context that triggered
                the response (e.g., the question asked).

        Returns:
            GeneratedResponse with the text to speak.
        """
        # Refresh context window
        await self.context_manager.get_context()

        # Build system prompt from persona
        system_prompt = self._build_system_prompt()
        persona_applied = self.persona is not None

        # Build messages from context manager
        messages = self.context_manager.build_prompt(
            system_prompt=system_prompt,
        )

        # Append the trigger context as an additional user instruction
        if trigger_context:
            messages.append(LLMMessage(
                role=LLMRole.USER,
                content=(
                    f"The following just happened and you should "
                    f"respond to it:\n{trigger_context}"
                ),
            ))

        # Generate with strict token limit
        response = await self.llm.generate(
            messages=messages,
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=self.llm_temperature,
        )

        cleaned = _clean_response(response.text)

        logger.info(
            "Generated response (%d tokens): %s",
            response.usage_output_tokens,
            cleaned[:80],
        )

        return GeneratedResponse(
            text=cleaned,
            persona_applied=persona_applied,
            token_usage=response.usage_output_tokens,
        )

    @property
    def llm_temperature(self) -> float:
        """Temperature for response generation.

        Slightly higher for personas with high verbosity to allow
        more creative responses.
        """
        if self.persona and self.persona.verbosity > 0.7:
            return 0.8
        return 0.7

    def _build_system_prompt(self) -> str:
        """Build the system prompt incorporating persona style.

        Includes persona styling and project knowledge context
        when available.

        Returns:
            System prompt string for the LLM.
        """
        base = (
            "You are participating in a live meeting. "
            "Respond naturally as a meeting participant. "
            "Keep your response concise and conversational -- "
            "this is spoken dialogue, not written text. "
            "Do not use markdown, bullet points, or numbered lists. "
            "Do not use emojis. "
            "Speak in plain, natural sentences."
        )

        if self.persona:
            persona_prompt = (
                self._persona_builder.generate_system_prompt(self.persona)
            )
            base = f"{persona_prompt}\n\n{base}"

        # Add project knowledge if available
        if self.knowledge_store:
            knowledge = self.knowledge_store.get_context_string()
            if knowledge:
                base += f"\n\n{knowledge}"

        return base

    async def set_persona(self, persona: PersonaProfile) -> None:
        """Update the persona profile used for response styling.

        Args:
            persona: The new persona profile.
        """
        self.persona = persona
        logger.info("Persona updated to: %s", persona.name)
