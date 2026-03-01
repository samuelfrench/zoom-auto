"""Response generation — what should the bot say?

Combines LLM generation with persona styling to produce
natural, in-character responses to meeting conversation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from zoom_auto.context.manager import ContextManager
from zoom_auto.llm.base import LLMProvider
from zoom_auto.persona.builder import PersonaProfile

logger = logging.getLogger(__name__)


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


class ResponseGenerator:
    """Generates context-aware, persona-styled responses.

    Uses the LLM with the persona's system prompt and current
    meeting context to generate natural responses.
    """

    def __init__(
        self,
        llm: LLMProvider,
        context_manager: ContextManager,
        persona: PersonaProfile | None = None,
    ) -> None:
        self.llm = llm
        self.context_manager = context_manager
        self.persona = persona

    async def generate(self, trigger_context: str = "") -> GeneratedResponse:
        """Generate a response based on current meeting context.

        Args:
            trigger_context: The specific context that triggered the response.

        Returns:
            GeneratedResponse with the text to speak.
        """
        raise NotImplementedError("Response generation not yet implemented")

    async def set_persona(self, persona: PersonaProfile) -> None:
        """Update the persona profile used for response styling.

        Args:
            persona: The new persona profile.
        """
        self.persona = persona
