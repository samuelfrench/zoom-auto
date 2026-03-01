"""Full real-time conversation loop.

Coordinates the complete conversation cycle: listen, understand,
decide, generate, and speak.
"""

from __future__ import annotations

import logging

from zoom_auto.config import Settings
from zoom_auto.context.manager import ContextManager
from zoom_auto.pipeline.audio_pipeline import AudioPipeline
from zoom_auto.response.decision import TriggerDetector
from zoom_auto.response.generator import ResponseGenerator
from zoom_auto.response.turn_manager import TurnManager

logger = logging.getLogger(__name__)


class ConversationLoop:
    """The main real-time conversation loop.

    Coordinates all components to create a natural conversation flow:
    1. Capture audio from meeting
    2. Transcribe speech (STT)
    3. Update context and meeting state
    4. Detect if a response is needed
    5. Generate response (LLM + persona)
    6. Synthesize speech (TTS)
    7. Send audio to meeting
    """

    def __init__(
        self,
        settings: Settings,
        audio_pipeline: AudioPipeline,
        context_manager: ContextManager,
        trigger_detector: TriggerDetector,
        response_generator: ResponseGenerator,
        turn_manager: TurnManager,
    ) -> None:
        self.settings = settings
        self.audio_pipeline = audio_pipeline
        self.context_manager = context_manager
        self.trigger_detector = trigger_detector
        self.response_generator = response_generator
        self.turn_manager = turn_manager
        self._running = False

    async def start(self) -> None:
        """Start the conversation loop."""
        raise NotImplementedError("Conversation loop not yet implemented")

    async def stop(self) -> None:
        """Stop the conversation loop gracefully."""
        self._running = False

    async def process_utterance(self, speaker: str, text: str) -> str | None:
        """Process a transcribed utterance and potentially generate a response.

        Args:
            speaker: Who said it.
            text: What they said.

        Returns:
            Response text if the bot should speak, None otherwise.
        """
        raise NotImplementedError("Utterance processing not yet implemented")

    @property
    def is_running(self) -> bool:
        """Whether the conversation loop is running."""
        return self._running
