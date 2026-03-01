"""Full real-time conversation loop.

Coordinates the complete conversation cycle: listen, understand,
decide, generate, and speak.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from zoom_auto.config import Settings
from zoom_auto.context.manager import ContextManager
from zoom_auto.persona.learner import ConversationLearner
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
        learner: ConversationLearner | None = None,
    ) -> None:
        self.settings = settings
        self.audio_pipeline = audio_pipeline
        self.context_manager = context_manager
        self.trigger_detector = trigger_detector
        self.response_generator = response_generator
        self.turn_manager = turn_manager
        self.learner = learner
        self._running = False
        self._response_task: asyncio.Task[None] | None = None
        self._utterance_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._main_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the conversation loop."""
        if self._running:
            logger.warning("ConversationLoop already running")
            return

        self._running = True

        if self.learner:
            self.learner.start_session()

        # Register our transcript callback on the audio pipeline
        self.audio_pipeline.set_transcript_callback(self._on_transcript)

        # Start the audio pipeline
        await self.audio_pipeline.start()

        # Start the main processing loop
        self._main_task = asyncio.create_task(self._main_loop())

        logger.info("ConversationLoop started")

    async def stop(self) -> None:
        """Stop the conversation loop gracefully."""
        if not self._running:
            return

        self._running = False

        if self.learner:
            session = self.learner.end_session()
            logger.info(
                "Session learnings: %d utterances, %d topics, type=%s",
                len(session.transcript),
                len(session.topics_discussed),
                session.meeting_type,
            )

        # Stop the audio pipeline
        await self.audio_pipeline.stop()

        # Cancel any in-progress response generation
        if self._response_task is not None:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
            self._response_task = None

        # Unblock the main loop
        await self._utterance_queue.put(("", ""))

        # Cancel the main loop task
        if self._main_task is not None:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None

        logger.info("ConversationLoop stopped")

    async def process_utterance(self, speaker: str, text: str) -> str | None:
        """Process a transcribed utterance and potentially generate a response.

        Args:
            speaker: Who said it.
            text: What they said.

        Returns:
            Response text if the bot should speak, None otherwise.
        """
        if not text.strip():
            return None

        bot_name = self.settings.zoom.bot_name

        # Record utterance for learning
        if self.learner:
            self.learner.record_utterance(speaker, text)

        # 1. Add transcript to context
        await self.context_manager.add_transcript(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
        )

        # 2. Update turn manager -- someone else spoke
        self.turn_manager.record_other_speaker()

        # 3. Check for interruption -- if bot is speaking and someone
        #    else starts talking, stop the bot
        if self.audio_pipeline.is_bot_speaking:
            logger.info("Interruption detected -- stopping bot speech")
            await self.audio_pipeline.stop_speaking()
            self.turn_manager.mark_bot_done()

        # 4. Check if we should respond
        #    someone_speaking reflects real-time VAD state from SDK events
        # Build recent transcript for trigger detection
        context = await self.context_manager.get_context()
        recent_text = "\n".join(context.recent_transcript)

        decision = await self.trigger_detector.should_respond(
            transcript=recent_text,
            bot_name=bot_name,
            is_cooldown_active=self.turn_manager.is_cooldown_active,
            someone_speaking=self.turn_manager.someone_speaking,
        )

        if not decision.should_respond:
            logger.debug(
                "Not responding (reason=%s, confidence=%.2f)",
                decision.reason,
                decision.confidence,
            )
            return None

        # 5. Check turn manager approval
        if not self.turn_manager.can_speak():
            logger.debug("Turn manager blocked response")
            return None

        # 6. Generate response
        logger.info(
            "Generating response (trigger=%s, confidence=%.2f)",
            decision.reason,
            decision.confidence,
        )

        response = await self.response_generator.generate(
            trigger_context=decision.context_snippet,
        )

        if not response.text.strip():
            logger.debug("Generated empty response, skipping")
            return None

        # Record bot response for learning
        if self.learner:
            self.learner.record_bot_response(
                trigger_reason=decision.reason,
                response_text=response.text,
                context_snippet=decision.context_snippet,
            )

        return response.text

    @property
    def is_running(self) -> bool:
        """Whether the conversation loop is running."""
        return self._running

    async def _on_transcript(self, speaker: str, text: str) -> None:
        """Callback from AudioPipeline when speech is transcribed.

        Queues the utterance for processing in the main loop.

        Args:
            speaker: Speaker display name.
            text: Transcribed text.
        """
        await self._utterance_queue.put((speaker, text))

    async def _main_loop(self) -> None:
        """Main conversation processing loop.

        Consumes transcribed utterances from the queue and processes
        them through trigger detection and response generation.
        """
        logger.debug("Main conversation loop started")

        try:
            while self._running:
                try:
                    speaker, text = await asyncio.wait_for(
                        self._utterance_queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                if not self._running:
                    break

                if not text.strip():
                    continue

                # Process the utterance and potentially generate a response
                response_text = await self.process_utterance(speaker, text)

                if response_text:
                    # Add a natural pause before speaking
                    pause = self.turn_manager.get_natural_pause()
                    await asyncio.sleep(pause)

                    # Check one more time that we should still speak
                    if not self.turn_manager.can_speak():
                        logger.debug(
                            "Turn manager blocked after pause, discarding"
                        )
                        continue

                    # Speak the response
                    await self._speak_response(response_text)

        except asyncio.CancelledError:
            logger.debug("Main loop cancelled")
        except Exception:
            logger.exception("Error in main conversation loop")
        finally:
            logger.debug("Main conversation loop exited")

    async def _speak_response(self, text: str) -> None:
        """Send a response through TTS and update turn state.

        Args:
            text: The response text to speak.
        """
        bot_name = self.settings.zoom.bot_name

        self.turn_manager.mark_bot_speaking()

        try:
            # Send through the audio pipeline (TTS -> AudioSender)
            await self.audio_pipeline.send_response(text)

            # Add bot's response to the conversation context
            await self.context_manager.add_transcript(
                speaker=bot_name,
                text=text,
                timestamp=datetime.now(),
            )

            # Record the response in turn manager
            self.turn_manager.record_response()

            logger.info("Bot spoke: %s", text[:80])
        except asyncio.CancelledError:
            logger.info("Response speaking cancelled")
            raise
        except Exception:
            logger.exception("Error speaking response")
        finally:
            self.turn_manager.mark_bot_done()
