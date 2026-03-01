"""Entry point and orchestrator for Zoom Auto.

Initializes all components, connects the audio pipeline, and starts
the web server for dashboard access.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from zoom_auto.config import Settings
from zoom_auto.context.manager import ContextManager
from zoom_auto.llm.base import LLMProvider
from zoom_auto.llm.claude import ClaudeProvider
from zoom_auto.llm.ollama import OllamaProvider
from zoom_auto.pipeline.audio_pipeline import AudioPipeline
from zoom_auto.pipeline.conversation import ConversationLoop
from zoom_auto.pipeline.vad import VADProcessor
from zoom_auto.response.decision import TriggerDetector
from zoom_auto.response.generator import ResponseGenerator
from zoom_auto.response.turn_manager import TurnManager
from zoom_auto.stt.faster_whisper import FasterWhisperEngine
from zoom_auto.tts.chatterbox import ChatterboxEngine
from zoom_auto.zoom.audio_capture import AudioCapture
from zoom_auto.zoom.audio_sender import AudioSender
from zoom_auto.zoom.client import MeetingInfo, ZoomClient
from zoom_auto.zoom.events import MeetingEvent, ZoomEventHandler

logger = logging.getLogger(__name__)


class ZoomAutoApp:
    """Main application orchestrator.

    Coordinates all subsystems: Zoom SDK connection, audio pipeline,
    STT/TTS engines, LLM providers, and the web dashboard.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._running = False

        # -- Zoom SDK layer --
        self.event_handler = ZoomEventHandler()
        self.zoom_client = ZoomClient(settings, self.event_handler)
        self.audio_capture = AudioCapture(
            config=settings.zoom,
            target_sample_rate=settings.zoom.sample_rate,
        )
        self.audio_sender = AudioSender(config=settings.zoom)

        # -- VAD --
        self.vad = VADProcessor(config=settings.vad)

        # -- STT --
        self.stt = FasterWhisperEngine(config=settings.stt)

        # -- TTS --
        self.tts = ChatterboxEngine(config=settings.tts)

        # -- LLM provider --
        self.llm: LLMProvider = self._create_llm_provider()

        # -- Context --
        self.context_manager = ContextManager(
            config=settings.context, llm=self.llm
        )

        # -- Response engine --
        self.trigger_detector = TriggerDetector(
            config=settings.response, llm=self.llm
        )
        self.response_generator = ResponseGenerator(
            llm=self.llm,
            context_manager=self.context_manager,
        )
        self.turn_manager = TurnManager(config=settings.response)

        # -- Pipelines --
        self.audio_pipeline = AudioPipeline(
            settings=settings,
            capture=self.audio_capture,
            sender=self.audio_sender,
            vad=self.vad,
            stt=self.stt,
            tts=self.tts,
        )
        self.conversation_loop = ConversationLoop(
            settings=settings,
            audio_pipeline=self.audio_pipeline,
            context_manager=self.context_manager,
            trigger_detector=self.trigger_detector,
            response_generator=self.response_generator,
            turn_manager=self.turn_manager,
        )

        # Wire up Zoom events
        self._register_event_handlers()

    async def start(self) -> None:
        """Initialize all components and start the application."""
        logger.info("Starting Zoom Auto v%s", "0.1.0")
        self._running = True
        self._loop = asyncio.get_running_loop()

        # Load ML models (STT, TTS, VAD)
        logger.info("Loading ML models...")
        await asyncio.gather(
            self.stt.load_model(),
            self.tts.load_model(),
            self.vad.load_model(),
        )
        logger.info("All ML models loaded")

        # Start audio capture and sender
        await self.audio_capture.start()
        await self.audio_sender.start()

        # Start the conversation loop (which starts the audio pipeline)
        await self.conversation_loop.start()

        logger.info("Zoom Auto is running")

        # Keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down Zoom Auto")
        self._running = False

        # Stop conversation loop (stops audio pipeline internally)
        await self.conversation_loop.stop()

        # Stop audio capture and sender
        await self.audio_capture.stop()
        await self.audio_sender.stop()

        # Leave meeting if connected
        if self.zoom_client.is_connected:
            await self.zoom_client.leave()

        # Unload ML models
        logger.info("Unloading ML models...")
        await self.stt.unload_model()
        await self.tts.unload_model()
        await self.vad.unload_model()

        # Close LLM provider if it has a close method
        if hasattr(self.llm, "close"):
            await self.llm.close()  # type: ignore[attr-defined]

        logger.info("Zoom Auto shut down complete")

    async def join_meeting(
        self, meeting_id: str, password: str = ""
    ) -> None:
        """Join a Zoom meeting.

        Args:
            meeting_id: The Zoom meeting ID.
            password: Meeting password (if required).
        """
        meeting = MeetingInfo(
            meeting_id=meeting_id,
            password=password,
            display_name=self.settings.zoom.bot_name,
        )
        await self.zoom_client.join(meeting)
        logger.info("Joined meeting %s", meeting_id)

    @property
    def is_running(self) -> bool:
        """Whether the application is currently running."""
        return self._running

    def _create_llm_provider(self) -> LLMProvider:
        """Create the LLM provider based on configuration.

        Returns:
            An LLMProvider instance (ClaudeProvider or OllamaProvider).
        """
        if self.settings.llm.provider == "claude":
            if not self.settings.anthropic_api_key:
                logger.warning(
                    "Claude provider selected but no API key set, "
                    "falling back to Ollama"
                )
                return OllamaProvider(
                    config=self.settings.llm,
                    host=self.settings.ollama_host,
                )
            return ClaudeProvider(
                config=self.settings.llm,
                api_key=self.settings.anthropic_api_key,
            )

        return OllamaProvider(
            config=self.settings.llm,
            host=self.settings.ollama_host,
        )

    def _register_event_handlers(self) -> None:
        """Register callbacks for Zoom meeting events."""

        def on_participant_joined(
            event: MeetingEvent, data: dict[str, Any]
        ) -> None:
            participant = data.get("participant")
            if participant:
                self.context_manager.meeting_state.add_participant(
                    participant.display_name
                )
                self.audio_capture.set_speaker_name(
                    participant.user_id, participant.display_name
                )
                logger.info(
                    "Participant joined: %s", participant.display_name
                )

        def on_participant_left(
            event: MeetingEvent, data: dict[str, Any]
        ) -> None:
            participant = data.get("participant")
            if participant:
                logger.info(
                    "Participant left: %s", participant.display_name
                )

        def on_meeting_ended(
            event: MeetingEvent, data: dict[str, Any]
        ) -> None:
            logger.info("Meeting ended -- initiating shutdown")
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.stop())
            )

        def on_speaker_changed(
            event: MeetingEvent, data: dict[str, Any]
        ) -> None:
            user_id = data.get("user_id")
            if user_id is not None:
                self.turn_manager.on_speech_detected()

        self.event_handler.on(
            MeetingEvent.PARTICIPANT_JOINED, on_participant_joined
        )
        self.event_handler.on(
            MeetingEvent.PARTICIPANT_LEFT, on_participant_left
        )
        self.event_handler.on(
            MeetingEvent.MEETING_ENDED, on_meeting_ended
        )
        self.event_handler.on(
            MeetingEvent.SPEAKER_CHANGED, on_speaker_changed
        )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    settings = Settings()  # type: ignore[call-arg]

    app = ZoomAutoApp(settings)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(app.stop()))

    try:
        loop.run_until_complete(app.start())
    except KeyboardInterrupt:
        loop.run_until_complete(app.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
