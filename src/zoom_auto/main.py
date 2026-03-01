"""Entry point and orchestrator for Zoom Auto.

Initializes all components, connects the audio pipeline, and starts
the web server for dashboard access.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
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
from zoom_auto.zoom.chat_sender import ChatSender
from zoom_auto.zoom.client import MeetingInfo, ZoomClient
from zoom_auto.zoom.events import MeetingEvent, ZoomEventHandler
from zoom_auto.zoom.url_parser import parse_meeting_input

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
        self.chat_sender = ChatSender(config=settings.zoom)

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

        # Pass the SDK instance to the chat sender so it can use the
        # SDK's chat controller for sending messages.
        if self.zoom_client.sdk_instance is not None:
            self.chat_sender.set_sdk(self.zoom_client.sdk_instance)

        logger.info("Joined meeting %s", meeting_id)

    async def send_chat_message(self, text: str) -> bool:
        """Send a message to the meeting chat.

        Args:
            text: The message text to send.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        return await self.chat_sender.send_message(text)

    async def _send_disclaimer(self) -> None:
        """Send disclaimer message to meeting chat after joining."""
        # Small delay to ensure chat is ready
        await asyncio.sleep(2)
        success = await self.chat_sender.send_disclaimer()
        if success:
            logger.info("Disclaimer sent to meeting chat")
        else:
            logger.warning("Failed to send disclaimer to meeting chat")

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

        def on_meeting_joined(
            event: MeetingEvent, data: dict[str, Any]
        ) -> None:
            if self.settings.zoom.send_disclaimer:
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._send_disclaimer())
                )

        self.event_handler.on(
            MeetingEvent.MEETING_JOINED, on_meeting_joined
        )
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="zoom-auto",
        description="AI-powered autonomous Zoom meeting participant",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Start command (default -- start the app/web server)
    subparsers.add_parser("start", help="Start the web server")

    # Join command -- join a meeting directly
    join_parser = subparsers.add_parser("join", help="Join a Zoom meeting")
    join_parser.add_argument(
        "meeting", help="Meeting ID or Zoom URL"
    )
    join_parser.add_argument(
        "--password", "-p", default="", help="Meeting password"
    )
    join_parser.add_argument(
        "--name", "-n", default=None, help="Display name"
    )

    # Index command -- index project directories for technical context
    index_parser = subparsers.add_parser(
        "index", help="Index project directories for technical context"
    )
    index_parser.add_argument(
        "paths", nargs="+", help="Project directories to index"
    )
    index_parser.add_argument(
        "--name", help="Override project name (only for single path)"
    )

    return parser


async def _run_join(
    app: ZoomAutoApp,
    meeting_id: str,
    password: str,
    display_name: str | None,
) -> None:
    """Start the app and join a meeting.

    Args:
        app: The ZoomAutoApp instance.
        meeting_id: Numeric Zoom meeting ID.
        password: Meeting password (empty string if none).
        display_name: Override display name (None uses config default).
    """
    if display_name:
        app.settings.zoom.bot_name = display_name

    logger.info(
        "Joining meeting %s as '%s'",
        meeting_id,
        app.settings.zoom.bot_name,
    )
    await app.start()
    await app.join_meeting(meeting_id, password)


async def _run_start(app: ZoomAutoApp) -> None:
    """Start the app in server mode, optionally auto-joining from env vars.

    Args:
        app: The ZoomAutoApp instance.
    """
    # Check for auto-join env vars
    meeting_id = os.environ.get("ZOOM_AUTO_MEETING_ID")
    meeting_password = os.environ.get("ZOOM_AUTO_MEETING_PASSWORD", "")

    await app.start()

    if meeting_id:
        logger.info("Auto-joining meeting %s from environment", meeting_id)
        await app.join_meeting(meeting_id, meeting_password)


def main() -> None:
    """CLI entry point with subcommands."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    app = ZoomAutoApp(settings)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(app.stop()))

    try:
        if args.command == "join":
            parsed = parse_meeting_input(args.meeting)
            password = args.password or parsed.password
            loop.run_until_complete(
                _run_join(app, parsed.meeting_id, password, args.name)
            )
        else:
            # Default: start web server (existing behavior)
            loop.run_until_complete(_run_start(app))
    except KeyboardInterrupt:
        loop.run_until_complete(app.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
