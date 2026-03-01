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
from pathlib import Path
from typing import Any

from zoom_auto.config import Settings
from zoom_auto.context.manager import ContextManager
from zoom_auto.llm.base import LLMProvider
from zoom_auto.llm.claude import ClaudeProvider
from zoom_auto.llm.ollama import OllamaProvider
from zoom_auto.persona.learner import ConversationLearner
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

        # -- Conversation learner --
        self.learner = ConversationLearner(
            data_dir=Path("data/learnings"),
            user=settings.zoom.bot_name.lower().replace(" ", "_"),
        )

        # -- Response engine --
        self.trigger_detector = TriggerDetector(
            config=settings.response, llm=self.llm
        )
        self.response_generator = ResponseGenerator(
            llm=self.llm,
            context_manager=self.context_manager,
            learner=self.learner,
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
            learner=self.learner,
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

    # Learnings command -- view accumulated learnings
    learn_parser = subparsers.add_parser(
        "learnings", help="View accumulated learnings"
    )
    learn_parser.add_argument(
        "--user", default="default", help="Username"
    )
    learn_parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of accumulated learnings",
    )
    learn_parser.add_argument(
        "--rebuild-persona",
        action="store_true",
        help="Rebuild persona from learnings",
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


def _run_index(paths: list[str], name_override: str | None) -> None:
    """Index project directories and save to the knowledge store.

    Args:
        paths: List of project directory paths to index.
        name_override: Optional name override (single project only).
    """
    from pathlib import Path

    from zoom_auto.persona.knowledge_store import KnowledgeStore
    from zoom_auto.persona.sources.project import ProjectIndexer

    indexer = ProjectIndexer()
    store = KnowledgeStore()

    resolved = [Path(p).resolve() for p in paths]
    indices = indexer.index_multiple(resolved)

    if name_override and len(indices) == 1:
        indices[0].name = name_override

    for idx in indices:
        store.save_index(idx)
        print(f"Indexed: {idx.name}")
        print(f"  Path: {idx.root_path}")
        print(f"  Tech stack: {', '.join(idx.tech_stack) or 'none detected'}")
        print(f"  Dependencies: {len(idx.dependencies)}")
        print(f"  Patterns: {', '.join(idx.patterns) or 'none detected'}")
        print(f"  Files: {idx.total_files}")
        print()

    print(f"Done. Indexed {len(indices)} project(s).")


def _run_learnings(user: str, summary: bool, rebuild_persona: bool) -> None:
    """Display accumulated learnings for a user.

    Args:
        user: Username to look up learnings for.
        summary: Whether to show a summary.
        rebuild_persona: Whether to rebuild persona from learnings.
    """
    import json as _json

    learner = ConversationLearner(
        data_dir=Path("data/learnings"), user=user,
    )

    sessions = learner.get_transcript_files()
    if not sessions:
        print(f"No learnings found for user '{user}'.")
        print(f"  Data directory: data/learnings/{user}/sessions/")
        return

    print(f"Learnings for user '{user}':")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Vocabulary size: {len(learner._vocabulary)}")

    if summary or not rebuild_persona:
        context = learner.get_learning_context()
        if context:
            print(f"\n{context}")
        else:
            print("  No learning context generated yet.")

        # Show recent sessions
        print("\nRecent sessions (last 5):")
        for path in sessions[-5:]:
            try:
                data = _json.loads(path.read_text())
                n_utterances = len(data.get("transcript", []))
                meeting_type = data.get("meeting_type", "unknown")
                topics = data.get("topics_discussed", [])
                topic_str = ", ".join(topics[:3]) if topics else "none"
                print(
                    f"  {path.stem}: "
                    f"{n_utterances} utterances, "
                    f"type={meeting_type}, "
                    f"topics=[{topic_str}]"
                )
            except Exception:
                print(f"  {path.stem}: (error reading)")

    if rebuild_persona:
        print("\nRebuilding persona from learnings...")
        # Collect all transcript text
        texts: list[str] = []
        for path in sessions:
            try:
                data = _json.loads(path.read_text())
                for entry in data.get("transcript", []):
                    texts.append(entry.get("text", ""))
            except Exception:
                continue

        if texts:
            from zoom_auto.persona.builder import PersonaBuilder

            builder = PersonaBuilder()
            profile = builder.build_from_texts(
                texts, name=user, source_type="transcript"
            )
            output_path = Path(f"config/personas/{user}_learned.toml")
            profile.to_toml(output_path)
            print(f"  Persona saved to: {output_path}")
        else:
            print("  No transcript data found for persona building.")


def main() -> None:
    """CLI entry point with subcommands."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Handle non-app commands first (no need to create ZoomAutoApp)
    if args.command == "index":
        _run_index(args.paths, args.name)
        return

    if args.command == "learnings":
        _run_learnings(args.user, args.summary, args.rebuild_persona)
        return

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
