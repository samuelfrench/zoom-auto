"""Entry point and orchestrator for Zoom Auto.

Initializes all components, connects the audio pipeline, and starts
the web server for dashboard access.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import NoReturn

from zoom_auto.config import Settings

logger = logging.getLogger(__name__)


class ZoomAutoApp:
    """Main application orchestrator.

    Coordinates all subsystems: Zoom SDK connection, audio pipeline,
    STT/TTS engines, LLM providers, and the web dashboard.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._running = False

    async def start(self) -> None:
        """Initialize all components and start the application."""
        logger.info("Starting Zoom Auto v%s", "0.1.0")
        self._running = True
        raise NotImplementedError("Application startup not yet implemented")

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down Zoom Auto")
        self._running = False
        raise NotImplementedError("Application shutdown not yet implemented")

    @property
    def is_running(self) -> bool:
        """Whether the application is currently running."""
        return self._running


def main() -> NoReturn:
    """CLI entry point."""
    settings = Settings()  # type: ignore[call-arg]

    app = ZoomAutoApp(settings)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(app.stop()))

    try:
        loop.run_until_complete(app.start())
    except KeyboardInterrupt:
        loop.run_until_complete(app.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
