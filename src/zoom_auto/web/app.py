"""FastAPI application factory.

Creates and configures the FastAPI application with all routes
and middleware for the Zoom Auto web dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from zoom_auto.config import Settings

if TYPE_CHECKING:
    from zoom_auto.main import ZoomAutoApp


def create_app(
    settings: Settings | None = None,
    zoom_app: ZoomAutoApp | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. Loads defaults if None.
        zoom_app: Optional ZoomAutoApp instance to wire into routes.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Zoom Auto",
        description="AI-powered autonomous Zoom meeting participant",
        version="0.1.0",
    )

    # CORS middleware for development (Vite dev server on port 5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include route modules
    from zoom_auto.web.routes import dashboard, meetings, persona, voice

    app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
    app.include_router(persona.router, prefix="/api/persona", tags=["persona"])
    app.include_router(meetings.router, prefix="/api/meetings", tags=["meetings"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

    # Wire the ZoomAutoApp instance into route modules
    if zoom_app is not None:
        meetings.set_app_instance(zoom_app)
        dashboard.set_app_instance(zoom_app)

    # Wire persona config
    persona.set_persona_config(settings.persona)

    # Settings endpoint for the frontend
    @app.get("/api/settings")
    async def get_settings() -> dict:
        """Get current application settings (read-only)."""
        return {
            "zoom": {
                "bot_name": settings.zoom.bot_name,
            },
            "llm": {
                "provider": settings.llm.provider,
                "response_model": settings.llm.response_model,
                "decision_model": settings.llm.decision_model,
                "max_tokens": settings.llm.max_tokens,
                "temperature": settings.llm.temperature,
            },
            "stt": {
                "model": settings.stt.model,
                "language": settings.stt.language,
                "beam_size": settings.stt.beam_size,
            },
            "tts": {
                "voice_sample_dir": settings.tts.voice_sample_dir,
                "sample_rate": settings.tts.sample_rate,
            },
            "context": {
                "max_window_tokens": settings.context.max_window_tokens,
                "verbatim_window_seconds": settings.context.verbatim_window_seconds,
                "summary_interval_seconds": settings.context.summary_interval_seconds,
            },
            "response": {
                "cooldown_seconds": settings.response.cooldown_seconds,
                "trigger_threshold": settings.response.trigger_threshold,
                "max_consecutive": settings.response.max_consecutive,
            },
            "vad": {
                "threshold": settings.vad.threshold,
                "min_speech_duration": settings.vad.min_speech_duration,
            },
        }

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Serve frontend static files (production build)
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app
