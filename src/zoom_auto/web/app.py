"""FastAPI application factory.

Creates and configures the FastAPI application with all routes
and middleware for the Zoom Auto web dashboard.
"""

from __future__ import annotations

from fastapi import FastAPI

from zoom_auto.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. Loads defaults if None.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Zoom Auto",
        description="AI-powered autonomous Zoom meeting participant",
        version="0.1.0",
    )

    # Import and include route modules
    from zoom_auto.web.routes import dashboard, meetings, persona, voice

    app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
    app.include_router(persona.router, prefix="/api/persona", tags=["persona"])
    app.include_router(meetings.router, prefix="/api/meetings", tags=["meetings"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app
