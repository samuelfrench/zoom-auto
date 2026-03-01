"""Meeting join and monitoring endpoints.

Provides API endpoints for joining Zoom meetings and monitoring
the bot's participation status.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level reference to the ZoomAutoApp instance
_app_instance: object | None = None
_meeting_start_time: float | None = None


def set_app_instance(app: object) -> None:
    """Set the ZoomAutoApp reference for meeting management.

    Args:
        app: The ZoomAutoApp instance.
    """
    global _app_instance
    _app_instance = app


class MeetingJoinRequest(BaseModel):
    """Request model for joining a meeting."""

    meeting_id: str
    password: str = ""
    display_name: str = "AI Assistant"


class MeetingStatusResponse(BaseModel):
    """Response model for meeting status."""

    connected: bool
    meeting_id: str | None = None
    participants: int = 0
    duration_seconds: float = 0.0
    utterances_count: int = 0
    responses_count: int = 0


@router.post("/join")
async def join_meeting(request: MeetingJoinRequest) -> dict[str, str]:
    """Join a Zoom meeting.

    Args:
        request: Meeting ID, password, and display name.
    """
    global _meeting_start_time

    if _app_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Application not initialized. Start the app first.",
        )

    app = _app_instance

    # Check if already connected
    if hasattr(app, "zoom_client") and app.zoom_client.is_connected:
        raise HTTPException(
            status_code=409,
            detail="Already connected to a meeting. Leave first.",
        )

    # Update display name in settings if provided
    if request.display_name and hasattr(app, "settings"):
        app.settings.zoom.bot_name = request.display_name

    try:
        await app.join_meeting(
            meeting_id=request.meeting_id,
            password=request.password,
        )
        _meeting_start_time = time.time()
        return {
            "status": "ok",
            "message": f"Joining meeting {request.meeting_id}",
        }
    except Exception as e:
        logger.error("Failed to join meeting: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to join meeting: {e}",
        ) from e


@router.post("/leave")
async def leave_meeting() -> dict[str, str]:
    """Leave the current meeting."""
    global _meeting_start_time

    if _app_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Application not initialized.",
        )

    app = _app_instance

    if hasattr(app, "zoom_client") and not app.zoom_client.is_connected:
        return {"status": "ok", "message": "Not currently in a meeting"}

    try:
        if hasattr(app, "zoom_client"):
            await app.zoom_client.leave()
        _meeting_start_time = None
        return {"status": "ok", "message": "Left the meeting"}
    except Exception as e:
        logger.error("Failed to leave meeting: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to leave meeting: {e}",
        ) from e


@router.get("/status", response_model=MeetingStatusResponse)
async def get_meeting_status() -> MeetingStatusResponse:
    """Get the current meeting status."""
    if _app_instance is None:
        return MeetingStatusResponse(connected=False)

    app = _app_instance
    connected = False
    meeting_id = None
    participants = 0
    duration_seconds = 0.0
    utterances_count = 0
    responses_count = 0

    # Check Zoom connection
    if hasattr(app, "zoom_client"):
        connected = app.zoom_client.is_connected
        if connected and app.zoom_client.meeting_info:
            meeting_id = app.zoom_client.meeting_info.meeting_id

    # Calculate duration
    if connected and _meeting_start_time is not None:
        duration_seconds = round(time.time() - _meeting_start_time, 1)

    # Get participant count from context manager
    if hasattr(app, "context_manager"):
        ctx = app.context_manager
        if hasattr(ctx, "meeting_state"):
            participants = len(ctx.meeting_state.participants)
        if hasattr(ctx, "transcript"):
            utterances_count = len(ctx.transcript.entries)

    # Get response count from conversation loop
    if hasattr(app, "conversation_loop"):
        loop = app.conversation_loop
        if hasattr(loop, "responses_generated"):
            responses_count = loop.responses_generated

    return MeetingStatusResponse(
        connected=connected,
        meeting_id=meeting_id,
        participants=participants,
        duration_seconds=duration_seconds,
        utterances_count=utterances_count,
        responses_count=responses_count,
    )
