"""Meeting join and monitoring endpoints.

Provides API endpoints for joining Zoom meetings and monitoring
the bot's participation status.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


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
    raise NotImplementedError("Meeting join not yet implemented")


@router.post("/leave")
async def leave_meeting() -> dict[str, str]:
    """Leave the current meeting."""
    raise NotImplementedError("Meeting leave not yet implemented")


@router.get("/status")
async def get_meeting_status() -> MeetingStatusResponse:
    """Get the current meeting status."""
    raise NotImplementedError("Meeting status not yet implemented")
