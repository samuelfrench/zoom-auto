"""Zoom Meeting SDK client for joining and leaving meetings.

Uses py-zoom-meeting-sdk to connect to Zoom meetings as a bot participant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from zoom_auto.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class MeetingInfo:
    """Information about a Zoom meeting to join."""

    meeting_id: str
    password: str = ""
    display_name: str = "AI Assistant"


class ZoomClient:
    """Manages Zoom Meeting SDK connection lifecycle.

    Handles authentication, joining meetings, and graceful disconnection.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._connected = False
        self._meeting_info: MeetingInfo | None = None

    async def join(self, meeting: MeetingInfo) -> None:
        """Join a Zoom meeting using the Meeting SDK.

        Args:
            meeting: Meeting ID, password, and display name.
        """
        raise NotImplementedError("Zoom SDK join not yet implemented")

    async def leave(self) -> None:
        """Leave the current Zoom meeting gracefully."""
        raise NotImplementedError("Zoom SDK leave not yet implemented")

    @property
    def is_connected(self) -> bool:
        """Whether currently connected to a meeting."""
        return self._connected

    @property
    def meeting_info(self) -> MeetingInfo | None:
        """Information about the current meeting, if connected."""
        return self._meeting_info
