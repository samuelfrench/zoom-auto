"""Zoom Meeting SDK event handlers.

Handles meeting lifecycle events such as participant join/leave,
meeting start/end, and audio state changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class MeetingEvent(StrEnum):
    """Types of meeting events."""

    MEETING_JOINED = "meeting_joined"
    MEETING_LEFT = "meeting_left"
    MEETING_ENDED = "meeting_ended"
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    AUDIO_STARTED = "audio_started"
    AUDIO_STOPPED = "audio_stopped"
    SPEAKER_CHANGED = "speaker_changed"


@dataclass
class ParticipantInfo:
    """Information about a meeting participant."""

    user_id: int
    display_name: str
    is_host: bool = False
    is_muted: bool = False


EventCallback = Callable[[MeetingEvent, dict[str, Any]], None]


class ZoomEventHandler:
    """Handles Zoom Meeting SDK events and dispatches to registered callbacks.

    Provides a pub/sub interface for meeting lifecycle events.
    """

    def __init__(self) -> None:
        self._callbacks: dict[MeetingEvent, list[EventCallback]] = {}
        self._participants: dict[int, ParticipantInfo] = {}

    def on(self, event: MeetingEvent, callback: EventCallback) -> None:
        """Register a callback for a specific event type.

        Args:
            event: The event type to listen for.
            callback: Function to call when the event occurs.
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def emit(self, event: MeetingEvent, data: dict | None = None) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event: The event type.
            data: Event-specific data payload.
        """
        for callback in self._callbacks.get(event, []):
            callback(event, data or {})

    @property
    def participants(self) -> dict[int, ParticipantInfo]:
        """Currently tracked meeting participants."""
        return self._participants.copy()
