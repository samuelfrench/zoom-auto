"""Zoom Meeting SDK event handlers.

Handles meeting lifecycle events such as participant join/leave,
meeting start/end, and audio state changes.

SDK callbacks are translated to our internal event system so that
higher-level components never depend on the Zoom SDK directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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
    CONNECTION_ERROR = "connection_error"


@dataclass
class ParticipantInfo:
    """Information about a meeting participant."""

    user_id: int
    display_name: str
    is_host: bool = False
    is_muted: bool = False


EventCallback = Callable[[MeetingEvent, dict[str, Any]], None]


@dataclass
class ZoomEventHandler:
    """Handles Zoom Meeting SDK events and dispatches to registered callbacks.

    Provides a pub/sub interface for meeting lifecycle events.
    SDK callback methods (on_meeting_joined, on_participant_joined, etc.)
    translate raw SDK data into our internal MeetingEvent system.
    """

    _callbacks: dict[MeetingEvent, list[EventCallback]] = field(
        default_factory=dict
    )
    _participants: dict[int, ParticipantInfo] = field(default_factory=dict)

    def on(self, event: MeetingEvent, callback: EventCallback) -> None:
        """Register a callback for a specific event type.

        Args:
            event: The event type to listen for.
            callback: Function to call when the event occurs.
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(
        self, event: MeetingEvent, callback: EventCallback
    ) -> None:
        """Remove a previously registered callback.

        Args:
            event: The event type.
            callback: The callback to remove.
        """
        callbacks = self._callbacks.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def emit(self, event: MeetingEvent, data: dict | None = None) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event: The event type.
            data: Event-specific data payload.
        """
        payload = data or {}
        logger.debug("Event emitted: %s data=%s", event, payload)
        for callback in self._callbacks.get(event, []):
            try:
                callback(event, payload)
            except Exception:
                logger.exception(
                    "Error in event callback for %s", event
                )

    @property
    def participants(self) -> dict[int, ParticipantInfo]:
        """Currently tracked meeting participants."""
        return self._participants.copy()

    def get_participant(self, user_id: int) -> ParticipantInfo | None:
        """Look up a participant by user ID."""
        return self._participants.get(user_id)

    # ------------------------------------------------------------------ #
    #  SDK callback handlers                                              #
    #  These methods should be registered with the Zoom Meeting SDK.      #
    #  They normalise SDK data and emit internal events.                  #
    # ------------------------------------------------------------------ #

    def on_meeting_joined(self) -> None:
        """Called by SDK when we successfully join a meeting."""
        logger.info("Successfully joined meeting")
        self.emit(MeetingEvent.MEETING_JOINED)

    def on_meeting_left(self) -> None:
        """Called by SDK when we leave / are removed from a meeting."""
        logger.info("Left meeting")
        self._participants.clear()
        self.emit(MeetingEvent.MEETING_LEFT)

    def on_meeting_ended(self) -> None:
        """Called by SDK when the host ends the meeting."""
        logger.info("Meeting ended by host")
        self._participants.clear()
        self.emit(MeetingEvent.MEETING_ENDED)

    def on_participant_joined(
        self,
        user_id: int,
        display_name: str,
        is_host: bool = False,
    ) -> None:
        """Called by SDK when a participant joins.

        Args:
            user_id: Zoom user ID of the participant.
            display_name: Display name of the participant.
            is_host: Whether the participant is the meeting host.
        """
        info = ParticipantInfo(
            user_id=user_id,
            display_name=display_name,
            is_host=is_host,
        )
        self._participants[user_id] = info
        logger.info(
            "Participant joined: %s (id=%d, host=%s)",
            display_name,
            user_id,
            is_host,
        )
        self.emit(
            MeetingEvent.PARTICIPANT_JOINED,
            {"participant": info},
        )

    def on_participant_left(self, user_id: int) -> None:
        """Called by SDK when a participant leaves.

        Args:
            user_id: Zoom user ID of the participant.
        """
        info = self._participants.pop(user_id, None)
        name = info.display_name if info else f"user_{user_id}"
        logger.info("Participant left: %s (id=%d)", name, user_id)
        self.emit(
            MeetingEvent.PARTICIPANT_LEFT,
            {"user_id": user_id, "participant": info},
        )

    def on_audio_started(self, user_id: int) -> None:
        """Called by SDK when a participant unmutes / starts audio.

        Args:
            user_id: Zoom user ID.
        """
        info = self._participants.get(user_id)
        if info:
            info.is_muted = False
        self.emit(
            MeetingEvent.AUDIO_STARTED,
            {"user_id": user_id, "participant": info},
        )

    def on_audio_stopped(self, user_id: int) -> None:
        """Called by SDK when a participant mutes / stops audio.

        Args:
            user_id: Zoom user ID.
        """
        info = self._participants.get(user_id)
        if info:
            info.is_muted = True
        self.emit(
            MeetingEvent.AUDIO_STOPPED,
            {"user_id": user_id, "participant": info},
        )

    def on_speaker_changed(self, user_id: int) -> None:
        """Called by SDK when the active speaker changes.

        Args:
            user_id: Zoom user ID of the new active speaker.
        """
        info = self._participants.get(user_id)
        self.emit(
            MeetingEvent.SPEAKER_CHANGED,
            {"user_id": user_id, "participant": info},
        )

    def on_connection_error(self, error_code: int, message: str) -> None:
        """Called by SDK on connection errors.

        Args:
            error_code: SDK error code.
            message: Human-readable error message.
        """
        logger.error(
            "Zoom connection error %d: %s", error_code, message
        )
        self.emit(
            MeetingEvent.CONNECTION_ERROR,
            {"error_code": error_code, "message": message},
        )
