"""Speaker tracking and diarization.

Tracks who is speaking, maps speaker IDs to names,
and maintains per-speaker statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Information about a tracked speaker.

    Attributes:
        speaker_id: Unique identifier from the Zoom SDK.
        name: Display name.
        total_speaking_time: Total seconds of speech.
        utterance_count: Number of utterances.
        is_active: Whether currently speaking.
        first_seen: When the speaker was first registered.
        last_spoke: When the speaker last spoke.
    """

    speaker_id: int
    name: str
    total_speaking_time: float = 0.0
    utterance_count: int = 0
    is_active: bool = False
    first_seen: datetime = field(default_factory=datetime.now)
    last_spoke: datetime | None = None


class SpeakerTracker:
    """Tracks meeting speakers and their activity.

    Maintains a mapping of speaker IDs to names and tracks
    per-speaker statistics like speaking time and utterance count.
    Maps Zoom user IDs to display names.
    """

    def __init__(self) -> None:
        self._speakers: dict[int, SpeakerInfo] = {}
        self._active_speaker_id: int | None = None
        self._active_start_time: float | None = None

    def register_speaker(self, speaker_id: int, name: str) -> SpeakerInfo:
        """Register a new speaker or update their name.

        Args:
            speaker_id: Unique speaker identifier (Zoom user ID).
            name: Display name.

        Returns:
            The SpeakerInfo for the registered speaker.
        """
        if speaker_id in self._speakers:
            self._speakers[speaker_id].name = name
            logger.debug("Updated speaker %d name to %s", speaker_id, name)
        else:
            self._speakers[speaker_id] = SpeakerInfo(
                speaker_id=speaker_id,
                name=name,
            )
            logger.debug("Registered new speaker %d: %s", speaker_id, name)
        return self._speakers[speaker_id]

    def set_active(self, speaker_id: int) -> None:
        """Mark a speaker as currently active (speaking).

        Handles deactivating the previous speaker and tracking
        their speaking time.

        Args:
            speaker_id: The speaker who started speaking.
        """
        now = datetime.now().timestamp()

        # Deactivate previous speaker and record their speaking time
        if self._active_speaker_id is not None and self._active_speaker_id in self._speakers:
            prev = self._speakers[self._active_speaker_id]
            prev.is_active = False
            if self._active_start_time is not None:
                duration = now - self._active_start_time
                prev.total_speaking_time += duration

        # Activate new speaker
        if speaker_id in self._speakers:
            self._speakers[speaker_id].is_active = True
            self._speakers[speaker_id].last_spoke = datetime.now()
        self._active_speaker_id = speaker_id
        self._active_start_time = now

    def clear_active(self) -> None:
        """Clear the active speaker (nobody is speaking).

        Records the speaking time for the previously active speaker.
        """
        now = datetime.now().timestamp()
        if self._active_speaker_id is not None and self._active_speaker_id in self._speakers:
            prev = self._speakers[self._active_speaker_id]
            prev.is_active = False
            if self._active_start_time is not None:
                duration = now - self._active_start_time
                prev.total_speaking_time += duration

        self._active_speaker_id = None
        self._active_start_time = None

    def record_utterance(self, speaker_id: int, duration: float = 0.0) -> None:
        """Record that a speaker made an utterance.

        Args:
            speaker_id: The speaker who spoke.
            duration: Duration of the utterance in seconds.
        """
        if speaker_id in self._speakers:
            self._speakers[speaker_id].utterance_count += 1
            self._speakers[speaker_id].last_spoke = datetime.now()
            if duration > 0:
                self._speakers[speaker_id].total_speaking_time += duration
        else:
            logger.warning(
                "Utterance recorded for unregistered speaker %d", speaker_id
            )

    def get_name(self, speaker_id: int) -> str:
        """Get the display name for a speaker ID.

        Args:
            speaker_id: The speaker identifier.

        Returns:
            Display name, or "Unknown" if not registered.
        """
        info = self._speakers.get(speaker_id)
        return info.name if info else "Unknown"

    def get_speaker(self, speaker_id: int) -> SpeakerInfo | None:
        """Get the full SpeakerInfo for a speaker ID.

        Args:
            speaker_id: The speaker identifier.

        Returns:
            SpeakerInfo or None if not registered.
        """
        return self._speakers.get(speaker_id)

    def find_by_name(self, name: str) -> SpeakerInfo | None:
        """Find a speaker by display name (case-insensitive).

        Args:
            name: Display name to search for.

        Returns:
            SpeakerInfo or None if not found.
        """
        name_lower = name.lower()
        for info in self._speakers.values():
            if info.name.lower() == name_lower:
                return info
        return None

    @property
    def active_speaker(self) -> SpeakerInfo | None:
        """The currently active speaker, if any."""
        if self._active_speaker_id is None:
            return None
        return self._speakers.get(self._active_speaker_id)

    @property
    def all_speakers(self) -> list[SpeakerInfo]:
        """List of all tracked speakers."""
        return list(self._speakers.values())

    @property
    def participant_names(self) -> list[str]:
        """List of all participant display names."""
        return [s.name for s in self._speakers.values()]

    @property
    def speaker_count(self) -> int:
        """Number of tracked speakers."""
        return len(self._speakers)

    def format_speaker_list(self) -> str:
        """Format the speaker list for LLM context.

        Returns:
            Formatted string listing participants and their stats.
        """
        if not self._speakers:
            return "No participants tracked."
        lines = []
        for info in self._speakers.values():
            status = " (speaking)" if info.is_active else ""
            lines.append(
                f"- {info.name}{status}: "
                f"{info.utterance_count} utterances, "
                f"{info.total_speaking_time:.0f}s speaking time"
            )
        return "Participants:\n" + "\n".join(lines)

    def reset(self) -> None:
        """Reset all speaker tracking for a new meeting."""
        self._speakers.clear()
        self._active_speaker_id = None
        self._active_start_time = None
        logger.debug("Speaker tracker reset")
