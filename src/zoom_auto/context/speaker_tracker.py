"""Speaker tracking and diarization.

Tracks who is speaking, maps speaker IDs to names,
and maintains per-speaker statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    """

    speaker_id: int
    name: str
    total_speaking_time: float = 0.0
    utterance_count: int = 0
    is_active: bool = False


class SpeakerTracker:
    """Tracks meeting speakers and their activity.

    Maintains a mapping of speaker IDs to names and tracks
    per-speaker statistics like speaking time and utterance count.
    """

    def __init__(self) -> None:
        self._speakers: dict[int, SpeakerInfo] = {}
        self._active_speaker_id: int | None = None

    def register_speaker(self, speaker_id: int, name: str) -> None:
        """Register a new speaker or update their name.

        Args:
            speaker_id: Unique speaker identifier.
            name: Display name.
        """
        if speaker_id in self._speakers:
            self._speakers[speaker_id].name = name
        else:
            self._speakers[speaker_id] = SpeakerInfo(speaker_id=speaker_id, name=name)

    def set_active(self, speaker_id: int) -> None:
        """Mark a speaker as currently active (speaking).

        Args:
            speaker_id: The speaker who started speaking.
        """
        if self._active_speaker_id is not None and self._active_speaker_id in self._speakers:
            self._speakers[self._active_speaker_id].is_active = False
        if speaker_id in self._speakers:
            self._speakers[speaker_id].is_active = True
        self._active_speaker_id = speaker_id

    def get_name(self, speaker_id: int) -> str:
        """Get the display name for a speaker ID.

        Args:
            speaker_id: The speaker identifier.

        Returns:
            Display name, or "Unknown" if not registered.
        """
        info = self._speakers.get(speaker_id)
        return info.name if info else "Unknown"

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
