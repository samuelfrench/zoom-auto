"""Live transcript accumulator.

Accumulates transcribed utterances into a structured meeting transcript
with speaker attribution and timestamps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    """A single entry in the meeting transcript.

    Attributes:
        speaker: Name or ID of the speaker.
        text: The transcribed text.
        timestamp: When this was said.
        confidence: STT confidence score.
    """

    speaker: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


class TranscriptAccumulator:
    """Accumulates live transcript entries during a meeting.

    Provides methods for adding entries, querying recent history,
    and exporting the full transcript.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []

    def add(self, speaker: str, text: str, confidence: float = 0.0) -> None:
        """Add a new transcript entry.

        Args:
            speaker: Speaker name or identifier.
            text: The transcribed text.
            confidence: STT confidence score.
        """
        self._entries.append(
            TranscriptEntry(speaker=speaker, text=text, confidence=confidence)
        )

    def recent(self, n: int = 10) -> list[TranscriptEntry]:
        """Get the N most recent transcript entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of recent TranscriptEntry objects.
        """
        return self._entries[-n:]

    def format_recent(self, n: int = 10) -> str:
        """Format recent entries as readable text.

        Args:
            n: Number of entries to format.

        Returns:
            Formatted transcript string.
        """
        entries = self.recent(n)
        return "\n".join(f"{e.speaker}: {e.text}" for e in entries)

    def clear(self) -> None:
        """Clear all transcript entries."""
        self._entries.clear()

    @property
    def entries(self) -> list[TranscriptEntry]:
        """All transcript entries."""
        return self._entries.copy()

    def __len__(self) -> int:
        return len(self._entries)
