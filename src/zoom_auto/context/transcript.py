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
        segment_id: Unique identifier for this entry.
    """

    speaker: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    segment_id: int = 0


class TranscriptAccumulator:
    """Accumulates live transcript entries during a meeting.

    Provides methods for adding entries, querying recent history,
    getting entries within a time window, and formatting for LLM consumption.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self._next_segment_id: int = 1

    def add(
        self,
        speaker: str,
        text: str,
        confidence: float = 0.0,
        timestamp: datetime | None = None,
    ) -> TranscriptEntry:
        """Add a new transcript entry.

        Args:
            speaker: Speaker name or identifier.
            text: The transcribed text.
            confidence: STT confidence score.
            timestamp: When this was said. Defaults to now.

        Returns:
            The created TranscriptEntry.
        """
        entry = TranscriptEntry(
            speaker=speaker,
            text=text,
            confidence=confidence,
            timestamp=timestamp or datetime.now(),
            segment_id=self._next_segment_id,
        )
        self._next_segment_id += 1
        self._entries.append(entry)
        logger.debug(
            "Transcript entry added: [%s] %s: %s",
            entry.segment_id,
            speaker,
            text[:50],
        )
        return entry

    def recent(self, n: int = 10) -> list[TranscriptEntry]:
        """Get the N most recent transcript entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of recent TranscriptEntry objects.
        """
        return self._entries[-n:]

    def get_window(self, seconds: float, reference_time: datetime | None = None) -> list[TranscriptEntry]:
        """Get entries within a time window.

        Returns all entries from the last `seconds` seconds relative
        to `reference_time` (defaults to now).

        Args:
            seconds: Window size in seconds.
            reference_time: Reference point for the window. Defaults to now.

        Returns:
            List of TranscriptEntry objects within the window.
        """
        ref = reference_time or datetime.now()
        cutoff = ref.timestamp() - seconds
        return [
            e for e in self._entries
            if e.timestamp.timestamp() >= cutoff
        ]

    def get_before(self, seconds: float, reference_time: datetime | None = None) -> list[TranscriptEntry]:
        """Get entries older than the given time window.

        Returns all entries from before `seconds` seconds ago relative
        to `reference_time` (defaults to now).

        Args:
            seconds: Time threshold in seconds.
            reference_time: Reference point. Defaults to now.

        Returns:
            List of TranscriptEntry objects before the window.
        """
        ref = reference_time or datetime.now()
        cutoff = ref.timestamp() - seconds
        return [
            e for e in self._entries
            if e.timestamp.timestamp() < cutoff
        ]

    def remove_before(self, cutoff_time: datetime) -> int:
        """Remove entries older than the given time.

        Args:
            cutoff_time: Remove entries with timestamps before this.

        Returns:
            Number of entries removed.
        """
        original_count = len(self._entries)
        cutoff_ts = cutoff_time.timestamp()
        self._entries = [
            e for e in self._entries
            if e.timestamp.timestamp() >= cutoff_ts
        ]
        removed = original_count - len(self._entries)
        if removed > 0:
            logger.debug("Removed %d old transcript entries", removed)
        return removed

    def format_recent(self, n: int = 10) -> str:
        """Format recent entries as readable text.

        Args:
            n: Number of entries to format.

        Returns:
            Formatted transcript string.
        """
        entries = self.recent(n)
        return self._format_entries(entries)

    def format_window(self, seconds: float, reference_time: datetime | None = None) -> str:
        """Format entries within a time window as readable text.

        Args:
            seconds: Window size in seconds.
            reference_time: Reference point for the window.

        Returns:
            Formatted transcript string with "Speaker (HH:MM): text" format.
        """
        entries = self.get_window(seconds, reference_time)
        return self._format_entries(entries)

    def _format_entries(self, entries: list[TranscriptEntry]) -> str:
        """Format a list of entries as readable transcript text.

        Format: "Speaker (HH:MM): text"

        Args:
            entries: List of TranscriptEntry objects.

        Returns:
            Formatted transcript string.
        """
        lines = []
        for e in entries:
            time_str = e.timestamp.strftime("%H:%M")
            lines.append(f"{e.speaker} ({time_str}): {e.text}")
        return "\n".join(lines)

    def get_plain_text(self, entries: list[TranscriptEntry] | None = None) -> str:
        """Get plain text of entries without timestamps.

        Args:
            entries: Specific entries to format. Defaults to all entries.

        Returns:
            Plain text string of speaker: text lines.
        """
        target = entries if entries is not None else self._entries
        return "\n".join(f"{e.speaker}: {e.text}" for e in target)

    def clear(self) -> None:
        """Clear all transcript entries."""
        self._entries.clear()
        self._next_segment_id = 1
        logger.debug("Transcript cleared")

    @property
    def entries(self) -> list[TranscriptEntry]:
        """All transcript entries (copy)."""
        return self._entries.copy()

    @property
    def entry_count(self) -> int:
        """Number of entries in the transcript."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
