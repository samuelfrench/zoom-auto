"""Audio capture from Zoom meetings.

Receives per-speaker PCM audio frames from the Zoom Meeting SDK
and routes them to the audio pipeline for processing.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from zoom_auto.config import ZoomConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    """A single audio frame from a meeting participant.

    Attributes:
        speaker_id: Unique identifier for the speaker.
        speaker_name: Display name of the speaker.
        pcm_data: Raw 16-bit PCM audio bytes.
        sample_rate: Sample rate in Hz.
        timestamp_ms: Timestamp in milliseconds since meeting start.
    """

    speaker_id: int
    speaker_name: str
    pcm_data: bytes
    sample_rate: int
    timestamp_ms: int


class AudioCapture:
    """Captures per-speaker audio from Zoom Meeting SDK.

    Provides an async iterator interface for consuming audio frames
    as they arrive from the meeting.
    """

    def __init__(self, config: ZoomConfig) -> None:
        self.config = config
        self._active = False

    async def start(self) -> None:
        """Start capturing audio from the meeting."""
        raise NotImplementedError("Audio capture not yet implemented")

    async def stop(self) -> None:
        """Stop capturing audio."""
        self._active = False

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield audio frames as they arrive from participants.

        Yields:
            AudioFrame objects with per-speaker PCM data.
        """
        raise NotImplementedError("Audio frame iteration not yet implemented")
        yield  # pragma: no cover — makes this a generator

    @property
    def is_active(self) -> bool:
        """Whether audio capture is currently active."""
        return self._active
