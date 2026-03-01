"""Audio sender for Zoom meetings.

Sends TTS-generated PCM audio back into the Zoom meeting
so other participants can hear the bot's responses.
"""

from __future__ import annotations

import logging

from zoom_auto.config import ZoomConfig

logger = logging.getLogger(__name__)


class AudioSender:
    """Sends PCM audio data to a Zoom meeting.

    Accepts audio frames (from TTS) and feeds them into the Zoom
    Meeting SDK's audio output channel.
    """

    def __init__(self, config: ZoomConfig) -> None:
        self.config = config
        self._active = False

    async def start(self) -> None:
        """Initialize the audio sender."""
        raise NotImplementedError("Audio sender not yet implemented")

    async def stop(self) -> None:
        """Stop sending audio."""
        self._active = False

    async def send_frame(self, pcm_data: bytes, sample_rate: int = 16000) -> None:
        """Send a single PCM audio frame to the meeting.

        Args:
            pcm_data: Raw 16-bit PCM audio bytes.
            sample_rate: Sample rate of the audio data.
        """
        raise NotImplementedError("Audio frame sending not yet implemented")

    async def send_audio(self, pcm_data: bytes, sample_rate: int = 16000) -> None:
        """Send a complete audio buffer to the meeting, chunked into frames.

        Args:
            pcm_data: Raw 16-bit PCM audio bytes (complete utterance).
            sample_rate: Sample rate of the audio data.
        """
        raise NotImplementedError("Complete audio sending not yet implemented")

    @property
    def is_active(self) -> bool:
        """Whether the audio sender is active."""
        return self._active
