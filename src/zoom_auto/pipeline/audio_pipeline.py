"""Audio frame routing between components.

Routes audio frames from Zoom capture through VAD, STT, and back
through TTS to the Zoom audio sender.
"""

from __future__ import annotations

import logging

from zoom_auto.config import Settings
from zoom_auto.pipeline.vad import VADProcessor
from zoom_auto.stt.base import STTEngine
from zoom_auto.tts.base import TTSEngine
from zoom_auto.zoom.audio_capture import AudioCapture
from zoom_auto.zoom.audio_sender import AudioSender

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Routes audio frames between Zoom SDK and processing components.

    Flow: Zoom Capture -> VAD -> STT -> [LLM] -> TTS -> Zoom Send

    The pipeline handles buffering, format conversion, and async
    coordination between components.
    """

    def __init__(
        self,
        settings: Settings,
        capture: AudioCapture,
        sender: AudioSender,
        vad: VADProcessor,
        stt: STTEngine,
        tts: TTSEngine,
    ) -> None:
        self.settings = settings
        self.capture = capture
        self.sender = sender
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self._running = False

    async def start(self) -> None:
        """Start the audio pipeline processing loop."""
        raise NotImplementedError("Audio pipeline not yet implemented")

    async def stop(self) -> None:
        """Stop the audio pipeline."""
        self._running = False

    async def send_response(self, text: str) -> None:
        """Synthesize text to speech and send to the meeting.

        Args:
            text: The response text to speak.
        """
        raise NotImplementedError("Response sending not yet implemented")

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return self._running
