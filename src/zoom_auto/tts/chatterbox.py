"""Chatterbox TTS engine with streaming voice cloning.

Uses Chatterbox Turbo for low-latency, high-quality speech synthesis
with voice cloning from reference samples.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from zoom_auto.config import TTSConfig
from zoom_auto.tts.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)

# Default sample rate for Chatterbox (overridden by model.sr at runtime)
_DEFAULT_SAMPLE_RATE = 24000

# Streaming chunk size in samples (50ms at 24kHz = 1200 samples)
_STREAM_CHUNK_SAMPLES = 1200


class ChatterboxEngine(TTSEngine):
    """Text-to-speech using Chatterbox Turbo with voice cloning.

    Supports both batch synthesis and streaming output for
    low-latency real-time conversation.

    Args:
        config: TTS configuration.
        device: Torch device for model inference (default "cuda").
        exaggeration: Emotion exaggeration level 0.0-1.0 (default 0.5).
    """

    def __init__(
        self,
        config: TTSConfig | None = None,
        device: str = "cuda",
        exaggeration: float = 0.5,
    ) -> None:
        self.config = config or TTSConfig()
        self._device = device
        self._exaggeration = max(0.0, min(1.0, exaggeration))
        self._model: object | None = None
        self._sample_rate: int = _DEFAULT_SAMPLE_RATE

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self._sample_rate

    @property
    def exaggeration(self) -> float:
        """Emotion exaggeration level (0.0-1.0)."""
        return self._exaggeration

    @exaggeration.setter
    def exaggeration(self, value: float) -> None:
        """Set emotion exaggeration level, clamped to 0.0-1.0."""
        self._exaggeration = max(0.0, min(1.0, value))

    async def load_model(self) -> None:
        """Load Chatterbox Turbo model into GPU memory."""
        if self._model is not None:
            logger.info("Chatterbox model already loaded")
            return

        logger.info("Loading Chatterbox model on device=%s ...", self._device)

        # Import lazily so the module can be imported without torch/chatterbox installed
        import torch
        from chatterbox.tts import ChatterboxTTS as ChatterboxModel

        # Run model loading in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def _load() -> ChatterboxModel:
            with torch.inference_mode():
                return ChatterboxModel.from_pretrained(device=self._device)

        model = await loop.run_in_executor(None, _load)
        self._model = model
        self._sample_rate = int(model.sr)

        logger.info(
            "Chatterbox model loaded (device=%s, sample_rate=%d)",
            self._device,
            self._sample_rate,
        )

    async def unload_model(self) -> None:
        """Unload the model from memory and free GPU resources."""
        if self._model is None:
            return

        logger.info("Unloading Chatterbox model ...")

        import torch

        del self._model
        self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Chatterbox model unloaded")

    def is_loaded(self) -> bool:
        """Whether the Chatterbox model is loaded."""
        return self._model is not None

    async def synthesize(self, text: str, voice_sample: Path | None = None) -> TTSResult:
        """Synthesize speech with optional voice cloning.

        Args:
            text: Text to convert to speech.
            voice_sample: Path to a reference voice WAV file for cloning.

        Returns:
            TTSResult with PCM 16-bit audio data.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Chatterbox model not loaded. Call load_model() first.")

        import torch

        model = self._model
        loop = asyncio.get_running_loop()

        # Load voice reference if provided
        audio_prompt = None
        if voice_sample is not None:
            audio_prompt = self._load_voice_reference(voice_sample)

        def _generate() -> torch.Tensor:
            with torch.inference_mode():
                kwargs: dict = {"text": text}
                if audio_prompt is not None:
                    kwargs["audio_prompt"] = audio_prompt
                kwargs["exaggeration"] = self._exaggeration
                return model.generate(**kwargs)

        wav_tensor = await loop.run_in_executor(None, _generate)

        # Convert torch tensor to PCM 16-bit bytes
        pcm_bytes = self._tensor_to_pcm_bytes(wav_tensor)

        # Calculate duration
        num_samples = len(pcm_bytes) // 2  # 16-bit = 2 bytes per sample
        duration = num_samples / self._sample_rate

        return TTSResult(
            audio_data=pcm_bytes,
            sample_rate=self._sample_rate,
            duration_seconds=duration,
        )

    async def synthesize_stream(
        self, text: str, voice_sample: Path | None = None
    ) -> AsyncIterator[bytes]:
        """Stream synthesized speech as PCM chunks.

        Currently generates the full audio and then yields it in chunks.
        True streaming will be implemented when Chatterbox supports it.

        Args:
            text: Text to convert to speech.
            voice_sample: Path to a reference voice WAV file for cloning.

        Yields:
            PCM 16-bit audio chunks for streaming playback.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        result = await self.synthesize(text, voice_sample)
        data = result.audio_data

        # Yield in chunks (each sample is 2 bytes for 16-bit PCM)
        chunk_bytes = _STREAM_CHUNK_SAMPLES * 2
        offset = 0
        while offset < len(data):
            end = min(offset + chunk_bytes, len(data))
            yield data[offset:end]
            offset = end
            # Yield control to the event loop between chunks
            await asyncio.sleep(0)

    def _load_voice_reference(self, path: Path) -> object:
        """Load a voice reference WAV file as a tensor for the model.

        Uses soundfile for reading audio (avoids torchaudio/torchcodec dependency
        issues) and converts to a torch tensor with optional resampling.

        Args:
            path: Path to the voice reference WAV file.

        Returns:
            Loaded audio tensor suitable for Chatterbox audio_prompt.

        Raises:
            FileNotFoundError: If the reference file does not exist.
            RuntimeError: If the file cannot be loaded.
        """
        import soundfile as sf
        import torch
        import torchaudio

        if not path.exists():
            raise FileNotFoundError(f"Voice reference file not found: {path}")

        try:
            audio_np, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            raise RuntimeError(f"Failed to load voice reference {path}: {e}") from e

        # Ensure mono
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]

        # Convert to torch tensor [1, N] (channels_first)
        wav = torch.from_numpy(audio_np).unsqueeze(0)

        # Resample if needed (model expects its own sample rate)
        if sr != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self._sample_rate)
            wav = resampler(wav)

        # Move to model device
        wav = wav.to(self._device)

        return wav

    @staticmethod
    def _tensor_to_pcm_bytes(wav_tensor: object) -> bytes:
        """Convert a torch audio tensor to PCM 16-bit bytes.

        Args:
            wav_tensor: Audio tensor from Chatterbox (shape [1, N] or [N]).

        Returns:
            Raw PCM 16-bit little-endian bytes.
        """
        import torch

        tensor = wav_tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        # Move to CPU and convert to numpy
        audio_np = tensor.squeeze().cpu().float().numpy()

        # Clamp to [-1.0, 1.0] to prevent overflow
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Convert to 16-bit signed integers
        audio_int16 = (audio_np * 32767).astype(np.int16)

        return audio_int16.tobytes()

    def save_wav(self, pcm_bytes: bytes, output_path: Path) -> None:
        """Save PCM 16-bit bytes as a WAV file.

        Convenience method for saving generated audio.

        Args:
            pcm_bytes: Raw PCM 16-bit little-endian audio bytes.
            output_path: Path to write the WAV file.
        """
        import soundfile as sf

        # Convert bytes back to numpy int16 array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Convert to float for soundfile
        audio_float = audio_int16.astype(np.float64) / 32768.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_float, self._sample_rate, subtype="PCM_16")
