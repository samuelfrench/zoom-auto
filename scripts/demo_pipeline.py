#!/usr/bin/env python3
"""Demo pipeline -- test the full conversation loop without Zoom.

Proves the entire system works end-to-end by wiring together STT, VAD,
TTS, LLM, context management, trigger detection, and response generation
using either a live microphone or a pre-recorded WAV file.

Usage:
    # Live mic mode (default) -- speak and hear the bot respond
    python scripts/demo_pipeline.py --user sam

    # File mode -- process a WAV recording
    python scripts/demo_pipeline.py --input meeting.wav --user sam

    # With persona
    python scripts/demo_pipeline.py --user sam --persona config/personas/sam.toml

    # With project knowledge
    python scripts/demo_pipeline.py --user sam --projects ~/myproject

    # Text-only mode (skip TTS)
    python scripts/demo_pipeline.py --user sam --no-tts

    # List audio devices
    python scripts/demo_pipeline.py --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

# Add src to path so we can import zoom_auto
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Terminal colors (ANSI)
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIC_SAMPLE_RATE = 16000  # VAD / STT expect 16 kHz mono
_MIC_CHANNELS = 1
_MIC_DTYPE = "int16"
_MIC_BLOCKSIZE = 512  # Silero VAD chunk size (512 samples @ 16 kHz = 32 ms)

logger = logging.getLogger("demo_pipeline")


# ============================================================================
# Utility helpers
# ============================================================================

def _ts() -> str:
    """Return a short timestamp string for terminal output."""
    return datetime.now().strftime("%H:%M:%S")


def _print_header() -> None:
    """Print the startup banner."""
    print(f"\n{_BOLD}{_CYAN}Demo Pipeline -- zoom-auto{_RESET}")
    print(f"{_DIM}{'=' * 48}{_RESET}")


def _print_model_status(name: str, detail: str, elapsed: float) -> None:
    """Print a model loading status line."""
    print(f"  {_GREEN}ok{_RESET} {name} ({detail}) -- {elapsed:.1f}s")


def _print_utterance(speaker: str, text: str) -> None:
    """Print a detected utterance."""
    print(f"\n{_DIM}[{_ts()}]{_RESET} {_BOLD}{_BLUE}{speaker}:{_RESET} {text}")


def _print_response(
    text: str,
    trigger: str,
    confidence: float,
    tts_time: float | None,
    tokens: int,
) -> None:
    """Print a bot response with metadata."""
    print(f"{_DIM}[{_ts()}]{_RESET} {_BOLD}{_MAGENTA}Bot:{_RESET} {text}")
    parts = [f"trigger: {trigger}", f"confidence: {confidence:.2f}"]
    if tts_time is not None:
        parts.append(f"TTS: {tts_time:.1f}s")
    parts.append(f"tokens: {tokens}")
    meta = ", ".join(parts)
    print(f"           {_DIM}({meta}){_RESET}")


def _print_no_response(reason: str, confidence: float) -> None:
    """Print when the bot decides not to respond."""
    print(
        f"           {_DIM}(no response -- {reason}, "
        f"confidence: {confidence:.2f}){_RESET}"
    )


# ============================================================================
# LLM provider auto-detection
# ============================================================================

async def _create_llm_provider(
    llm_choice: str,
    settings: object,
) -> object:
    """Create and validate an LLM provider.

    Tries Claude first (if API key set), falls back to Ollama.

    Returns:
        An LLMProvider instance.

    Raises:
        SystemExit: If no LLM provider is available.
    """
    from zoom_auto.config import Settings
    from zoom_auto.llm.claude import ClaudeProvider
    from zoom_auto.llm.ollama import OllamaProvider

    assert isinstance(settings, Settings)
    llm_config = settings.llm

    if llm_choice == "claude":
        api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print(f"{_RED}Error: ANTHROPIC_API_KEY not set and --llm=claude requested{_RESET}")
            sys.exit(1)
        provider = ClaudeProvider(config=llm_config, api_key=api_key)
        if await provider.is_available():
            print(f"  {_GREEN}ok{_RESET} LLM: Claude ({llm_config.response_model})")
            return provider
        print(f"{_RED}Error: Claude API not reachable{_RESET}")
        sys.exit(1)

    if llm_choice == "ollama":
        provider = OllamaProvider(config=llm_config, host=settings.ollama_host)
        if await provider.is_available():
            print(f"  {_GREEN}ok{_RESET} LLM: Ollama ({llm_config.ollama_model})")
            return provider
        print(f"{_RED}Error: Ollama not running or model not found{_RESET}")
        print(f"       Host: {settings.ollama_host}")
        print(f"       Model: {llm_config.ollama_model}")
        print(f"       Try: ollama pull {llm_config.ollama_model}")
        sys.exit(1)

    # Auto-detect: try Claude first, then Ollama
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        provider = ClaudeProvider(config=llm_config, api_key=api_key)
        try:
            if await provider.is_available():
                print(f"  {_GREEN}ok{_RESET} LLM: Claude ({llm_config.response_model})")
                return provider
        except Exception:
            pass

    provider = OllamaProvider(config=llm_config, host=settings.ollama_host)
    try:
        if await provider.is_available():
            print(f"  {_GREEN}ok{_RESET} LLM: Ollama ({llm_config.ollama_model})")
            return provider
    except Exception:
        pass

    print(f"\n{_RED}Error: No LLM provider available.{_RESET}")
    print("  Set ANTHROPIC_API_KEY for Claude, or start Ollama:")
    print(f"    ollama serve && ollama pull {llm_config.ollama_model}")
    sys.exit(1)


# ============================================================================
# Core demo pipeline
# ============================================================================

class DemoPipeline:
    """Wires together all pipeline components for demo/testing.

    Attributes:
        settings: Application settings.
        stt: Speech-to-text engine.
        tts: Text-to-speech engine (None if --no-tts).
        vad: Voice activity detector.
        llm: LLM provider.
        context_manager: Meeting context manager.
        trigger_detector: Response trigger detector.
        response_generator: Response generator.
        turn_manager: Turn-taking manager.
        voice_sample: Path to voice reference WAV.
        no_tts: Whether TTS is disabled.
        no_vad: Whether VAD is disabled.
    """

    def __init__(
        self,
        settings: object,
        llm: object,
        voice_sample: Path | None,
        persona: object | None = None,
        knowledge_store: object | None = None,
        no_tts: bool = False,
        no_vad: bool = False,
        user: str = "default",
    ) -> None:
        from zoom_auto.config import Settings
        from zoom_auto.context.manager import ContextManager
        from zoom_auto.llm.base import LLMProvider
        from zoom_auto.persona.builder import PersonaProfile
        from zoom_auto.persona.knowledge_store import KnowledgeStore
        from zoom_auto.persona.learner import ConversationLearner
        from zoom_auto.pipeline.vad import VADProcessor
        from zoom_auto.response.decision import TriggerDetector
        from zoom_auto.response.generator import ResponseGenerator
        from zoom_auto.response.turn_manager import TurnManager
        from zoom_auto.stt.faster_whisper import FasterWhisperEngine
        from zoom_auto.tts.chatterbox import ChatterboxEngine

        assert isinstance(settings, Settings)
        assert isinstance(llm, LLMProvider)
        self.settings = settings
        self.no_tts = no_tts
        self.no_vad = no_vad
        self.voice_sample = voice_sample
        self.user = user

        # Conversation learner
        self.learner = ConversationLearner(
            data_dir=Path("data/learnings"), user=user,
        )

        # STT
        self.stt = FasterWhisperEngine(
            config=settings.stt,
            device="cuda" if _gpu_available() else "cpu",
        )

        # TTS (optional)
        self.tts: ChatterboxEngine | None = None
        if not no_tts:
            self.tts = ChatterboxEngine(
                config=settings.tts,
                device="cuda" if _gpu_available() else "cpu",
            )

        # VAD
        self.vad = VADProcessor(config=settings.vad)

        # LLM
        self.llm = llm

        # Context
        self.context_manager = ContextManager(
            config=settings.context,
            llm=llm,
        )

        # Turn manager
        self.turn_manager = TurnManager(config=settings.response)

        # Trigger detector
        self.trigger_detector = TriggerDetector(
            config=settings.response,
            llm=llm,
        )

        # Response generator
        self.response_generator = ResponseGenerator(
            llm=llm,
            context_manager=self.context_manager,
            persona=persona if isinstance(persona, PersonaProfile) else None,
            knowledge_store=(
                knowledge_store
                if isinstance(knowledge_store, KnowledgeStore)
                else None
            ),
            learner=self.learner,
        )

    async def load_models(self) -> None:
        """Load all ML models with timing."""
        print(f"\n{_BOLD}Loading models...{_RESET}")

        # STT
        t0 = time.monotonic()
        await self.stt.load_model()
        _print_model_status(
            "STT",
            f"faster-whisper {self.settings.stt.model}, {self.settings.stt.compute_type}",
            time.monotonic() - t0,
        )

        # TTS
        if self.tts is not None:
            t0 = time.monotonic()
            await self.tts.load_model()
            _print_model_status("TTS", "Chatterbox Turbo", time.monotonic() - t0)

        # VAD
        if not self.no_vad:
            t0 = time.monotonic()
            await self.vad.load_model()
            _print_model_status("VAD", "Silero", time.monotonic() - t0)

        # Voice reference
        if self.voice_sample and self.voice_sample.exists():
            print(
                f"  {_GREEN}ok{_RESET} Voice reference: {self.voice_sample}"
            )
        elif self.voice_sample:
            print(
                f"  {_YELLOW}warn{_RESET} Voice reference not found: "
                f"{self.voice_sample} (using default voice)"
            )
            self.voice_sample = None

    async def unload_models(self) -> None:
        """Unload all ML models."""
        print(f"\n{_DIM}Unloading models...{_RESET}")
        await self.stt.unload_model()
        if self.tts is not None:
            await self.tts.unload_model()
        if not self.no_vad:
            await self.vad.unload_model()

    async def process_utterance(self, speaker: str, text: str) -> None:
        """Process a single transcribed utterance through the full pipeline.

        1. Add to context
        2. Check trigger
        3. Generate response if triggered
        4. Synthesize and play via TTS

        Args:
            speaker: Speaker name.
            text: Transcribed text.
        """
        if not text.strip():
            return

        bot_name = self.settings.zoom.bot_name

        # Record utterance for learning
        self.learner.record_utterance(speaker, text)

        # 1. Add to context
        await self.context_manager.add_transcript(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
        )
        self.turn_manager.record_other_speaker()

        # 2. Check trigger
        context = await self.context_manager.get_context()
        recent_text = "\n".join(context.recent_transcript)

        decision = await self.trigger_detector.should_respond(
            transcript=recent_text,
            bot_name=bot_name,
            is_cooldown_active=self.turn_manager.is_cooldown_active,
            someone_speaking=False,  # In demo, we process after speech ends
        )

        if not decision.should_respond:
            _print_no_response(decision.reason.value, decision.confidence)
            return

        if not self.turn_manager.can_speak():
            _print_no_response("turn_manager_blocked", 0.0)
            return

        # 3. Generate response
        logger.info(
            "Generating response (trigger=%s, confidence=%.2f)",
            decision.reason,
            decision.confidence,
        )

        t_llm_start = time.monotonic()
        response = await self.response_generator.generate(
            trigger_context=decision.context_snippet,
        )
        llm_elapsed = time.monotonic() - t_llm_start
        logger.debug("LLM generation took %.2fs", llm_elapsed)

        if not response.text.strip():
            _print_no_response("empty_response", decision.confidence)
            return

        # 4. TTS (optional)
        tts_time: float | None = None
        if self.tts is not None and not self.no_tts:
            t_tts_start = time.monotonic()
            tts_result = await self.tts.synthesize(
                text=response.text,
                voice_sample=self.voice_sample,
            )
            tts_time = time.monotonic() - t_tts_start

            # Play audio through speakers
            try:
                await _play_audio(
                    tts_result.audio_data,
                    tts_result.sample_rate,
                )
            except Exception as exc:
                logger.warning("Failed to play audio: %s", exc)

        _print_response(
            text=response.text,
            trigger=decision.reason.value,
            confidence=decision.confidence,
            tts_time=tts_time,
            tokens=response.token_usage,
        )

        # Record bot response for learning
        self.learner.record_bot_response(
            trigger_reason=decision.reason.value,
            response_text=response.text,
            context_snippet=decision.context_snippet,
        )

        # 5. Record in context and turn manager
        await self.context_manager.add_transcript(
            speaker=bot_name,
            text=response.text,
            timestamp=datetime.now(),
        )
        self.turn_manager.record_response()


# ============================================================================
# Audio I/O helpers
# ============================================================================

def _gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


async def _play_audio(pcm_data: bytes, sample_rate: int) -> None:
    """Play PCM 16-bit audio through the default output device.

    Args:
        pcm_data: Raw 16-bit PCM bytes (mono).
        sample_rate: Sample rate of the audio.
    """
    import numpy as np
    import sounddevice as sd

    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def _play() -> None:
        try:
            sd.play(audio, samplerate=sample_rate, blocking=True)
        finally:
            loop.call_soon_threadsafe(done.set)

    loop.run_in_executor(None, _play)
    await done.wait()


def _list_audio_devices() -> None:
    """List available audio devices and exit."""
    try:
        import sounddevice as sd
    except ImportError:
        print(f"{_RED}Error: sounddevice not installed.{_RESET}")
        print("  Install: pip install sounddevice")
        sys.exit(1)

    print(f"\n{_BOLD}Available audio devices:{_RESET}\n")
    print(sd.query_devices())
    print(f"\n{_DIM}Use --device N to select a specific input device.{_RESET}")
    sys.exit(0)


# ============================================================================
# Live mic mode
# ============================================================================

async def _run_live_mic(
    pipeline: DemoPipeline,
    device: int | None,
    user: str,
) -> None:
    """Run the pipeline with live microphone input.

    Captures mic audio via sounddevice, feeds through VAD in real-time,
    and on speech_end transcribes and processes the utterance.

    Args:
        pipeline: The initialized DemoPipeline.
        device: Audio device index (None for default).
        user: Username for display.
    """
    import numpy as np
    import sounddevice as sd

    print(f"\n{_BOLD}Listening on microphone...{_RESET} (Ctrl+C to stop)")
    print(f"{_DIM}{'=' * 48}{_RESET}\n")

    # Shared state for the audio callback
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Sounddevice input callback -- runs in audio thread."""
        if status:
            logger.warning("Audio input status: %s", status)
        # Convert to bytes (already int16 from dtype)
        pcm_bytes = indata.tobytes()
        loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_bytes)

    # Open input stream
    stream = sd.InputStream(
        samplerate=_MIC_SAMPLE_RATE,
        blocksize=_MIC_BLOCKSIZE,
        channels=_MIC_CHANNELS,
        dtype=_MIC_DTYPE,
        device=device,
        callback=_audio_callback,
    )

    stop_event = asyncio.Event()

    # Handle Ctrl+C gracefully
    def _signal_handler(sig: int, frame: object) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        stream.start()
        logger.info("Microphone stream started (device=%s)", device or "default")

        while not stop_event.is_set():
            try:
                pcm_chunk = await asyncio.wait_for(
                    audio_queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue

            if pipeline.no_vad:
                # Without VAD, accumulate chunks and process periodically
                # This mode is less useful but allows testing STT directly
                continue

            # Feed through VAD
            event = await pipeline.vad.process_chunk(pcm_chunk)

            if event is not None:
                if event.is_speech_start:
                    print(f"{_DIM}[{_ts()}] (speech detected...){_RESET}", end="", flush=True)

                elif event.is_speech_end and event.audio_buffer:
                    audio_data = event.audio_buffer

                    # Transcribe
                    t_stt_start = time.monotonic()
                    try:
                        result = await pipeline.stt.transcribe(
                            audio_data=audio_data,
                            sample_rate=_MIC_SAMPLE_RATE,
                        )
                    except Exception as exc:
                        logger.warning("STT failed: %s", exc)
                        continue

                    t_stt = time.monotonic() - t_stt_start

                    text = result.text.strip()
                    if not text:
                        print(f"\r{_DIM}[{_ts()}] (no speech detected){_RESET}")
                        continue

                    # Clear the "speech detected" line
                    print("\r", end="")

                    _print_utterance(user, text)
                    print(
                        f"           {_DIM}(STT: {t_stt:.1f}s, "
                        f"confidence: {result.confidence:.2f}){_RESET}"
                    )

                    # Process through the full pipeline
                    await pipeline.process_utterance(speaker=user, text=text)

    finally:
        stream.stop()
        stream.close()
        print(f"\n\n{_DIM}Microphone stream closed.{_RESET}")


# ============================================================================
# File mode
# ============================================================================

async def _run_file_mode(
    pipeline: DemoPipeline,
    input_path: Path,
    output_dir: Path | None,
    user: str,
) -> None:
    """Process a WAV file through the pipeline.

    Reads the WAV file, feeds it through VAD in chunks (simulating
    real-time), transcribes detected speech segments, and processes
    each utterance through trigger detection and response generation.

    Args:
        pipeline: The initialized DemoPipeline.
        input_path: Path to the input WAV file.
        output_dir: Directory for output WAV files (bot responses).
        user: Username for display.
    """
    import numpy as np

    if not input_path.exists():
        print(f"{_RED}Error: Input file not found: {input_path}{_RESET}")
        sys.exit(1)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{_BOLD}Processing: {input_path}{_RESET}")

    # Read WAV file
    try:
        with wave.open(str(input_path), "rb") as wf:
            n_channels = wf.getnchannels()
            _ = wf.getsampwidth()  # not needed but validates format
            file_sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)
    except Exception as exc:
        print(f"{_RED}Error reading WAV file: {exc}{_RESET}")
        sys.exit(1)

    duration = n_frames / file_sample_rate
    print(
        f"  Duration: {duration:.1f}s, "
        f"Sample rate: {file_sample_rate}Hz, "
        f"Channels: {n_channels}"
    )

    # Convert to mono 16-bit 16kHz if needed
    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)

    if n_channels > 1:
        # Take first channel
        audio_int16 = audio_int16[::n_channels]

    if file_sample_rate != _MIC_SAMPLE_RATE:
        # Simple resampling via librosa
        try:
            import librosa

            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_resampled = librosa.resample(
                audio_float,
                orig_sr=file_sample_rate,
                target_sr=_MIC_SAMPLE_RATE,
            )
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            print(f"  Resampled: {file_sample_rate}Hz -> {_MIC_SAMPLE_RATE}Hz")
        except ImportError:
            print(f"{_YELLOW}Warning: librosa not available for resampling{_RESET}")
            print(
                f"  File sample rate ({file_sample_rate}) "
                f"differs from expected ({_MIC_SAMPLE_RATE})"
            )

    pcm_bytes = audio_int16.tobytes()

    print(f"\n{_BOLD}Processing audio...{_RESET}")
    print(f"{_DIM}{'=' * 48}{_RESET}\n")

    response_count = 0

    if pipeline.no_vad:
        # Without VAD, process the entire file as one utterance
        t_stt_start = time.monotonic()
        result = await pipeline.stt.transcribe(
            audio_data=pcm_bytes,
            sample_rate=_MIC_SAMPLE_RATE,
        )
        t_stt = time.monotonic() - t_stt_start

        if result.text.strip():
            _print_utterance(user, result.text)
            print(
                f"           {_DIM}(STT: {t_stt:.1f}s, "
                f"confidence: {result.confidence:.2f}){_RESET}"
            )
            await pipeline.process_utterance(speaker=user, text=result.text)
    else:
        # Feed through VAD in chunks
        chunk_size = _MIC_BLOCKSIZE * 2  # 512 samples * 2 bytes/sample
        offset = 0

        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset : offset + chunk_size]
            offset += chunk_size

            if len(chunk) < chunk_size:
                # Pad the last chunk with silence
                chunk = chunk + b"\x00" * (chunk_size - len(chunk))

            event = await pipeline.vad.process_chunk(chunk)

            if event is not None and event.is_speech_end and event.audio_buffer:
                audio_data = event.audio_buffer

                # Transcribe
                t_stt_start = time.monotonic()
                try:
                    result = await pipeline.stt.transcribe(
                        audio_data=audio_data,
                        sample_rate=_MIC_SAMPLE_RATE,
                    )
                except Exception as exc:
                    logger.warning("STT failed: %s", exc)
                    continue

                t_stt = time.monotonic() - t_stt_start

                text = result.text.strip()
                if not text:
                    continue

                _print_utterance(user, text)
                print(
                    f"           {_DIM}(STT: {t_stt:.1f}s, "
                    f"confidence: {result.confidence:.2f}){_RESET}"
                )

                await pipeline.process_utterance(speaker=user, text=text)
                response_count += 1

        # Flush any remaining buffered speech in VAD
        remaining_audio = await pipeline.vad.get_speech_segment()
        if remaining_audio:
            t_stt_start = time.monotonic()
            try:
                result = await pipeline.stt.transcribe(
                    audio_data=remaining_audio,
                    sample_rate=_MIC_SAMPLE_RATE,
                )
            except Exception:
                result = None

            if result and result.text.strip():
                t_stt = time.monotonic() - t_stt_start
                _print_utterance(user, result.text)
                print(
                    f"           {_DIM}(STT: {t_stt:.1f}s, "
                    f"confidence: {result.confidence:.2f}){_RESET}"
                )
                await pipeline.process_utterance(
                    speaker=user, text=result.text
                )

    print(f"\n{_DIM}{'=' * 48}{_RESET}")
    print(f"{_BOLD}File processing complete.{_RESET}")


# ============================================================================
# Voice sample discovery
# ============================================================================

def _find_voice_sample(user: str, voice_dir: str) -> Path | None:
    """Find a voice reference WAV for the given user.

    Looks for combined_reference.wav first, then any WAV in the
    user's voice sample directory.

    Args:
        user: Username to look up.
        voice_dir: Base voice samples directory.

    Returns:
        Path to the voice reference, or None if not found.
    """
    base = Path(voice_dir) / user

    # Preferred: combined reference
    combined = base / "combined_reference.wav"
    if combined.exists():
        return combined

    # Fall back to any WAV file
    if base.is_dir():
        wavs = sorted(base.glob("*.wav"))
        if wavs:
            return wavs[0]

    return None


# ============================================================================
# Persona and knowledge loading
# ============================================================================

def _load_persona(persona_path: str | None) -> object | None:
    """Load a persona profile from a TOML file.

    Args:
        persona_path: Path to the persona TOML file.

    Returns:
        PersonaProfile or None.
    """
    if not persona_path:
        return None

    path = Path(persona_path)
    if not path.exists():
        print(f"{_YELLOW}Warning: Persona file not found: {path}{_RESET}")
        return None

    from zoom_auto.persona.builder import PersonaProfile

    try:
        profile = PersonaProfile.from_toml(path)
        print(f"  {_GREEN}ok{_RESET} Persona: {profile.name or path.stem}")
        return profile
    except Exception as exc:
        print(f"{_YELLOW}Warning: Failed to load persona: {exc}{_RESET}")
        return None


def _index_projects(project_dirs: list[str] | None) -> object | None:
    """Index project directories and create a KnowledgeStore.

    Args:
        project_dirs: List of project directory paths to index.

    Returns:
        KnowledgeStore or None.
    """
    if not project_dirs:
        return None

    from zoom_auto.persona.knowledge_store import KnowledgeStore
    from zoom_auto.persona.sources.project import ProjectIndexer

    store = KnowledgeStore()
    indexer = ProjectIndexer()

    for dir_path in project_dirs:
        path = Path(dir_path).expanduser().resolve()
        if not path.is_dir():
            print(f"{_YELLOW}Warning: Not a directory: {path}{_RESET}")
            continue

        try:
            index = indexer.index(path)
            store.save_index(index)
            print(
                f"  {_GREEN}ok{_RESET} Project: {index.name} "
                f"({index.total_files} files, "
                f"tech: {', '.join(index.tech_stack[:3])})"
            )
        except Exception as exc:
            print(f"{_YELLOW}Warning: Failed to index {path}: {exc}{_RESET}")

    return store


# ============================================================================
# Argument parsing
# ============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo pipeline -- test the full zoom-auto conversation loop without Zoom.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --user sam                          # Live mic\n"
            "  %(prog)s --input meeting.wav --user sam      # Process file\n"
            "  %(prog)s --user sam --persona sam.toml       # With persona\n"
            "  %(prog)s --user sam --no-tts                 # Text only\n"
            "  %(prog)s --list-devices                      # Show audio devices\n"
        ),
    )

    parser.add_argument(
        "--user",
        default="default",
        help="Username for voice samples (default: 'default')",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        default=None,
        help="WAV file to process (omit for live mic mode)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output WAV files (file mode)",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="Persona TOML file to use",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        help="Project directories to index for context",
    )
    parser.add_argument(
        "--llm",
        choices=["claude", "ollama", "auto"],
        default="auto",
        help="LLM provider (default: auto-detect)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index for microphone input",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip TTS (text output only)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Skip VAD (process all audio as speech)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


# ============================================================================
# Main entry point
# ============================================================================

async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point for the demo pipeline."""
    from zoom_auto.config import Settings

    _print_header()

    # Load settings
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    settings = Settings.from_toml(config_dir)

    # GPU check
    if _gpu_available():
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  {_GREEN}ok{_RESET} GPU: {gpu_name}")
    else:
        print(f"  {_YELLOW}warn{_RESET} No CUDA GPU detected -- using CPU (slower)")

    # LLM provider
    llm = await _create_llm_provider(args.llm, settings)

    # Persona
    persona = _load_persona(args.persona)

    # Project knowledge
    knowledge_store = _index_projects(args.projects)

    # Voice sample
    voice_sample = _find_voice_sample(args.user, settings.tts.voice_sample_dir)

    # Create pipeline
    pipeline = DemoPipeline(
        settings=settings,
        llm=llm,
        voice_sample=voice_sample,
        persona=persona,
        knowledge_store=knowledge_store,
        no_tts=args.no_tts,
        no_vad=args.no_vad,
        user=args.user,
    )

    # Load models
    await pipeline.load_models()

    # Start learning session
    pipeline.learner.start_session()

    try:
        if args.input_file:
            # File mode
            await _run_file_mode(
                pipeline=pipeline,
                input_path=Path(args.input_file),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                user=args.user,
            )
        else:
            # Live mic mode
            await _run_live_mic(
                pipeline=pipeline,
                device=args.device,
                user=args.user,
            )
    finally:
        # End learning session and show summary
        session = pipeline.learner.end_session()
        print(f"\n{_CYAN}Session learnings saved:{_RESET}")
        print(f"   Transcript: {len(session.transcript)} utterances")
        print(
            f"   Topics: "
            f"{', '.join(session.topics_discussed) or 'none detected'}"
        )
        print(f"   Meeting type: {session.meeting_type}")
        print(f"   New vocabulary: {len(session.vocabulary_learned)} words")
        print(
            f"   Saved to: data/learnings/{args.user}/sessions/"
        )
        await pipeline.unload_models()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if args.list_devices:
        _list_audio_devices()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print(f"\n{_DIM}Interrupted.{_RESET}")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\n{_RED}Fatal error: {exc}{_RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
