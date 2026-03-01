#!/usr/bin/env python3
"""CLI tool for recording voice samples for TTS voice cloning.

Records voice samples from the microphone, one prompt at a time,
validates quality (SNR, clipping, volume), and saves to the voice store.

Usage:
    python scripts/record_voice_samples.py --user sam
    python scripts/record_voice_samples.py --user sam --list-devices
    python scripts/record_voice_samples.py --user sam --device 1
    python scripts/record_voice_samples.py --user sam --start-from 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

# Add src to path so we can import zoom_auto
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import sounddevice as sd  # noqa: E402

from zoom_auto.tts.audio_validation import TARGET_SAMPLE_RATE, validate_audio_data  # noqa: E402
from zoom_auto.tts.prompts import RECORDING_PROMPTS  # noqa: E402
from zoom_auto.tts.voice_store import VoiceStore  # noqa: E402

# Recording parameters
SAMPLE_RATE = TARGET_SAMPLE_RATE  # 22050 Hz
CHANNELS = 1
DTYPE = "float64"
PRE_ROLL_SECONDS = 0.3  # Short silence before recording starts
POST_ROLL_SECONDS = 0.3  # Short silence after recording stops


def print_header() -> None:
    """Print the tool header."""
    print()
    print("=" * 60)
    print("  Voice Sample Recorder for TTS Cloning")
    print("=" * 60)
    print()


def print_devices() -> None:
    """Print available audio input devices."""
    print("Available audio input devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:  # type: ignore[index]
            name = dev["name"]  # type: ignore[index]
            sr = dev["default_samplerate"]  # type: ignore[index]
            ch = dev["max_input_channels"]  # type: ignore[index]
            marker = " <-- default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {name} ({ch}ch, {sr:.0f}Hz){marker}")
    print()


def record_segment(duration_hint: float = 10.0, device: int | None = None) -> np.ndarray:
    """Record a single audio segment from the microphone.

    Uses a press-Enter-to-start, press-Enter-to-stop approach.

    Args:
        duration_hint: Maximum recording duration in seconds.
        device: Audio input device index (None = default).

    Returns:
        Recorded audio as a 1D numpy array (float64, mono).
    """
    max_samples = int(SAMPLE_RATE * duration_hint)
    buffer = np.zeros(max_samples, dtype=np.float64)
    recording_pos = [0]
    is_recording = [True]

    def callback(
        indata: np.ndarray,
        frames: int,
        time_info: dict,  # noqa: ARG001
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            pass  # Silently ignore status flags during recording
        if not is_recording[0]:
            return
        end = min(recording_pos[0] + frames, max_samples)
        n = end - recording_pos[0]
        if n > 0:
            buffer[recording_pos[0] : end] = indata[:n, 0]
            recording_pos[0] = end
        if recording_pos[0] >= max_samples:
            is_recording[0] = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
        device=device,
        blocksize=1024,
    )

    print("    Press ENTER to start recording...")
    input()

    print("    Recording... (press ENTER to stop)")
    stream.start()

    # Wait for Enter or max duration
    try:
        input()
    except EOFError:
        pass

    is_recording[0] = False
    stream.stop()
    stream.close()

    # Trim to actual recorded length
    actual_samples = recording_pos[0]
    if actual_samples == 0:
        print("    WARNING: No audio was recorded!")
        return np.array([], dtype=np.float64)

    recorded = buffer[:actual_samples].copy()

    # Trim pre/post roll silence
    pre_samples = int(SAMPLE_RATE * PRE_ROLL_SECONDS)
    post_samples = int(SAMPLE_RATE * POST_ROLL_SECONDS)
    if len(recorded) > pre_samples + post_samples:
        recorded = recorded[pre_samples : len(recorded) - post_samples]

    duration = len(recorded) / SAMPLE_RATE
    print(f"    Recorded {duration:.1f}s")

    return recorded


def print_quality_report(
    snr_db: float,
    peak: float,
    rms: float,
    has_clipping: bool,
    is_valid: bool,
    issues: list[str],
) -> None:
    """Print a formatted quality report for a recorded segment."""
    status = "PASS" if is_valid else "FAIL"
    print(f"    Quality: [{status}]  SNR: {snr_db:.1f}dB  Peak: {peak:.3f}  RMS: {rms:.4f}")
    if has_clipping:
        print("    WARNING: Clipping detected!")
    if issues:
        for issue in issues:
            print(f"    Issue: {issue}")


async def run_recording_session(
    user: str,
    device: int | None = None,
    start_from: int = 0,
    base_dir: Path | None = None,
) -> None:
    """Run an interactive recording session through all prompts.

    Args:
        user: Username for the voice profile.
        device: Audio input device index (None = default).
        start_from: Prompt index to start from (0-based).
        base_dir: Override base directory for voice samples.
    """
    store = VoiceStore(base_dir=base_dir)

    total_prompts = len(RECORDING_PROMPTS)
    prompts_to_record = list(enumerate(RECORDING_PROMPTS))[start_from:]

    print(f"Recording voice samples for user: {user}")
    print(f"Output directory: {store.base_dir / user / 'segments'}")
    print(f"Sample rate: {SAMPLE_RATE} Hz, Mono, 16-bit WAV")
    print(f"Prompts: {len(prompts_to_record)} of {total_prompts} remaining")
    print()
    print("Tips:")
    print("  - Speak naturally at your normal meeting volume")
    print("  - Keep consistent distance from the microphone")
    print("  - Record in a quiet room")
    print("  - Wait a beat after pressing Enter before speaking")
    print()

    recorded_count = 0
    skipped_count = 0

    for prompt_idx, prompt_text in prompts_to_record:
        print("-" * 60)
        print(f"  Prompt {prompt_idx + 1}/{total_prompts}:")
        print(f'  "{prompt_text}"')
        print()

        while True:
            audio = record_segment(duration_hint=15.0, device=device)

            if len(audio) == 0:
                print("    No audio recorded. Try again? [y/n/s(kip)] ", end="")
                choice = input().strip().lower()
                if choice == "s":
                    skipped_count += 1
                    break
                elif choice == "n":
                    print("\nSession ended early.")
                    return
                continue

            # Validate quality
            quality = validate_audio_data(audio, SAMPLE_RATE)
            print_quality_report(
                quality.snr_db,
                quality.peak_amplitude,
                quality.rms_amplitude,
                quality.has_clipping,
                quality.is_valid,
                quality.issues,
            )

            if not quality.is_valid:
                print("    Re-record? [y/n/s(kip)/k(eep anyway)] ", end="")
                choice = input().strip().lower()
                if choice == "s":
                    skipped_count += 1
                    break
                elif choice == "n":
                    print("\nSession ended early.")
                    return
                elif choice == "k":
                    pass  # Fall through to save
                else:
                    continue  # Re-record

            # Save the segment
            segment, _ = await store.add_sample_from_array(
                user=user,
                audio_data=audio,
                sample_rate=SAMPLE_RATE,
                prompt_index=prompt_idx,
                prompt_text=prompt_text,
            )
            print(f"    Saved: {segment.filename} (id: {segment.segment_id})")
            recorded_count += 1
            break

        print()

    # Summary
    print("=" * 60)
    print("  Recording Session Complete!")
    print(f"  Recorded: {recorded_count} segments")
    print(f"  Skipped: {skipped_count} segments")
    print("=" * 60)
    print()

    # Show status
    status = await store.get_status(user)
    print(f"Total segments: {status['total_segments']}")
    print(f"Valid segments: {status['valid_segments']}")
    print(f"Total valid duration: {status['total_valid_duration_seconds']:.1f}s")
    print(f"Average SNR: {status['average_snr_db']:.1f}dB")
    print(f"Recommendation: {status['recommendation']}")
    print()

    # Offer to combine
    if status["valid_segments"] >= 5:
        print("Would you like to combine samples into a reference file? [y/n] ", end="")
        choice = input().strip().lower()
        if choice == "y":
            try:
                path = await store.combine_reference(user)
                print(f"Combined reference saved to: {path}")
                meta = await store.get_metadata(user)
                print(f"Duration: {meta.combined_duration_seconds:.1f}s")
            except Exception as e:
                print(f"Error combining samples: {e}")


def main() -> None:
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Record voice samples for TTS voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --user sam\n"
            "  %(prog)s --user sam --list-devices\n"
            "  %(prog)s --user sam --device 1\n"
            "  %(prog)s --user sam --start-from 5\n"
        ),
    )
    parser.add_argument(
        "--user",
        required=True,
        help="Username for the voice profile",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Prompt index to start from (0-based, default: 0)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override voice sample base directory (default: data/voice_samples)",
    )

    args = parser.parse_args()

    print_header()

    if args.list_devices:
        print_devices()
        return

    base_dir = Path(args.data_dir) if args.data_dir else None

    # Verify audio device is available
    try:
        sd.query_devices(args.device or sd.default.device[0], "input")
    except Exception as e:
        print(f"Error: Could not access audio input device: {e}")
        print("Use --list-devices to see available devices.")
        sys.exit(1)

    asyncio.run(
        run_recording_session(
            user=args.user,
            device=args.device,
            start_from=args.start_from,
            base_dir=base_dir,
        )
    )


if __name__ == "__main__":
    main()
