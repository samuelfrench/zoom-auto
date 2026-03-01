#!/usr/bin/env python3
"""CLI test script for Chatterbox TTS voice cloning.

Loads the Chatterbox model, generates test phrases with optional
voice cloning from a reference WAV, saves output WAV files,
and reports timing benchmarks.

This script is meant to be run manually on a machine with a GPU
(e.g., RTX 4090). It is not a unit test.

Usage:
    python scripts/test_tts.py --voice-reference data/voice_samples/sam/combined_reference.wav
    python scripts/test_tts.py --output-dir output/tts_test
    python scripts/test_tts.py --voice-reference ref.wav --exaggeration 0.8
    python scripts/test_tts.py --device cpu  # (slow, for testing without GPU)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add src to path so we can import zoom_auto
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from zoom_auto.tts.chatterbox import ChatterboxEngine  # noqa: E402

# Test phrases covering different speech patterns
TEST_PHRASES = [
    "Hello everyone, thanks for joining the meeting today.",
    "I think we should focus on the API redesign for Q2.",
    "Could you share your screen so we can look at the data together?",
    "That's a great point. Let me pull up the latest numbers.",
    "The quick brown fox jumps over the lazy dog.",
]


def print_header() -> None:
    """Print the tool header."""
    print()
    print("=" * 60)
    print("  Chatterbox TTS Test Script")
    print("=" * 60)
    print()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


async def run_test(
    voice_reference: Path | None,
    output_dir: Path,
    device: str,
    exaggeration: float,
) -> None:
    """Run the TTS test with timing benchmarks.

    Args:
        voice_reference: Path to a voice reference WAV file.
        output_dir: Directory to save output WAV files.
        device: Torch device (e.g., "cuda", "cpu").
        exaggeration: Emotion exaggeration level 0.0-1.0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = ChatterboxEngine(device=device, exaggeration=exaggeration)

    # Load model
    print(f"Device: {device}")
    print(f"Exaggeration: {exaggeration}")
    if voice_reference:
        print(f"Voice reference: {voice_reference}")
    else:
        print("Voice reference: None (using default voice)")
    print(f"Output directory: {output_dir}")
    print()

    print("Loading model...")
    t_load_start = time.perf_counter()
    await engine.load_model()
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {format_duration(t_load)}")
    print(f"Model sample rate: {engine.sample_rate} Hz")
    print()

    # Generate test phrases
    print("-" * 60)
    print("Generating test phrases:")
    print("-" * 60)

    total_gen_time = 0.0
    total_audio_duration = 0.0

    for i, phrase in enumerate(TEST_PHRASES):
        truncated = f"{phrase[:60]}..." if len(phrase) > 60 else phrase
        print(f'\n[{i + 1}/{len(TEST_PHRASES)}] "{truncated}"')

        t_gen_start = time.perf_counter()
        result = await engine.synthesize(phrase, voice_sample=voice_reference)
        t_gen = time.perf_counter() - t_gen_start

        total_gen_time += t_gen
        total_audio_duration += result.duration_seconds

        # Save output WAV
        output_path = output_dir / f"test_{i + 1:02d}.wav"
        engine.save_wav(result.audio_data, output_path)

        rtf = t_gen / result.duration_seconds if result.duration_seconds > 0 else float("inf")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Generation time: {format_duration(t_gen)}")
        print(f"  Real-time factor: {rtf:.2f}x")
        print(f"  Saved: {output_path}")

    # Test streaming mode
    print()
    print("-" * 60)
    print("Testing streaming mode:")
    print("-" * 60)

    stream_phrase = TEST_PHRASES[0]
    print(f"\nPhrase: \"{stream_phrase[:60]}\"")

    t_stream_start = time.perf_counter()
    chunks = []
    first_chunk_time = None
    async for chunk in engine.synthesize_stream(stream_phrase, voice_sample=voice_reference):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t_stream_start
        chunks.append(chunk)
    t_stream = time.perf_counter() - t_stream_start

    total_stream_bytes = sum(len(c) for c in chunks)
    stream_samples = total_stream_bytes // 2
    stream_duration = stream_samples / engine.sample_rate

    print(f"  Chunks received: {len(chunks)}")
    print(f"  Time to first chunk: {format_duration(first_chunk_time or 0)}")
    print(f"  Total stream time: {format_duration(t_stream)}")
    print(f"  Audio duration: {stream_duration:.2f}s")

    # Save streamed audio
    stream_output = output_dir / "test_stream.wav"
    engine.save_wav(b"".join(chunks), stream_output)
    print(f"  Saved: {stream_output}")

    # Summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Model load time:       {format_duration(t_load)}")
    print(f"  Phrases generated:     {len(TEST_PHRASES)}")
    print(f"  Total generation time: {format_duration(total_gen_time)}")
    print(f"  Total audio duration:  {total_audio_duration:.2f}s")
    avg_rtf = total_gen_time / total_audio_duration if total_audio_duration > 0 else 0
    print(f"  Average RTF:           {avg_rtf:.2f}x")
    print(f"  Output directory:      {output_dir}")
    print()

    # Unload model
    await engine.unload_model()
    print("Model unloaded.")


def main() -> None:
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test Chatterbox TTS voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --voice-reference data/voice_samples/sam/combined_reference.wav\n"
            "  %(prog)s --output-dir output/tts_test --exaggeration 0.8\n"
            "  %(prog)s --device cpu\n"
        ),
    )
    parser.add_argument(
        "--voice-reference",
        type=Path,
        default=None,
        help="Path to a voice reference WAV file for cloning",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tts_test"),
        help="Directory to save output WAV files (default: output/tts_test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for inference (default: cuda)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration level 0.0-1.0 (default: 0.5)",
    )

    args = parser.parse_args()

    # Validate voice reference
    if args.voice_reference and not args.voice_reference.exists():
        print(f"Error: Voice reference file not found: {args.voice_reference}")
        sys.exit(1)

    print_header()
    asyncio.run(
        run_test(
            voice_reference=args.voice_reference,
            output_dir=args.output_dir,
            device=args.device,
            exaggeration=args.exaggeration,
        )
    )


if __name__ == "__main__":
    main()
