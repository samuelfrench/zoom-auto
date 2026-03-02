#!/usr/bin/env python3
"""Import voice recordings from verbal-direction into the VoiceStore.

Scans all WAV files from the verbal-direction recordings directory,
imports each via VoiceStore.add_sample() (which handles resampling
and quality validation), then combines valid samples into a single
reference WAV for TTS voice cloning.

Usage:
    python scripts/import_voice_samples.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import zoom_auto
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from zoom_auto.tts.voice_store import VoiceStore  # noqa: E402

RECORDINGS_DIR = Path.home() / ".local/share/verbal-direction/recordings"
USER = "sam"


async def main() -> None:
    """Import all voice recordings and create combined reference."""
    if not RECORDINGS_DIR.exists():
        print(f"Error: Recordings directory not found: {RECORDINGS_DIR}")
        sys.exit(1)

    # Collect all WAV files across session folders
    wav_files = sorted(RECORDINGS_DIR.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found.")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV files in {RECORDINGS_DIR}")
    print()

    store = VoiceStore(base_dir=_project_root / "data" / "voice_samples")

    passed = 0
    failed = 0
    fail_reasons: dict[str, int] = {}

    for i, wav_path in enumerate(wav_files, 1):
        audio_bytes = wav_path.read_bytes()
        segment, quality = await store.add_sample(
            user=USER,
            audio_data=audio_bytes,
            prompt_text=wav_path.stem,
        )

        if quality.is_valid:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            for issue in quality.issues:
                reason = issue.split(":")[0]
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        if i % 25 == 0 or i == len(wav_files):
            print(f"  [{i}/{len(wav_files)}] {status} — {wav_path.name} "
                  f"(SNR={quality.snr_db:.1f}dB, dur={quality.duration_seconds:.1f}s)")

    print()
    print(f"Import complete: {passed} passed, {failed} failed, {len(wav_files)} total")
    if fail_reasons:
        print("Failure reasons:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Combine valid segments into reference file
    print()
    print("Combining valid segments into reference file...")
    ref_path = await store.combine_reference(USER)
    metadata = await store.get_metadata(USER)
    print(f"Combined reference: {ref_path}")
    print(f"Combined duration: {metadata.combined_duration_seconds}s")
    print(f"Valid segments used: {metadata.total_valid_segments}")


if __name__ == "__main__":
    asyncio.run(main())
