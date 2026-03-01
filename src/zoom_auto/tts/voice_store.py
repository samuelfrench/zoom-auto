"""Voice sample management and metadata storage.

Handles storage, retrieval, validation, and combination of voice reference
samples used for TTS voice cloning. Each user has a directory under
data/voice_samples/{user}/segments/ containing individual WAV segments,
a metadata.json tracking all segments and quality info, and optionally
a combined_reference.wav for TTS consumption.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from zoom_auto.config import TTSConfig
from zoom_auto.tts.audio_validation import (
    AudioQualityReport,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TARGET_SUBTYPE,
    convert_to_target_format,
    normalize_audio,
    validate_audio_data,
    validate_audio_file,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetadata:
    """Metadata for a single voice segment.

    Attributes:
        segment_id: Unique identifier for this segment.
        filename: Filename of the WAV file in the segments directory.
        prompt_index: Index of the prompt this segment was recorded for (-1 if free-form).
        prompt_text: The prompt text that was read.
        duration_seconds: Duration of the audio in seconds.
        snr_db: Signal-to-noise ratio in decibels.
        peak_amplitude: Peak absolute amplitude.
        rms_amplitude: RMS amplitude.
        has_clipping: Whether clipping was detected.
        is_valid: Whether the segment passed quality validation.
        recorded_at: ISO 8601 timestamp of when the segment was recorded.
    """

    segment_id: str
    filename: str
    prompt_index: int = -1
    prompt_text: str = ""
    duration_seconds: float = 0.0
    snr_db: float = 0.0
    peak_amplitude: float = 0.0
    rms_amplitude: float = 0.0
    has_clipping: bool = False
    is_valid: bool = False
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "segment_id": self.segment_id,
            "filename": self.filename,
            "prompt_index": self.prompt_index,
            "prompt_text": self.prompt_text,
            "duration_seconds": self.duration_seconds,
            "snr_db": self.snr_db,
            "peak_amplitude": self.peak_amplitude,
            "rms_amplitude": self.rms_amplitude,
            "has_clipping": self.has_clipping,
            "is_valid": self.is_valid,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentMetadata:
        """Deserialize from a dictionary."""
        return cls(
            segment_id=data["segment_id"],
            filename=data["filename"],
            prompt_index=data.get("prompt_index", -1),
            prompt_text=data.get("prompt_text", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            snr_db=data.get("snr_db", 0.0),
            peak_amplitude=data.get("peak_amplitude", 0.0),
            rms_amplitude=data.get("rms_amplitude", 0.0),
            has_clipping=data.get("has_clipping", False),
            is_valid=data.get("is_valid", False),
            recorded_at=data.get("recorded_at", ""),
        )


@dataclass
class UserVoiceMetadata:
    """Metadata for a user's entire voice sample collection.

    Attributes:
        user: Username / profile name.
        segments: List of segment metadata.
        combined_reference: Filename of the combined reference WAV, if generated.
        combined_duration_seconds: Total duration of the combined reference.
        total_valid_segments: Count of segments passing quality checks.
        created_at: ISO 8601 timestamp of when the profile was created.
        updated_at: ISO 8601 timestamp of the last update.
    """

    user: str
    segments: list[SegmentMetadata] = field(default_factory=list)
    combined_reference: str = ""
    combined_duration_seconds: float = 0.0
    total_valid_segments: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "user": self.user,
            "segments": [s.to_dict() for s in self.segments],
            "combined_reference": self.combined_reference,
            "combined_duration_seconds": self.combined_duration_seconds,
            "total_valid_segments": self.total_valid_segments,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserVoiceMetadata:
        """Deserialize from a dictionary."""
        return cls(
            user=data["user"],
            segments=[SegmentMetadata.from_dict(s) for s in data.get("segments", [])],
            combined_reference=data.get("combined_reference", ""),
            combined_duration_seconds=data.get("combined_duration_seconds", 0.0),
            total_valid_segments=data.get("total_valid_segments", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class VoiceProfile:
    """A voice profile with reference samples and metadata.

    Attributes:
        name: Human-readable name for this voice profile.
        sample_paths: Paths to WAV reference samples.
        sample_rate: Expected sample rate of the samples.
        description: Optional description of the voice characteristics.
        preferred_sample: Index of the preferred sample (best quality).
    """

    name: str
    sample_paths: list[Path] = field(default_factory=list)
    sample_rate: int = TARGET_SAMPLE_RATE
    description: str = ""
    preferred_sample: int = 0

    @property
    def primary_sample(self) -> Path | None:
        """Get the primary (preferred) voice sample path."""
        if not self.sample_paths:
            return None
        idx = min(self.preferred_sample, len(self.sample_paths) - 1)
        return self.sample_paths[idx]


class VoiceStore:
    """Manages voice sample files and profiles for TTS voice cloning.

    Provides methods for adding, removing, listing, validating, and combining
    voice samples. Each user has their own directory with segments and metadata.

    Directory structure:
        data/voice_samples/{user}/
            metadata.json
            segments/
                {segment_id}.wav
                ...
            combined_reference.wav
    """

    def __init__(self, config: TTSConfig | None = None, base_dir: Path | None = None) -> None:
        """Initialize the VoiceStore.

        Args:
            config: TTS configuration (uses voice_sample_dir from config).
            base_dir: Override base directory for voice samples.
                      If provided, takes precedence over config.
        """
        if base_dir is not None:
            self._base_dir = base_dir
        elif config is not None:
            self._base_dir = Path(config.voice_sample_dir)
        else:
            self._base_dir = Path("data/voice_samples")

        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, VoiceProfile] = {}

    @property
    def base_dir(self) -> Path:
        """Base directory for all voice samples."""
        return self._base_dir

    def _user_dir(self, user: str) -> Path:
        """Get the directory for a user's voice samples."""
        return self._base_dir / user

    def _segments_dir(self, user: str) -> Path:
        """Get the segments directory for a user."""
        return self._user_dir(user) / "segments"

    def _metadata_path(self, user: str) -> Path:
        """Get the metadata.json path for a user."""
        return self._user_dir(user) / "metadata.json"

    def _combined_path(self, user: str) -> Path:
        """Get the combined reference WAV path for a user."""
        return self._user_dir(user) / "combined_reference.wav"

    def _ensure_user_dirs(self, user: str) -> None:
        """Create user directories if they don't exist."""
        self._segments_dir(user).mkdir(parents=True, exist_ok=True)

    def _load_metadata(self, user: str) -> UserVoiceMetadata:
        """Load metadata for a user from disk.

        Args:
            user: The username.

        Returns:
            UserVoiceMetadata, or a fresh one if not found.
        """
        meta_path = self._metadata_path(user)
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                return UserVoiceMetadata.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load metadata for %s: %s", user, e)

        now = datetime.now(timezone.utc).isoformat()
        return UserVoiceMetadata(user=user, created_at=now, updated_at=now)

    def _save_metadata(self, metadata: UserVoiceMetadata) -> None:
        """Save metadata for a user to disk.

        Args:
            metadata: The metadata to save.
        """
        metadata.updated_at = datetime.now(timezone.utc).isoformat()
        metadata.total_valid_segments = sum(1 for s in metadata.segments if s.is_valid)

        meta_path = self._metadata_path(metadata.user)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    async def add_sample(
        self,
        user: str,
        audio_data: bytes,
        prompt_index: int = -1,
        prompt_text: str = "",
        source_format: str = "wav",
    ) -> tuple[SegmentMetadata, AudioQualityReport]:
        """Add a new voice sample to a user's profile.

        The audio data is saved as a WAV file, validated for quality,
        and metadata is updated.

        Args:
            user: Username / profile name.
            audio_data: Raw audio file bytes (WAV, MP3, FLAC, M4A, etc.).
            prompt_index: Index of the prompt (-1 for free-form).
            prompt_text: The prompt text that was read.
            source_format: Source audio format extension (wav, mp3, flac, m4a).

        Returns:
            Tuple of (SegmentMetadata, AudioQualityReport).
        """
        self._ensure_user_dirs(user)

        segment_id = str(uuid.uuid4())[:8]
        segments_dir = self._segments_dir(user)

        # Write the raw upload to a temp file
        temp_path = segments_dir / f"_temp_{segment_id}.{source_format}"
        target_path = segments_dir / f"{segment_id}.wav"

        try:
            temp_path.write_bytes(audio_data)

            # Convert to target format if needed
            if source_format.lower() != "wav":
                convert_to_target_format(temp_path, target_path)
                temp_path.unlink(missing_ok=True)
            else:
                # Even for WAV, ensure correct format (sample rate, channels, bit depth)
                try:
                    data, sr = sf.read(temp_path, dtype="float64")
                except Exception:
                    # If soundfile can't read it directly, try ffmpeg
                    convert_to_target_format(temp_path, target_path)
                    temp_path.unlink(missing_ok=True)
                else:
                    if data.ndim > 1:
                        data = data[:, 0]  # Take first channel
                    if sr != TARGET_SAMPLE_RATE:
                        # Resample via ffmpeg
                        convert_to_target_format(temp_path, target_path)
                        temp_path.unlink(missing_ok=True)
                    else:
                        # Already correct format, just rename
                        temp_path.rename(target_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            raise

        # Validate the final WAV
        quality = validate_audio_file(target_path)

        # Create segment metadata
        segment = SegmentMetadata(
            segment_id=segment_id,
            filename=f"{segment_id}.wav",
            prompt_index=prompt_index,
            prompt_text=prompt_text,
            duration_seconds=quality.duration_seconds,
            snr_db=quality.snr_db,
            peak_amplitude=quality.peak_amplitude,
            rms_amplitude=quality.rms_amplitude,
            has_clipping=quality.has_clipping,
            is_valid=quality.is_valid,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )

        # Update metadata
        metadata = self._load_metadata(user)
        metadata.segments.append(segment)
        self._save_metadata(metadata)

        logger.info(
            "Added segment %s for user %s (valid=%s, snr=%.1fdB, duration=%.1fs)",
            segment_id,
            user,
            quality.is_valid,
            quality.snr_db,
            quality.duration_seconds,
        )

        return segment, quality

    async def add_sample_from_array(
        self,
        user: str,
        audio_data: np.ndarray,
        sample_rate: int,
        prompt_index: int = -1,
        prompt_text: str = "",
    ) -> tuple[SegmentMetadata, AudioQualityReport]:
        """Add a voice sample from a numpy array (used by CLI recording).

        Args:
            user: Username / profile name.
            audio_data: Audio as a 1D numpy array (float64 or int16).
            sample_rate: Sample rate of the audio data.
            prompt_index: Index of the prompt (-1 for free-form).
            prompt_text: The prompt text that was read.

        Returns:
            Tuple of (SegmentMetadata, AudioQualityReport).
        """
        self._ensure_user_dirs(user)

        segment_id = str(uuid.uuid4())[:8]
        target_path = self._segments_dir(user) / f"{segment_id}.wav"

        # Ensure float64 for processing
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float64) / 32768.0
        elif audio_data.dtype == np.float32:
            audio_float = audio_data.astype(np.float64)
        else:
            audio_float = audio_data

        # Ensure mono
        if audio_float.ndim > 1:
            audio_float = audio_float[:, 0]

        # Validate before saving
        quality = validate_audio_data(audio_float, sample_rate)

        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            # Simple resampling: save at original rate, then convert via ffmpeg
            temp_path = target_path.with_suffix(".tmp.wav")
            sf.write(str(temp_path), audio_float, sample_rate, subtype=TARGET_SUBTYPE)
            convert_to_target_format(temp_path, target_path)
            temp_path.unlink(missing_ok=True)
            # Re-validate at target rate
            quality = validate_audio_file(target_path)
        else:
            sf.write(str(target_path), audio_float, TARGET_SAMPLE_RATE, subtype=TARGET_SUBTYPE)

        # Create segment metadata
        segment = SegmentMetadata(
            segment_id=segment_id,
            filename=f"{segment_id}.wav",
            prompt_index=prompt_index,
            prompt_text=prompt_text,
            duration_seconds=quality.duration_seconds,
            snr_db=quality.snr_db,
            peak_amplitude=quality.peak_amplitude,
            rms_amplitude=quality.rms_amplitude,
            has_clipping=quality.has_clipping,
            is_valid=quality.is_valid,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )

        # Update metadata
        metadata = self._load_metadata(user)
        metadata.segments.append(segment)
        self._save_metadata(metadata)

        logger.info(
            "Added segment %s for user %s (valid=%s, snr=%.1fdB, duration=%.1fs)",
            segment_id,
            user,
            quality.is_valid,
            quality.snr_db,
            quality.duration_seconds,
        )

        return segment, quality

    async def remove_sample(self, user: str, segment_id: str) -> bool:
        """Remove a specific voice segment.

        Deletes the WAV file and removes the entry from metadata.

        Args:
            user: Username / profile name.
            segment_id: The segment ID to remove.

        Returns:
            True if the segment was found and removed, False otherwise.
        """
        metadata = self._load_metadata(user)
        segment = None
        for s in metadata.segments:
            if s.segment_id == segment_id:
                segment = s
                break

        if segment is None:
            logger.warning("Segment %s not found for user %s", segment_id, user)
            return False

        # Delete the WAV file
        wav_path = self._segments_dir(user) / segment.filename
        wav_path.unlink(missing_ok=True)

        # Remove from metadata
        metadata.segments = [s for s in metadata.segments if s.segment_id != segment_id]
        self._save_metadata(metadata)

        logger.info("Removed segment %s for user %s", segment_id, user)
        return True

    async def list_samples(self, user: str) -> list[SegmentMetadata]:
        """List all voice segments for a user.

        Args:
            user: Username / profile name.

        Returns:
            List of SegmentMetadata for all segments.
        """
        metadata = self._load_metadata(user)
        return metadata.segments

    async def get_metadata(self, user: str) -> UserVoiceMetadata:
        """Get the full metadata for a user's voice profile.

        Args:
            user: Username / profile name.

        Returns:
            UserVoiceMetadata with all segment info.
        """
        return self._load_metadata(user)

    async def get_status(self, user: str) -> dict[str, Any]:
        """Get voice setup status and quality summary for a user.

        Args:
            user: Username / profile name.

        Returns:
            Dictionary with status information.
        """
        metadata = self._load_metadata(user)
        has_combined = self._combined_path(user).exists()

        valid_segments = [s for s in metadata.segments if s.is_valid]
        total_duration = sum(s.duration_seconds for s in valid_segments)
        avg_snr = (
            sum(s.snr_db for s in valid_segments) / len(valid_segments)
            if valid_segments
            else 0.0
        )

        return {
            "user": user,
            "total_segments": len(metadata.segments),
            "valid_segments": len(valid_segments),
            "total_valid_duration_seconds": round(total_duration, 1),
            "average_snr_db": round(avg_snr, 1),
            "has_combined_reference": has_combined,
            "combined_duration_seconds": metadata.combined_duration_seconds,
            "ready_for_tts": has_combined and metadata.combined_duration_seconds >= 120.0,
            "recommendation": _get_recommendation(len(valid_segments), total_duration, has_combined),
        }

    async def combine_reference(
        self,
        user: str,
        max_duration_seconds: float = 300.0,
        min_duration_seconds: float = 120.0,
    ) -> Path:
        """Combine the best voice segments into a single reference WAV.

        Selects valid segments sorted by quality (SNR), concatenates them
        with short silences, and normalizes the volume.

        Args:
            user: Username / profile name.
            max_duration_seconds: Maximum combined duration (default 5 min).
            min_duration_seconds: Minimum combined duration warning threshold (default 2 min).

        Returns:
            Path to the combined reference WAV file.

        Raises:
            ValueError: If there are no valid segments.
        """
        metadata = self._load_metadata(user)
        valid_segments = [s for s in metadata.segments if s.is_valid]

        if not valid_segments:
            raise ValueError(f"No valid segments found for user {user}")

        # Sort by SNR descending (best quality first)
        valid_segments.sort(key=lambda s: s.snr_db, reverse=True)

        # Concatenate segments with 0.5s silence gaps
        silence_samples = int(TARGET_SAMPLE_RATE * 0.5)
        silence = np.zeros(silence_samples, dtype=np.float64)

        combined_parts: list[np.ndarray] = []
        total_duration = 0.0

        for segment in valid_segments:
            if total_duration >= max_duration_seconds:
                break

            wav_path = self._segments_dir(user) / segment.filename
            if not wav_path.exists():
                logger.warning("Segment file missing: %s", wav_path)
                continue

            try:
                data, sr = sf.read(str(wav_path), dtype="float64")
                if data.ndim > 1:
                    data = data[:, 0]
            except Exception as e:
                logger.warning("Failed to read segment %s: %s", segment.segment_id, e)
                continue

            if combined_parts:
                combined_parts.append(silence)
                total_duration += 0.5

            combined_parts.append(data)
            total_duration += len(data) / sr

        if not combined_parts:
            raise ValueError(f"No readable segments found for user {user}")

        # Concatenate all parts
        combined = np.concatenate(combined_parts)

        # Normalize volume
        combined = normalize_audio(combined, target_rms=0.1)

        # Save combined reference
        output_path = self._combined_path(user)
        sf.write(str(output_path), combined, TARGET_SAMPLE_RATE, subtype=TARGET_SUBTYPE)

        # Update metadata
        metadata.combined_reference = "combined_reference.wav"
        metadata.combined_duration_seconds = round(total_duration, 1)
        self._save_metadata(metadata)

        if total_duration < min_duration_seconds:
            logger.warning(
                "Combined reference for %s is only %.1fs (recommended minimum: %.0fs)",
                user,
                total_duration,
                min_duration_seconds,
            )

        logger.info(
            "Created combined reference for %s: %.1fs from %d segments",
            user,
            total_duration,
            len(combined_parts) // 2 + 1,  # Account for silence gaps
        )

        return output_path

    async def get_reference_path(self, user: str) -> Path | None:
        """Get the path to the combined reference WAV for a user.

        Args:
            user: Username / profile name.

        Returns:
            Path to the combined reference WAV, or None if not yet generated.
        """
        path = self._combined_path(user)
        return path if path.exists() else None

    # Legacy compatibility methods

    async def scan_directory(self) -> list[VoiceProfile]:
        """Scan the voice sample directory for available profiles.

        Returns:
            List of discovered voice profiles.
        """
        profiles: list[VoiceProfile] = []
        if not self._base_dir.exists():
            return profiles

        for user_dir in sorted(self._base_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            user = user_dir.name
            metadata = self._load_metadata(user)
            sample_paths = [
                self._segments_dir(user) / s.filename
                for s in metadata.segments
                if (self._segments_dir(user) / s.filename).exists()
            ]
            combined = self._combined_path(user)
            if combined.exists():
                sample_paths.insert(0, combined)

            profile = VoiceProfile(
                name=user,
                sample_paths=sample_paths,
                sample_rate=TARGET_SAMPLE_RATE,
                description=f"{len(metadata.segments)} segments, "
                f"{metadata.total_valid_segments} valid",
            )
            profiles.append(profile)
            self._profiles[user] = profile

        return profiles

    async def get_profile(self, name: str) -> VoiceProfile | None:
        """Get a voice profile by name.

        Args:
            name: The profile name.

        Returns:
            The VoiceProfile, or None if not found.
        """
        if name not in self._profiles:
            await self.scan_directory()
        return self._profiles.get(name)

    async def list_profiles(self) -> list[VoiceProfile]:
        """List all available voice profiles.

        Returns:
            List of all voice profiles.
        """
        await self.scan_directory()
        return list(self._profiles.values())

    async def validate_sample(self, path: Path) -> bool:
        """Validate that a voice sample meets requirements.

        Args:
            path: Path to the sample file.

        Returns:
            True if the sample is valid.
        """
        try:
            report = validate_audio_file(path)
            return report.is_valid
        except (FileNotFoundError, RuntimeError):
            return False


def _get_recommendation(valid_count: int, total_duration: float, has_combined: bool) -> str:
    """Generate a recommendation string based on voice setup status."""
    if valid_count == 0:
        return "Record voice samples to get started. Aim for at least 10 prompts."
    if valid_count < 5:
        return f"Only {valid_count} valid segments. Record more for better quality (aim for 15+)."
    if total_duration < 60:
        return (
            f"Total duration is {total_duration:.0f}s. "
            "Record more prompts to reach at least 2 minutes."
        )
    if total_duration < 120:
        return (
            f"Good progress! {total_duration:.0f}s recorded. "
            "A few more prompts will improve cloning quality."
        )
    if not has_combined:
        return "Enough samples collected! Run 'combine' to create the reference file."
    return "Voice setup complete and ready for TTS cloning."
