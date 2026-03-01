"""Tests for the voice sample collection system.

Tests cover:
- Audio validation (SNR, clipping, volume, duration)
- VoiceStore operations (add, remove, list, combine, metadata)
- Recording prompts
- Web API endpoints
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from zoom_auto.tts.audio_validation import (
    TARGET_SAMPLE_RATE,
    AudioQualityReport,
    normalize_audio,
    validate_audio_data,
    validate_audio_file,
)
from zoom_auto.tts.prompts import RECORDING_PROMPTS
from zoom_auto.tts.voice_store import (
    SegmentMetadata,
    UserVoiceMetadata,
    VoiceStore,
)


# --- Fixtures ---


@pytest.fixture
def tmp_voice_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for voice samples."""
    voice_dir = tmp_path / "voice_samples"
    voice_dir.mkdir()
    return voice_dir


@pytest.fixture
def voice_store(tmp_voice_dir: Path) -> VoiceStore:
    """Create a VoiceStore with a temporary directory."""
    return VoiceStore(base_dir=tmp_voice_dir)


@pytest.fixture
def good_audio() -> np.ndarray:
    """Generate a good quality audio signal (clean tone + low noise).

    Returns a 3-second 440Hz sine wave with low background noise.
    """
    duration = 3.0
    t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
    # Clean 440Hz tone
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    # Add very light noise
    noise = np.random.default_rng(42).normal(0, 0.002, len(t))
    return signal + noise


@pytest.fixture
def quiet_audio() -> np.ndarray:
    """Generate a too-quiet audio signal."""
    duration = 3.0
    t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
    return 0.005 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def clipping_audio() -> np.ndarray:
    """Generate an audio signal with clipping."""
    duration = 3.0
    t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
    signal = 1.5 * np.sin(2 * np.pi * 440 * t)
    # Clip to [-1.0, 1.0]
    return np.clip(signal, -1.0, 1.0)


@pytest.fixture
def noisy_audio() -> np.ndarray:
    """Generate a noisy audio signal (low SNR)."""
    duration = 3.0
    t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.default_rng(42).normal(0, 0.08, len(t))
    return signal + noise


@pytest.fixture
def short_audio() -> np.ndarray:
    """Generate a too-short audio signal (0.2s)."""
    duration = 0.2
    t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def good_wav_file(tmp_path: Path, good_audio: np.ndarray) -> Path:
    """Save good audio to a WAV file."""
    path = tmp_path / "good.wav"
    sf.write(str(path), good_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return path


@pytest.fixture
def good_wav_bytes(good_wav_file: Path) -> bytes:
    """Read good WAV file as bytes."""
    return good_wav_file.read_bytes()


# --- Prompts Tests ---


class TestRecordingPrompts:
    """Tests for the recording prompts constant."""

    def test_prompts_exist(self) -> None:
        """Verify prompts list is populated."""
        assert len(RECORDING_PROMPTS) == 20

    def test_prompts_are_strings(self) -> None:
        """Verify all prompts are non-empty strings."""
        for prompt in RECORDING_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_prompts_include_key_types(self) -> None:
        """Verify prompts cover different speech patterns."""
        all_text = " ".join(RECORDING_PROMPTS).lower()
        # Greetings
        assert "good morning" in all_text
        # Questions
        assert "?" in " ".join(RECORDING_PROMPTS)
        # Technical content
        assert "api" in all_text or "database" in all_text
        # Pangram (for phoneme coverage)
        assert "quick brown fox" in all_text


# --- Audio Validation Tests ---


class TestAudioValidation:
    """Tests for audio quality validation."""

    def test_good_audio_passes(self, good_audio: np.ndarray) -> None:
        """Good quality audio should pass validation."""
        report = validate_audio_data(good_audio, TARGET_SAMPLE_RATE)
        assert report.is_valid
        assert report.snr_db > 20.0
        assert not report.has_clipping
        assert len(report.issues) == 0

    def test_quiet_audio_fails(self, quiet_audio: np.ndarray) -> None:
        """Too-quiet audio should fail validation."""
        report = validate_audio_data(quiet_audio, TARGET_SAMPLE_RATE)
        assert not report.is_valid
        assert any("quiet" in issue.lower() for issue in report.issues)

    def test_clipping_audio_detected(self, clipping_audio: np.ndarray) -> None:
        """Clipping should be detected."""
        report = validate_audio_data(clipping_audio, TARGET_SAMPLE_RATE)
        assert report.has_clipping
        assert any("clipping" in issue.lower() for issue in report.issues)

    def test_short_audio_fails(self, short_audio: np.ndarray) -> None:
        """Too-short audio should fail validation."""
        report = validate_audio_data(short_audio, TARGET_SAMPLE_RATE)
        assert not report.is_valid
        assert any("short" in issue.lower() for issue in report.issues)

    def test_long_audio_fails(self) -> None:
        """Too-long audio should fail validation."""
        duration = 35.0  # Exceeds 30s max
        samples = int(TARGET_SAMPLE_RATE * duration)
        audio = 0.1 * np.ones(samples)
        report = validate_audio_data(audio, TARGET_SAMPLE_RATE)
        assert not report.is_valid
        assert any("long" in issue.lower() for issue in report.issues)

    def test_empty_audio(self) -> None:
        """Empty audio should have zero metrics."""
        report = validate_audio_data(np.array([]), TARGET_SAMPLE_RATE)
        assert report.peak_amplitude == 0.0
        assert report.rms_amplitude == 0.0
        assert report.snr_db == 0.0

    def test_validate_file(self, good_wav_file: Path) -> None:
        """Validation of a WAV file should work."""
        report = validate_audio_file(good_wav_file)
        assert report.is_valid
        assert report.sample_rate == TARGET_SAMPLE_RATE

    def test_validate_missing_file(self, tmp_path: Path) -> None:
        """Validation of a missing file should raise."""
        with pytest.raises(FileNotFoundError):
            validate_audio_file(tmp_path / "nonexistent.wav")

    def test_report_fields(self, good_audio: np.ndarray) -> None:
        """Verify all report fields are populated."""
        report = validate_audio_data(good_audio, TARGET_SAMPLE_RATE)
        assert isinstance(report.is_valid, bool)
        assert isinstance(report.snr_db, float)
        assert isinstance(report.peak_amplitude, float)
        assert isinstance(report.rms_amplitude, float)
        assert isinstance(report.duration_seconds, float)
        assert isinstance(report.sample_rate, int)
        assert isinstance(report.channels, int)
        assert isinstance(report.has_clipping, bool)
        assert isinstance(report.issues, list)


class TestNormalizeAudio:
    """Tests for audio normalization."""

    def test_normalize_scales_volume(self) -> None:
        """Normalization should adjust RMS to target."""
        audio = np.ones(1000) * 0.5
        normalized = normalize_audio(audio, target_rms=0.1)
        actual_rms = float(np.sqrt(np.mean(normalized**2)))
        assert abs(actual_rms - 0.1) < 0.01

    def test_normalize_prevents_clipping(self) -> None:
        """Normalization should not exceed peak of 0.95."""
        audio = np.ones(1000) * 0.01
        normalized = normalize_audio(audio, target_rms=0.5)
        assert float(np.max(np.abs(normalized))) <= 0.95

    def test_normalize_silent_audio(self) -> None:
        """Normalizing silence should return silence."""
        audio = np.zeros(1000)
        normalized = normalize_audio(audio, target_rms=0.1)
        assert float(np.max(np.abs(normalized))) == 0.0


# --- Segment Metadata Tests ---


class TestSegmentMetadata:
    """Tests for SegmentMetadata serialization."""

    def test_round_trip(self) -> None:
        """Metadata should survive serialization round trip."""
        segment = SegmentMetadata(
            segment_id="abc12345",
            filename="abc12345.wav",
            prompt_index=3,
            prompt_text="Test prompt",
            duration_seconds=2.5,
            snr_db=25.0,
            peak_amplitude=0.8,
            rms_amplitude=0.15,
            has_clipping=False,
            is_valid=True,
            recorded_at="2026-03-01T10:00:00+00:00",
        )
        data = segment.to_dict()
        restored = SegmentMetadata.from_dict(data)
        assert restored.segment_id == segment.segment_id
        assert restored.prompt_index == segment.prompt_index
        assert restored.snr_db == segment.snr_db
        assert restored.is_valid == segment.is_valid


class TestUserVoiceMetadata:
    """Tests for UserVoiceMetadata serialization."""

    def test_round_trip(self) -> None:
        """User metadata should survive serialization round trip."""
        meta = UserVoiceMetadata(
            user="testuser",
            segments=[
                SegmentMetadata(
                    segment_id="seg1",
                    filename="seg1.wav",
                    is_valid=True,
                ),
            ],
            combined_reference="combined_reference.wav",
            combined_duration_seconds=120.5,
            total_valid_segments=1,
            created_at="2026-03-01T10:00:00+00:00",
            updated_at="2026-03-01T10:00:00+00:00",
        )
        data = meta.to_dict()
        restored = UserVoiceMetadata.from_dict(data)
        assert restored.user == "testuser"
        assert len(restored.segments) == 1
        assert restored.combined_duration_seconds == 120.5


# --- VoiceStore Tests ---


class TestVoiceStore:
    """Tests for the VoiceStore class."""

    @pytest.mark.asyncio
    async def test_add_sample_from_array(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Adding a sample from array should save WAV and metadata."""
        segment, quality = await voice_store.add_sample_from_array(
            user="testuser",
            audio_data=good_audio,
            sample_rate=TARGET_SAMPLE_RATE,
            prompt_index=0,
            prompt_text="Test prompt",
        )

        assert segment.segment_id
        assert segment.filename.endswith(".wav")
        assert segment.prompt_index == 0
        assert segment.prompt_text == "Test prompt"
        assert quality.is_valid

        # Verify file exists
        wav_path = voice_store.base_dir / "testuser" / "segments" / segment.filename
        assert wav_path.exists()

        # Verify metadata was saved
        meta_path = voice_store.base_dir / "testuser" / "metadata.json"
        assert meta_path.exists()

    @pytest.mark.asyncio
    async def test_add_sample_from_bytes(
        self,
        voice_store: VoiceStore,
        good_wav_bytes: bytes,
    ) -> None:
        """Adding a sample from WAV bytes should work."""
        segment, quality = await voice_store.add_sample(
            user="testuser",
            audio_data=good_wav_bytes,
            prompt_index=1,
            prompt_text="Test prompt 2",
            source_format="wav",
        )

        assert segment.segment_id
        assert quality.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_list_samples(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Listing samples should return all added segments."""
        await voice_store.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt 1"
        )
        await voice_store.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 1, "Prompt 2"
        )

        samples = await voice_store.list_samples("testuser")
        assert len(samples) == 2

    @pytest.mark.asyncio
    async def test_list_samples_empty_user(self, voice_store: VoiceStore) -> None:
        """Listing samples for nonexistent user returns empty list."""
        samples = await voice_store.list_samples("nonexistent")
        assert len(samples) == 0

    @pytest.mark.asyncio
    async def test_remove_sample(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Removing a sample should delete the WAV and update metadata."""
        segment, _ = await voice_store.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )

        # Verify it exists
        wav_path = voice_store.base_dir / "testuser" / "segments" / segment.filename
        assert wav_path.exists()

        # Remove it
        result = await voice_store.remove_sample("testuser", segment.segment_id)
        assert result is True

        # Verify it's gone
        assert not wav_path.exists()
        samples = await voice_store.list_samples("testuser")
        assert len(samples) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_sample(self, voice_store: VoiceStore) -> None:
        """Removing a nonexistent sample should return False."""
        result = await voice_store.remove_sample("testuser", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metadata(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Getting metadata should return full user info."""
        await voice_store.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )

        metadata = await voice_store.get_metadata("testuser")
        assert metadata.user == "testuser"
        assert len(metadata.segments) == 1
        assert metadata.created_at
        assert metadata.updated_at

    @pytest.mark.asyncio
    async def test_get_status(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Getting status should return summary info."""
        await voice_store.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )

        status = await voice_store.get_status("testuser")
        assert status["user"] == "testuser"
        assert status["total_segments"] == 1
        assert status["valid_segments"] >= 0
        assert "recommendation" in status
        assert "ready_for_tts" in status

    @pytest.mark.asyncio
    async def test_combine_reference(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Combining valid segments should produce a reference WAV."""
        # Add several valid segments
        for i in range(5):
            await voice_store.add_sample_from_array(
                "testuser", good_audio, TARGET_SAMPLE_RATE, i, f"Prompt {i}"
            )

        path = await voice_store.combine_reference("testuser")
        assert path.exists()
        assert path.name == "combined_reference.wav"

        # Verify the combined file is readable
        data, sr = sf.read(str(path))
        assert sr == TARGET_SAMPLE_RATE
        assert len(data) > 0

        # Verify metadata was updated
        metadata = await voice_store.get_metadata("testuser")
        assert metadata.combined_reference == "combined_reference.wav"
        assert metadata.combined_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_combine_no_valid_segments(self, voice_store: VoiceStore) -> None:
        """Combining with no valid segments should raise ValueError."""
        with pytest.raises(ValueError, match="No valid segments"):
            await voice_store.combine_reference("testuser")

    @pytest.mark.asyncio
    async def test_get_reference_path(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Getting reference path should return None before combine, path after."""
        # Before combine
        path = await voice_store.get_reference_path("testuser")
        assert path is None

        # Add segments and combine
        for i in range(3):
            await voice_store.add_sample_from_array(
                "testuser", good_audio, TARGET_SAMPLE_RATE, i, f"Prompt {i}"
            )
        await voice_store.combine_reference("testuser")

        # After combine
        path = await voice_store.get_reference_path("testuser")
        assert path is not None
        assert path.exists()

    @pytest.mark.asyncio
    async def test_validate_sample(
        self,
        voice_store: VoiceStore,
        good_wav_file: Path,
    ) -> None:
        """validate_sample should return True for good audio."""
        result = await voice_store.validate_sample(good_wav_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_missing_sample(
        self,
        voice_store: VoiceStore,
        tmp_path: Path,
    ) -> None:
        """validate_sample should return False for missing file."""
        result = await voice_store.validate_sample(tmp_path / "nonexistent.wav")
        assert result is False

    @pytest.mark.asyncio
    async def test_scan_directory(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Scanning should discover user profiles."""
        await voice_store.add_sample_from_array(
            "user1", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )
        await voice_store.add_sample_from_array(
            "user2", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )

        profiles = await voice_store.scan_directory()
        names = [p.name for p in profiles]
        assert "user1" in names
        assert "user2" in names

    @pytest.mark.asyncio
    async def test_get_profile(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Getting a profile by name should work after scan."""
        await voice_store.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )

        profile = await voice_store.get_profile("testuser")
        assert profile is not None
        assert profile.name == "testuser"
        assert len(profile.sample_paths) > 0

    @pytest.mark.asyncio
    async def test_add_int16_audio(
        self,
        voice_store: VoiceStore,
    ) -> None:
        """Adding int16 audio data should auto-convert."""
        duration = 3.0
        t = np.linspace(
            0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False
        )
        audio_int16 = (0.3 * np.sin(2 * np.pi * 440 * t) * 32768).astype(np.int16)

        segment, quality = await voice_store.add_sample_from_array(
            "testuser", audio_int16, TARGET_SAMPLE_RATE, 0, "Prompt"
        )
        assert segment.segment_id
        assert quality.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_metadata_persistence(
        self,
        tmp_voice_dir: Path,
        good_audio: np.ndarray,
    ) -> None:
        """Metadata should persist across VoiceStore instances."""
        store1 = VoiceStore(base_dir=tmp_voice_dir)
        await store1.add_sample_from_array(
            "testuser", good_audio, TARGET_SAMPLE_RATE, 0, "Prompt"
        )

        # Create a new store pointing to the same directory
        store2 = VoiceStore(base_dir=tmp_voice_dir)
        samples = await store2.list_samples("testuser")
        assert len(samples) == 1
        assert samples[0].prompt_text == "Prompt"

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(
        self,
        voice_store: VoiceStore,
        good_audio: np.ndarray,
    ) -> None:
        """Different users should have isolated sample collections."""
        await voice_store.add_sample_from_array(
            "alice", good_audio, TARGET_SAMPLE_RATE, 0, "Alice prompt"
        )
        await voice_store.add_sample_from_array(
            "bob", good_audio, TARGET_SAMPLE_RATE, 0, "Bob prompt"
        )

        alice_samples = await voice_store.list_samples("alice")
        bob_samples = await voice_store.list_samples("bob")
        assert len(alice_samples) == 1
        assert len(bob_samples) == 1
        assert alice_samples[0].prompt_text == "Alice prompt"
        assert bob_samples[0].prompt_text == "Bob prompt"


# --- Web API Tests ---


class TestVoiceWebAPI:
    """Tests for the voice web API endpoints using FastAPI TestClient."""

    @pytest.fixture
    def client(self, voice_store: VoiceStore):
        """Create a FastAPI test client with the voice router."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from zoom_auto.web.routes.voice import router, set_voice_store

        app = FastAPI()
        app.include_router(router, prefix="/api/voice")
        set_voice_store(voice_store)
        return TestClient(app)

    def test_get_prompts(self, client) -> None:
        """GET /api/voice/prompts should return all prompts."""
        response = client.get("/api/voice/prompts")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 20
        assert len(data["prompts"]) == 20
        assert data["prompts"][0]["index"] == 0
        assert data["prompts"][0]["text"] == RECORDING_PROMPTS[0]

    def test_upload_wav(self, client, good_wav_bytes: bytes) -> None:
        """POST /api/voice/upload should accept WAV uploads."""
        response = client.post(
            "/api/voice/upload",
            params={"user": "testuser", "prompt_index": 0},
            files={"file": ("test.wav", good_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["segment"]["prompt_index"] == 0
        assert data["quality"]["duration_seconds"] > 0

    def test_upload_empty_file(self, client) -> None:
        """POST /api/voice/upload should reject empty files."""
        response = client.post(
            "/api/voice/upload",
            params={"user": "testuser"},
            files={"file": ("test.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400

    def test_list_samples(self, client, good_wav_bytes: bytes) -> None:
        """GET /api/voice/samples/{user} should list uploaded samples."""
        # Upload first
        client.post(
            "/api/voice/upload",
            params={"user": "testuser", "prompt_index": 0},
            files={"file": ("test.wav", good_wav_bytes, "audio/wav")},
        )

        response = client.get("/api/voice/samples/testuser")
        assert response.status_code == 200
        data = response.json()
        assert data["user"] == "testuser"
        assert data["total"] == 1

    def test_list_samples_empty_user(self, client) -> None:
        """GET /api/voice/samples/{user} should return empty for unknown user."""
        response = client.get("/api/voice/samples/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    def test_delete_sample(self, client, good_wav_bytes: bytes) -> None:
        """DELETE /api/voice/samples/{user}/{segment_id} should delete."""
        # Upload first
        upload_resp = client.post(
            "/api/voice/upload",
            params={"user": "testuser"},
            files={"file": ("test.wav", good_wav_bytes, "audio/wav")},
        )
        segment_id = upload_resp.json()["segment"]["segment_id"]

        # Delete
        response = client.delete(f"/api/voice/samples/testuser/{segment_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True

        # Verify it's gone
        list_resp = client.get("/api/voice/samples/testuser")
        assert list_resp.json()["total"] == 0

    def test_delete_nonexistent_sample(self, client) -> None:
        """DELETE for nonexistent sample should return 404."""
        response = client.delete("/api/voice/samples/testuser/nonexistent")
        assert response.status_code == 404

    def test_combine_samples(self, client, good_wav_bytes: bytes) -> None:
        """POST /api/voice/combine/{user} should combine valid samples."""
        # Upload several samples
        for i in range(3):
            client.post(
                "/api/voice/upload",
                params={"user": "testuser", "prompt_index": i},
                files={"file": (f"test{i}.wav", good_wav_bytes, "audio/wav")},
            )

        response = client.post("/api/voice/combine/testuser")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["duration_seconds"] > 0

    def test_combine_no_samples(self, client) -> None:
        """POST /api/voice/combine/{user} with no samples should fail."""
        response = client.post("/api/voice/combine/testuser")
        assert response.status_code == 400

    def test_get_status(self, client, good_wav_bytes: bytes) -> None:
        """GET /api/voice/status/{user} should return status info."""
        # Upload a sample
        client.post(
            "/api/voice/upload",
            params={"user": "testuser"},
            files={"file": ("test.wav", good_wav_bytes, "audio/wav")},
        )

        response = client.get("/api/voice/status/testuser")
        assert response.status_code == 200
        data = response.json()
        assert data["user"] == "testuser"
        assert data["total_segments"] == 1
        assert "recommendation" in data
        assert "ready_for_tts" in data

    def test_get_status_empty_user(self, client) -> None:
        """GET /api/voice/status/{user} for unknown user should work."""
        response = client.get("/api/voice/status/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["total_segments"] == 0
        assert data["ready_for_tts"] is False

    def test_upload_fills_prompt_text_automatically(
        self, client, good_wav_bytes: bytes
    ) -> None:
        """Upload with prompt_index should auto-fill prompt_text."""
        response = client.post(
            "/api/voice/upload",
            params={"user": "testuser", "prompt_index": 5},
            files={"file": ("test.wav", good_wav_bytes, "audio/wav")},
        )
        data = response.json()
        assert data["segment"]["prompt_text"] == RECORDING_PROMPTS[5]
