"""Voice sample recording and upload endpoints.

Provides REST API endpoints for managing voice reference samples
used for TTS voice cloning. Supports uploading audio in various formats,
quality validation, segment management, and combining segments into
a reference file for TTS consumption.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from zoom_auto.tts.audio_validation import AudioQualityReport
from zoom_auto.tts.prompts import RECORDING_PROMPTS
from zoom_auto.tts.voice_store import SegmentMetadata, VoiceStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level voice store instance, initialized on first use
_voice_store: VoiceStore | None = None

# Allowed upload extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


def _get_voice_store() -> VoiceStore:
    """Get or create the module-level VoiceStore instance."""
    global _voice_store
    if _voice_store is None:
        _voice_store = VoiceStore()
    return _voice_store


def set_voice_store(store: VoiceStore) -> None:
    """Set the module-level VoiceStore instance (for testing / app startup).

    Args:
        store: The VoiceStore instance to use.
    """
    global _voice_store
    _voice_store = store


# --- Response Models ---


class PromptItem(BaseModel):
    """A single recording prompt."""

    index: int
    text: str


class PromptsResponse(BaseModel):
    """Response model for the prompts list."""

    prompts: list[PromptItem]
    total: int


class SegmentResponse(BaseModel):
    """Response model for a voice segment."""

    segment_id: str
    filename: str
    prompt_index: int
    prompt_text: str
    duration_seconds: float
    snr_db: float
    peak_amplitude: float
    rms_amplitude: float
    has_clipping: bool
    is_valid: bool
    recorded_at: str


class QualityResponse(BaseModel):
    """Response model for audio quality validation."""

    is_valid: bool
    snr_db: float
    peak_amplitude: float
    rms_amplitude: float
    duration_seconds: float
    has_clipping: bool
    issues: list[str]


class UploadResponse(BaseModel):
    """Response model for voice sample upload."""

    success: bool
    segment: SegmentResponse
    quality: QualityResponse
    message: str


class SamplesListResponse(BaseModel):
    """Response model for listing samples."""

    user: str
    segments: list[SegmentResponse]
    total: int


class CombineResponse(BaseModel):
    """Response model for combining samples."""

    success: bool
    combined_path: str
    duration_seconds: float
    message: str


class StatusResponse(BaseModel):
    """Response model for voice setup status."""

    user: str
    total_segments: int
    valid_segments: int
    total_valid_duration_seconds: float
    average_snr_db: float
    has_combined_reference: bool
    combined_duration_seconds: float
    ready_for_tts: bool
    recommendation: str


class DeleteResponse(BaseModel):
    """Response model for segment deletion."""

    success: bool
    message: str


# --- Helper functions ---


def _segment_to_response(seg: SegmentMetadata) -> SegmentResponse:
    """Convert a SegmentMetadata to a SegmentResponse."""
    return SegmentResponse(
        segment_id=seg.segment_id,
        filename=seg.filename,
        prompt_index=seg.prompt_index,
        prompt_text=seg.prompt_text,
        duration_seconds=round(seg.duration_seconds, 2),
        snr_db=round(seg.snr_db, 1),
        peak_amplitude=round(seg.peak_amplitude, 4),
        rms_amplitude=round(seg.rms_amplitude, 4),
        has_clipping=seg.has_clipping,
        is_valid=seg.is_valid,
        recorded_at=seg.recorded_at,
    )


def _quality_to_response(q: AudioQualityReport) -> QualityResponse:
    """Convert an AudioQualityReport to a QualityResponse."""
    return QualityResponse(
        is_valid=q.is_valid,
        snr_db=round(q.snr_db, 1),
        peak_amplitude=round(q.peak_amplitude, 4),
        rms_amplitude=round(q.rms_amplitude, 4),
        duration_seconds=round(q.duration_seconds, 2),
        has_clipping=q.has_clipping,
        issues=q.issues,
    )


def _get_extension(filename: str | None) -> str:
    """Extract and validate file extension from a filename."""
    if not filename:
        return "wav"
    ext = Path(filename).suffix.lstrip(".").lower()
    return ext if ext in ALLOWED_EXTENSIONS else "wav"


# --- Endpoints ---


@router.get("/prompts", response_model=PromptsResponse)
async def get_prompts() -> PromptsResponse:
    """Return the list of recording prompts.

    Returns the scripted prompts that users should read aloud when
    recording voice samples for TTS cloning.
    """
    prompts = [
        PromptItem(index=i, text=text) for i, text in enumerate(RECORDING_PROMPTS)
    ]
    return PromptsResponse(prompts=prompts, total=len(prompts))


@router.post("/upload", response_model=UploadResponse)
async def upload_sample(
    file: UploadFile,
    user: str = "default",
    prompt_index: int = -1,
    prompt_text: str = "",
) -> UploadResponse:
    """Upload a recorded audio segment.

    Accepts WAV, MP3, FLAC, and M4A files. The audio is auto-converted
    to WAV 22.05kHz mono 16-bit and validated for quality.

    Args:
        file: The audio file to upload.
        user: Username / profile name (query parameter).
        prompt_index: Index of the recording prompt (query parameter, -1 for free-form).
        prompt_text: The prompt text that was read (query parameter).
    """
    store = _get_voice_store()

    # Validate file extension
    ext = _get_extension(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file data
    audio_data = await file.read()
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(audio_data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(audio_data)} bytes (max {MAX_UPLOAD_SIZE})",
        )

    # If prompt_index is valid and no prompt_text was provided, fill it in
    if 0 <= prompt_index < len(RECORDING_PROMPTS) and not prompt_text:
        prompt_text = RECORDING_PROMPTS[prompt_index]

    try:
        segment, quality = await store.add_sample(
            user=user,
            audio_data=audio_data,
            prompt_index=prompt_index,
            prompt_text=prompt_text,
            source_format=ext,
        )
    except Exception as e:
        logger.error("Failed to process upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {e}") from e

    message = "Sample uploaded and validated successfully."
    if not quality.is_valid:
        issues_str = "; ".join(quality.issues)
        message = f"Sample uploaded but has quality issues: {issues_str}"

    return UploadResponse(
        success=True,
        segment=_segment_to_response(segment),
        quality=_quality_to_response(quality),
        message=message,
    )


@router.get("/samples/{user}", response_model=SamplesListResponse)
async def list_samples(user: str) -> SamplesListResponse:
    """List all voice samples for a user.

    Args:
        user: The username / profile name.
    """
    store = _get_voice_store()
    segments = await store.list_samples(user)

    return SamplesListResponse(
        user=user,
        segments=[_segment_to_response(s) for s in segments],
        total=len(segments),
    )


@router.delete("/samples/{user}/{segment_id}", response_model=DeleteResponse)
async def delete_sample(user: str, segment_id: str) -> DeleteResponse:
    """Delete a specific voice sample.

    Args:
        user: The username / profile name.
        segment_id: The segment ID to delete.
    """
    store = _get_voice_store()
    removed = await store.remove_sample(user, segment_id)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Segment {segment_id} not found for user {user}",
        )

    return DeleteResponse(
        success=True,
        message=f"Segment {segment_id} deleted successfully.",
    )


@router.post("/combine/{user}", response_model=CombineResponse)
async def combine_samples(user: str) -> CombineResponse:
    """Combine voice samples into a single reference WAV.

    Selects the best quality segments and concatenates them into a
    combined_reference.wav file suitable for TTS voice cloning.

    Args:
        user: The username / profile name.
    """
    store = _get_voice_store()

    try:
        combined_path = await store.combine_reference(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to combine samples: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to combine samples: {e}"
        ) from e

    metadata = await store.get_metadata(user)

    return CombineResponse(
        success=True,
        combined_path=str(combined_path),
        duration_seconds=metadata.combined_duration_seconds,
        message=f"Combined reference created: {metadata.combined_duration_seconds:.1f}s",
    )


@router.get("/status/{user}", response_model=StatusResponse)
async def get_status(user: str) -> StatusResponse:
    """Get voice setup status and quality information.

    Returns a summary of the user's voice sample collection including
    segment counts, quality metrics, and readiness for TTS cloning.

    Args:
        user: The username / profile name.
    """
    store = _get_voice_store()
    status = await store.get_status(user)

    return StatusResponse(**status)
