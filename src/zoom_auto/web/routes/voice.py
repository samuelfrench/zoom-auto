"""Voice sample recording and upload endpoints.

Provides API endpoints for managing voice reference samples
used for TTS voice cloning.
"""

from __future__ import annotations

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

router = APIRouter()


class VoiceProfileResponse(BaseModel):
    """Response model for voice profile operations."""

    name: str
    sample_count: int
    sample_rate: int


class VoiceUploadResponse(BaseModel):
    """Response model for voice sample upload."""

    success: bool
    profile_name: str
    message: str


@router.get("/profiles")
async def list_profiles() -> list[VoiceProfileResponse]:
    """List all available voice profiles."""
    raise NotImplementedError("Voice profile listing not yet implemented")


@router.post("/upload/{profile_name}")
async def upload_sample(profile_name: str, file: UploadFile) -> VoiceUploadResponse:
    """Upload a voice sample to a profile.

    Args:
        profile_name: Name of the voice profile.
        file: WAV audio file upload.
    """
    raise NotImplementedError("Voice sample upload not yet implemented")


@router.delete("/profiles/{profile_name}")
async def delete_profile(profile_name: str) -> dict[str, str]:
    """Delete a voice profile and all its samples.

    Args:
        profile_name: Name of the profile to delete.
    """
    raise NotImplementedError("Profile deletion not yet implemented")
