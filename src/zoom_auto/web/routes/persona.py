"""Persona configuration endpoints.

Provides API endpoints for managing the communication persona
used for response generation.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class PersonaConfigResponse(BaseModel):
    """Response model for persona configuration."""

    name: str
    style_weight: float
    formality_level: float
    avg_response_length: int
    topic_expertise: list[str]


class PersonaUpdateRequest(BaseModel):
    """Request model for updating persona settings."""

    name: str | None = None
    style_weight: float | None = None
    formality_level: float | None = None
    topic_expertise: list[str] | None = None


@router.get("/config")
async def get_persona_config() -> PersonaConfigResponse:
    """Get the current persona configuration."""
    raise NotImplementedError("Persona config retrieval not yet implemented")


@router.put("/config")
async def update_persona_config(request: PersonaUpdateRequest) -> PersonaConfigResponse:
    """Update persona configuration settings.

    Args:
        request: Fields to update.
    """
    raise NotImplementedError("Persona config update not yet implemented")


@router.post("/rebuild")
async def rebuild_persona() -> dict[str, str]:
    """Trigger a persona rebuild from source data."""
    raise NotImplementedError("Persona rebuild not yet implemented")
