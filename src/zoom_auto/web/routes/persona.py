"""Persona configuration endpoints.

Provides API endpoints for managing the communication persona
used for response generation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from zoom_auto.config import PersonaConfig
from zoom_auto.persona.builder import PersonaBuilder, PersonaProfile, TextSample
from zoom_auto.persona.knowledge_store import KnowledgeStore
from zoom_auto.persona.sources.project import ProjectIndexer

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level persona state
_persona_config: PersonaConfig | None = None
_persona_profile: PersonaProfile | None = None


def set_persona_config(config: PersonaConfig) -> None:
    """Set the persona configuration (called at app startup).

    Args:
        config: The persona configuration from settings.
    """
    global _persona_config
    _persona_config = config


def _get_persona_config() -> PersonaConfig:
    """Get the current persona config."""
    if _persona_config is None:
        return PersonaConfig()
    return _persona_config


def _get_profile_path() -> Path:
    """Get the path to the persona profile TOML file."""
    config = _get_persona_config()
    return Path(config.data_dir) / "profile.toml"


def _load_profile() -> PersonaProfile:
    """Load the current persona profile from disk."""
    global _persona_profile
    path = _get_profile_path()
    if path.exists():
        _persona_profile = PersonaProfile.from_toml(path)
    else:
        _persona_profile = PersonaProfile()
    return _persona_profile


class PersonaConfigResponse(BaseModel):
    """Response model for persona configuration."""

    name: str
    formality: float
    verbosity: float
    technical_depth: float
    assertiveness: float
    avg_response_words: int
    greeting_style: str
    agreement_style: str
    filler_words: dict[str, float]
    common_phrases: list[str]
    preferred_terms: list[str]
    avoided_terms: list[str]
    standup_format: str
    vocabulary_richness: float
    question_frequency: float
    exclamation_rate: float


class PersonaUpdateRequest(BaseModel):
    """Request model for updating persona settings."""

    name: str | None = None
    formality: float | None = None
    verbosity: float | None = None
    technical_depth: float | None = None
    assertiveness: float | None = None
    avg_response_words: int | None = None
    greeting_style: str | None = None
    agreement_style: str | None = None
    filler_words: dict[str, float] | None = None
    common_phrases: list[str] | None = None
    preferred_terms: list[str] | None = None
    avoided_terms: list[str] | None = None
    standup_format: str | None = None
    vocabulary_richness: float | None = None
    question_frequency: float | None = None
    exclamation_rate: float | None = None


def _profile_to_response(profile: PersonaProfile) -> PersonaConfigResponse:
    """Convert a PersonaProfile to the API response model."""
    return PersonaConfigResponse(
        name=profile.name,
        formality=profile.formality,
        verbosity=profile.verbosity,
        technical_depth=profile.technical_depth,
        assertiveness=profile.assertiveness,
        avg_response_words=profile.avg_response_words,
        greeting_style=profile.greeting_style,
        agreement_style=profile.agreement_style,
        filler_words=profile.filler_words,
        common_phrases=profile.common_phrases,
        preferred_terms=profile.preferred_terms,
        avoided_terms=profile.avoided_terms,
        standup_format=profile.standup_format,
        vocabulary_richness=profile.vocabulary_richness,
        question_frequency=profile.question_frequency,
        exclamation_rate=profile.exclamation_rate,
    )


@router.get("/config", response_model=PersonaConfigResponse)
async def get_persona_config() -> PersonaConfigResponse:
    """Get the current persona configuration."""
    profile = _load_profile()
    return _profile_to_response(profile)


@router.put("/config", response_model=PersonaConfigResponse)
async def update_persona_config(request: PersonaUpdateRequest) -> PersonaConfigResponse:
    """Update persona configuration settings.

    Args:
        request: Fields to update.
    """
    profile = _load_profile()

    # Apply only the fields that were provided
    update_data = request.model_dump(exclude_none=True)
    for field_name, value in update_data.items():
        if hasattr(profile, field_name):
            setattr(profile, field_name, value)

    # Save back to disk
    path = _get_profile_path()
    try:
        profile.to_toml(path)
    except Exception as e:
        logger.error("Failed to save persona profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save persona profile. Check server logs for details.",
        ) from e

    return _profile_to_response(profile)


def _collect_samples_sync(data_dir: Path) -> list[TextSample]:
    """Collect text samples from the data directory (blocking I/O).

    This runs in a thread to avoid blocking the event loop.

    Args:
        data_dir: Path to the persona data directory.

    Returns:
        List of text samples found on disk.
    """
    samples: list[TextSample] = []

    # Look for transcript files
    transcripts_dir = data_dir / "transcripts"
    if transcripts_dir.exists():
        for f in transcripts_dir.iterdir():
            if f.is_file() and f.suffix in (".txt", ".vtt", ".srt"):
                text = f.read_text(encoding="utf-8")
                if text.strip():
                    samples.append(TextSample(text=text, source_type="transcript"))

    # Look for email / writing files
    writing_dir = data_dir / "writing"
    if writing_dir.exists():
        for f in writing_dir.iterdir():
            if f.is_file() and f.suffix in (".txt", ".md"):
                text = f.read_text(encoding="utf-8")
                if text.strip():
                    samples.append(TextSample(text=text, source_type="writing"))

    return samples


@router.post("/rebuild")
async def rebuild_persona() -> dict[str, str]:
    """Trigger a persona rebuild from source data."""
    config = _get_persona_config()
    builder = PersonaBuilder(config=config)
    data_dir = Path(config.data_dir)

    # Collect samples in a thread to avoid blocking the event loop
    samples = await asyncio.to_thread(_collect_samples_sync, data_dir)

    if not samples:
        # If no samples found, just save a default profile
        profile = PersonaProfile()
        path = _get_profile_path()
        await asyncio.to_thread(profile.to_toml, path)
        return {"status": "ok", "message": "No source data found, saved default profile"}

    # Build the profile in a thread (CPU-intensive)
    profile = await asyncio.to_thread(builder.build_from_samples, samples)
    path = _get_profile_path()
    await asyncio.to_thread(profile.to_toml, path)

    return {
        "status": "ok",
        "message": f"Persona rebuilt from {len(samples)} source files",
    }


# --- Project Knowledge Endpoints ---


class IndexProjectRequest(BaseModel):
    """Request model for indexing a project directory."""

    path: str
    name: str | None = None


def _index_project_sync(project_path: str, name_override: str | None) -> dict:
    """Index a project directory (blocking I/O, runs in thread).

    Args:
        project_path: Absolute path to the project directory.
        name_override: Optional name override.

    Returns:
        Dict with index summary.
    """
    indexer = ProjectIndexer()
    store = KnowledgeStore()

    resolved = Path(project_path).resolve()
    index = indexer.index(resolved)

    if name_override:
        index.name = name_override

    store.save_index(index)

    return {
        "name": index.name,
        "root_path": index.root_path,
        "tech_stack": index.tech_stack,
        "dependencies_count": len(index.dependencies),
        "patterns": index.patterns,
        "total_files": index.total_files,
    }


@router.post("/index-project")
async def index_project(request: IndexProjectRequest) -> dict:
    """Index a project directory for technical context.

    Args:
        request: Contains the project path and optional name override.
    """
    project_path = Path(request.path)
    if not project_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {request.path}",
        )

    result = await asyncio.to_thread(
        _index_project_sync, request.path, request.name
    )
    return {"status": "ok", **result}


@router.get("/knowledge")
async def list_knowledge() -> dict:
    """List all indexed project knowledge."""
    store = KnowledgeStore()
    projects = store.list_projects()
    return {"projects": projects, "count": len(projects)}


@router.delete("/knowledge/{project_name}")
async def delete_knowledge(project_name: str) -> dict:
    """Delete indexed knowledge for a project.

    Args:
        project_name: Name of the project to delete.
    """
    store = KnowledgeStore()
    deleted = store.delete_index(project_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No knowledge found for project: {project_name}",
        )
    return {"status": "ok", "deleted": project_name}
