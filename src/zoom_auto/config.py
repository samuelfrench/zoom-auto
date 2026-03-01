"""Configuration management using Pydantic Settings v2.

Loads configuration from TOML files and environment variables.
Priority: environment variables > local.toml > default.toml
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080


class ZoomConfig(BaseModel):
    """Zoom SDK configuration."""

    bot_name: str = "AI Assistant"
    sample_rate: int = 16000
    channels: int = 1


class STTConfig(BaseModel):
    """Speech-to-text configuration."""

    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    voice_sample_dir: str = "data/voice_samples"
    sample_rate: int = 22050
    chunk_size: int = 4096


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "claude"
    response_model: str = "claude-sonnet-4-20250514"
    decision_model: str = "claude-haiku-4-20250414"
    ollama_model: str = "llama3.1:8b"
    max_tokens: int = 300
    temperature: float = 0.7


class PersonaConfig(BaseModel):
    """Persona system configuration."""

    data_dir: str = "data/persona"
    style_weight: float = 0.7


class ContextConfig(BaseModel):
    """Context management configuration."""

    max_window_tokens: int = 4000
    summarize_at: int = 3000
    max_action_items: int = 20


class ResponseConfig(BaseModel):
    """Response engine configuration."""

    cooldown_seconds: float = 10.0
    trigger_threshold: float = 0.6
    max_consecutive: int = 3


class VADConfig(BaseModel):
    """Voice activity detection configuration."""

    threshold: float = 0.5
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.3


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


class Settings(BaseSettings):
    """Application settings loaded from TOML config and environment variables.

    Load priority:
    1. Environment variables (highest)
    2. config/local.toml
    3. config/default.toml (lowest)
    """

    model_config = SettingsConfigDict(
        env_prefix="ZOOM_AUTO_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # API keys from environment
    zoom_client_id: str = ""
    zoom_client_secret: str = ""
    zoom_meeting_sdk_key: str = ""
    zoom_meeting_sdk_secret: str = ""
    anthropic_api_key: str = ""
    ollama_host: str = "http://localhost:11434"

    # Nested config sections
    server: ServerConfig = ServerConfig()
    zoom: ZoomConfig = ZoomConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    llm: LLMConfig = LLMConfig()
    persona: PersonaConfig = PersonaConfig()
    context: ContextConfig = ContextConfig()
    response: ResponseConfig = ResponseConfig()
    vad: VADConfig = VADConfig()

    @classmethod
    def from_toml(cls, config_dir: str | Path = "config") -> Settings:
        """Load settings from TOML files in the given config directory."""
        config_path = Path(config_dir)
        defaults = _load_toml(config_path / "default.toml")
        local = _load_toml(config_path / "local.toml")

        # Merge: local overrides defaults
        merged: dict[str, Any] = {}
        for key in set(list(defaults.keys()) + list(local.keys())):
            if key in local and key in defaults:
                if isinstance(defaults[key], dict) and isinstance(local[key], dict):
                    merged[key] = {**defaults[key], **local[key]}
                else:
                    merged[key] = local[key]
            elif key in local:
                merged[key] = local[key]
            else:
                merged[key] = defaults[key]

        return cls(**merged)
