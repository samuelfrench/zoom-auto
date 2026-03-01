# Zoom Auto — Project Instructions

## Quick Start
- Read `TODO.md` at the start of each session for current priorities
- Read the global `~/CLAUDE.md` for memory system instructions

## Project Overview
- **Purpose:** AI-powered autonomous Zoom meeting participant
- **Language:** Python 3.11+
- **Package Manager:** uv
- **GPU:** NVIDIA RTX 4090 with CUDA 12
- **Key APIs:** Zoom Meeting SDK, Anthropic Claude, Ollama

## Tech Stack
- **STT:** Faster Whisper (large-v3-turbo, int8)
- **TTS:** Chatterbox Turbo (streaming voice cloning)
- **LLM:** Claude Sonnet (responses) + Haiku (decisions), Ollama fallback
- **VAD:** Silero VAD (via torch)
- **Web:** FastAPI + Uvicorn + WebSocket
- **Config:** Pydantic Settings v2 + TOML

## Key Directories
- `src/zoom_auto/` — Main Python package
- `config/` — TOML configuration files
- `data/` — Voice samples, transcripts (gitignored)
- `docker/` — Dockerfile + docker-compose with CUDA
- `tests/` — pytest test suite
- `scripts/` — Utility scripts

## Conventions
- Use Python `abc` module for abstract base classes
- Use Pydantic v2 models and settings
- Type hints required on all public functions
- Docstrings required on all classes and public methods
- Configuration via TOML files + environment variables
- Tests use pytest + pytest-asyncio

## Running
```bash
uv sync              # Install deps
uv run ruff check .  # Lint
uv run mypy src/     # Type check
uv run pytest        # Test
```
