# Zoom Auto

AI-powered autonomous Zoom meeting participant that listens, understands context, and responds with a cloned voice — acting as a realistic stand-in during meetings.

## Features

- **Real-time Speech-to-Text** — Faster Whisper (large-v3-turbo, int8) for low-latency transcription
- **Voice Cloning TTS** — Chatterbox Turbo streaming synthesis using your voice samples
- **AI Response Generation** — Claude Sonnet for responses, Haiku for quick decisions
- **Persona System** — Build a communication persona from transcripts, Slack exports, emails, and test conversations
- **Context Management** — Sliding window + summarization for full meeting awareness
- **Smart Turn-Taking** — Trigger detection, cooldowns, and interruption avoidance
- **Speaker Tracking** — Per-speaker audio capture and diarization
- **Voice Activity Detection** — Silero VAD for precise speech boundary detection
- **Web Dashboard** — FastAPI server with live WebSocket dashboard
- **Local Fallback** — Ollama support for offline/private operation

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Zoom SDK   │───>│ Audio Capture │───>│  STT        │
│  (Meeting)  │    │ (per-speaker) │    │ (Whisper)   │
└─────────────┘    └──────────────┘    └──────┬──────┘
       ▲                                       │
       │                                       ▼
┌──────┴──────┐    ┌──────────────┐    ┌─────────────┐
│ Audio Send  │<───│  TTS         │<───│  LLM        │
│ (to Zoom)   │    │ (Chatterbox) │    │ (Claude)    │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                               │
                          ┌────────────────────┤
                          ▼                    ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Persona    │    │  Context     │
                   │  System     │    │  Manager     │
                   └─────────────┘    └─────────────┘
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12+ (for Whisper, Chatterbox, Silero)
- Zoom Meeting SDK credentials
- Anthropic API key (for Claude)
- Docker + NVIDIA Container Toolkit (recommended)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/samuelfrench/zoom-auto.git
cd zoom-auto
```

### 2. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv sync
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 4. Configure Settings

Edit `config/default.toml` for default settings, or create `config/local.toml` for overrides.

### 5. Prepare Voice Samples

Place voice sample WAV files in `data/voice_samples/`:

- **Format:** WAV, 16-bit, 22050 Hz mono (or 44100 Hz — will be resampled)
- **Duration:** 10–30 seconds per sample, 2–5 samples recommended
- **Content:** Clear speech with minimal background noise
- **Tip:** Record yourself reading a paragraph naturally — avoid monotone

### 6. Run

```bash
# Direct
uv run python -m zoom_auto

# Or with Docker
docker compose -f docker/docker-compose.yml up
```

### 7. Access Dashboard

Open `http://localhost:8080` for the live meeting dashboard.

## Docker

```bash
# Build
docker compose -f docker/docker-compose.yml build

# Run
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=zoom_auto
```

## Project Structure

```
zoom-auto/
├── src/zoom_auto/          # Main package
│   ├── zoom/               # Zoom SDK integration
│   ├── stt/                # Speech-to-text (Faster Whisper)
│   ├── tts/                # Text-to-speech (Chatterbox)
│   ├── llm/                # LLM providers (Claude, Ollama)
│   ├── persona/            # Persona building & analysis
│   ├── context/            # Meeting context management
│   ├── response/           # Response decision & generation
│   ├── pipeline/           # Audio pipeline & conversation loop
│   └── web/                # FastAPI dashboard & API
├── config/                 # TOML configuration files
├── data/                   # Voice samples, transcripts (gitignored)
├── docker/                 # Docker setup with CUDA
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

## License

MIT License - Copyright 2026 Samuel French
