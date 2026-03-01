# Zoom Auto

AI-powered autonomous Zoom meeting participant that listens, understands context, and responds in your cloned voice — acting as a realistic stand-in during meetings.

The bot joins Zoom meetings via the Meeting SDK, transcribes conversation in real-time with Faster Whisper, uses Claude to decide when and how to respond (matching your communication style through a trained persona), and speaks back using Chatterbox TTS voice cloning. It learns from every conversation and can index your project codebases for technical context.

## Features

- **Real-time Speech-to-Text** — Faster Whisper large-v3-turbo (int8) for low-latency transcription
- **Voice Cloning** — Chatterbox Turbo zero-shot cloning from your voice samples (~470ms first-chunk on RTX 4090)
- **AI Responses** — Claude Sonnet for natural responses, Haiku for split-second "should I speak?" decisions
- **Persona System** — Learns your communication style from transcripts, Slack exports, emails, and live conversations
- **Conversation Learning** — Accumulates vocabulary, topics, and meeting patterns across sessions
- **Project Knowledge** — Index local codebases so the bot can discuss your technical work intelligently
- **Smart Turn-Taking** — Trigger detection, configurable cooldowns, interruption handling
- **Context Management** — Sliding window + rolling summarization for full meeting awareness
- **Chat Disclaimer** — Automatically posts a transparency notice in Zoom chat when joining
- **Speaker Tracking** — Per-speaker audio capture and attribution
- **Voice Activity Detection** — Silero VAD for precise speech boundary detection
- **Web Dashboard** — Live WebSocket dashboard with transcript, decision log, and controls
- **Easy Meeting Join** — Join via URL, meeting ID, CLI args, or environment variables
- **Demo Mode** — Full pipeline testing with live microphone or WAV files (no Zoom SDK required)
- **Local Fallback** — Ollama support for fully offline/private operation
- **Docker Ready** — Multi-stage build with NVIDIA CUDA support

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────────┐
│  Zoom SDK   │───>│ Audio Capture │───>│  Silero  │───>│    STT      │
│  (Meeting)  │    │ (per-speaker) │    │   VAD    │    │ (Whisper)   │
└──────┬──────┘    └──────────────┘    └──────────┘    └──────┬──────┘
       │                                                       │
       │  ┌──────────┐                                         ▼
       │  │   Chat   │    ┌──────────────┐             ┌─────────────┐
       ├──│  Sender  │    │  TTS         │<────────────│    LLM      │
       │  │(disclaimr)│    │ (Chatterbox) │             │  (Claude)   │
       │  └──────────┘    └──────┬───────┘             └──────┬──────┘
       │                         │                             │
       │  ┌──────────┐          │               ┌─────────────┤
       │  │  Audio   │<─────────┘               │             │
       └──│  Sender  │                   ┌──────┴─────┐ ┌─────┴──────┐
          │ (to Zoom)│                   │  Persona   │ │  Context   │
          └──────────┘                   │  + Learner │ │  Manager   │
                                         └────────────┘ └────────────┘
```

**Audio flow:** Zoom SDK delivers 32 kHz PCM per speaker -> resample to 16 kHz -> VAD detects utterance boundaries -> Whisper transcribes -> Context Manager adds to sliding window -> Trigger Detector decides if bot should respond -> Claude generates persona-matched response -> Chatterbox clones your voice -> resample to 32 kHz -> send back to Zoom.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12+ (RTX 4090 recommended — runs Whisper + Chatterbox + Silero simultaneously)
- Zoom Meeting SDK credentials ([marketplace.zoom.us](https://marketplace.zoom.us))
- Anthropic API key
- ~10 GB VRAM (Whisper ~6 GB + Chatterbox ~3 GB + VAD ~0.5 GB)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/samuelfrench/zoom-auto.git && cd zoom-auto
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2. Configure Credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- `ZOOM_AUTO_ZOOM__MEETING_SDK_KEY` — from your Meeting SDK app on [marketplace.zoom.us](https://marketplace.zoom.us)
- `ZOOM_AUTO_ZOOM__MEETING_SDK_SECRET` — Meeting SDK app secret
- `ZOOM_AUTO_LLM__ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)

### 3. Record Voice Samples

```bash
uv run python scripts/record_voice_samples.py
```

This presents 20 scripted prompts covering greetings, technical statements, casual chat, and emotional range. Recordings are saved to `data/voice_samples/`. You need at least 10 seconds of audio, but 2-5 minutes is recommended for best quality.

You can also upload existing audio files through the web dashboard at `http://localhost:8080`.

### 4. Join a Meeting

```bash
# Join by meeting URL
uv run python -m zoom_auto join "https://zoom.us/j/1234567890?pwd=abc123"

# Join by meeting ID
uv run python -m zoom_auto join 1234567890

# Join with password
uv run python -m zoom_auto join 1234567890 --password abc123

# Auto-join via environment variables
export ZOOM_AUTO_MEETING_ID=1234567890
export ZOOM_AUTO_MEETING_PASSWORD=abc123
uv run python -m zoom_auto start
```

## CLI Commands

### `start` — Start the bot (optionally auto-join a meeting)

```bash
uv run python -m zoom_auto start
```

Starts the web server on port 8080 and waits for a meeting join command via the dashboard. If `ZOOM_AUTO_MEETING_ID` is set, auto-joins on startup.

### `join` — Join a specific meeting

```bash
uv run python -m zoom_auto join <meeting-url-or-id> [--password PWD] [--name "Display Name"]
```

Accepts Zoom URLs (`https://zoom.us/j/123?pwd=abc`), plain meeting IDs, or IDs with spaces/dashes.

### `index` — Index project directories for technical context

```bash
# Index a single project
uv run python -m zoom_auto index ~/projects/my-app

# Index multiple projects
uv run python -m zoom_auto index ~/projects/api ~/projects/frontend ~/projects/infra
```

Scans directories for tech stack, dependencies, README content, key files, and code patterns. This knowledge is injected into the LLM context so the bot can discuss your technical work intelligently in meetings.

### `learnings` — View accumulated conversation learnings

```bash
# Show all learnings
uv run python -m zoom_auto learnings --user sam

# Show summary only
uv run python -m zoom_auto learnings --user sam --summary

# Rebuild persona from learnings
uv run python -m zoom_auto learnings --user sam --rebuild-persona
```

## Demo Mode (No Zoom SDK Required)

Test the full pipeline with your microphone or a WAV file — no Zoom account needed:

```bash
# Live mic — speak and hear the bot respond in your voice
uv run python scripts/demo_pipeline.py --user sam

# Process a recorded meeting
uv run python scripts/demo_pipeline.py --input meeting.wav --user sam

# With persona + project knowledge
uv run python scripts/demo_pipeline.py --user sam --persona config/personas/sam.toml --projects ~/myproject

# Text-only (skip TTS output)
uv run python scripts/demo_pipeline.py --user sam --no-tts

# List available audio devices
uv run python scripts/demo_pipeline.py --list-devices
```

The demo pipeline wires together STT, VAD, LLM, TTS, context management, trigger detection, response generation, and the conversation learner. Learnings persist across sessions.

## Conversation Learning

The bot learns from every conversation it participates in:

- **Vocabulary tracking** — Builds a frequency map of domain-specific terms used in meetings
- **Topic extraction** — Identifies discussion topics from explicit markers and repeated technical terms
- **Meeting type detection** — Classifies meetings (standup, planning, retro, technical, brainstorm, 1:1, general)
- **Session persistence** — Saves transcripts and learnings to `data/learnings/{user}/`
- **Persona feedback** — Learning context is fed back into the LLM system prompt so responses improve over time

View learnings:

```bash
uv run python -m zoom_auto learnings --user sam --summary
```

## Persona System

The persona system captures your unique communication style so the bot sounds like you, not a generic AI.

### Building a Persona

The bot can build your persona from multiple sources:

- **Meeting transcripts** — Past recordings of you speaking (weighted highest)
- **Slack exports** — Your messaging patterns and vocabulary
- **Email/documents** — Writing samples for formality and technical depth
- **Test conversations** — Interactive sessions to capture your natural responses
- **Live learning** — Accumulated patterns from meetings the bot attends

### Persona Profile

Generated profiles are stored in `config/personas/{user}.toml` and include:

- `formality` / `verbosity` / `technical_depth` / `assertiveness` (0.0-1.0 scales)
- `filler_words` / `common_phrases` / `greeting_style` / `agreement_style`
- `avg_response_words` / `preferred_terms` / `avoided_terms`
- `standup_format` — how you structure standup updates

Edit the TOML file to fine-tune any trait.

## Project Knowledge Indexing

Index your codebases so the bot has technical context during meetings:

```bash
uv run python -m zoom_auto index ~/projects/api ~/projects/frontend
```

The indexer automatically detects:
- **Tech stack** — Languages, frameworks, tools (from file extensions, config files, imports)
- **Dependencies** — Parsed from `pyproject.toml`, `package.json`, `requirements.txt`, `Cargo.toml`, `go.mod`
- **Key files** — README, configs, entry points
- **Code patterns** — Common patterns and conventions

Knowledge is stored in `data/knowledge/` and injected into the LLM context during meetings.

## Web Dashboard

```bash
uv run python -m zoom_auto start
# Open http://localhost:8080
```

The dashboard provides:

- **Voice Setup** — Record voice samples, upload files, review quality, combine into reference audio
- **Persona Config** — View and edit persona traits, rebuild from sources
- **Meeting Dashboard** — Live transcript via WebSocket, bot decision log, join/leave controls, send chat messages
- **Settings** — View current configuration

The API is also available at `http://localhost:8080/api/` for programmatic access.

## Chat Disclaimer

When joining a meeting, the bot automatically sends a transparency message in the Zoom meeting chat:

> AI assistant is participating in this meeting on behalf of the host. Audio is processed locally.

Configure in `config/default.toml`:

```toml
[zoom]
disclaimer_message = "AI assistant is participating in this meeting on behalf of the host. Audio is processed locally."
send_disclaimer = true
```

## Configuration

### Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `ZOOM_AUTO_ZOOM__MEETING_SDK_KEY` | Zoom Meeting SDK app key |
| `ZOOM_AUTO_ZOOM__MEETING_SDK_SECRET` | Zoom Meeting SDK app secret |
| `ZOOM_AUTO_LLM__ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `ZOOM_AUTO_LLM__OLLAMA_HOST` | Ollama host URL (default: `http://localhost:11434`) |
| `ZOOM_AUTO_MEETING_ID` | Auto-join this meeting on startup |
| `ZOOM_AUTO_MEETING_PASSWORD` | Password for auto-join meeting |

### Settings (`config/default.toml`)

Create `config/local.toml` to override defaults:

```toml
[zoom]
bot_name = "Sam's Assistant"
sample_rate = 16000
send_disclaimer = true

[stt]
model = "large-v3-turbo"
compute_type = "int8"
language = "en"

[tts]
voice_sample_dir = "data/voice_samples"
sample_rate = 22050

[llm]
provider = "claude"          # or "ollama" for offline mode
response_model = "claude-sonnet-4-20250514"
decision_model = "claude-haiku-4-20250414"
max_tokens = 300

[response]
cooldown_seconds = 10.0      # minimum seconds between bot responses
trigger_threshold = 0.6      # confidence threshold to speak (0-1)
max_consecutive = 3          # max responses before forced cooldown

[vad]
threshold = 0.5
min_speech_duration = 0.25
min_silence_duration = 0.3
```

## Zoom SDK Setup

1. Go to [marketplace.zoom.us](https://marketplace.zoom.us) and sign in
2. Click **Develop** > **Build App**
3. Select **Meeting SDK** app type
4. Note your **SDK Key** (Client ID) and **SDK Secret** (Client Secret)
5. Add them to your `.env` file

A free Zoom account works for development. One-on-one meetings (you + the bot) have no time limit, which is ideal for testing and voice sample collection.

## Docker

### Build and Run

```bash
# Build
docker compose -f docker/docker-compose.yml build

# Run
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f

# Or use the convenience script
./scripts/docker-run.sh
```

### Requirements

- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
- NVIDIA GPU with CUDA 12.4+ drivers

The Docker image uses a multi-stage build: Node.js for the frontend, then NVIDIA CUDA 12.4 + Ubuntu 22.04 for the runtime. GPU passthrough is configured via `docker-compose.yml` with `compute,utility,video` capabilities.

### Volumes

| Mount | Purpose |
|-------|---------|
| `./data` | Voice samples, transcripts, learnings, knowledge |
| `./config` | TOML configuration files |
| `./.env` | Environment variables |

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=zoom_auto

# Run a specific test file
uv run pytest tests/test_learner.py -v
```

### Test Suite

The project has 600+ tests covering all modules. Tests run without the Zoom SDK installed — all SDK calls use lazy imports and are mocked in tests.

```bash
# Run all tests
uv run pytest

# Run by module
uv run pytest tests/test_pipeline.py      # Audio pipeline + conversation loop
uv run pytest tests/test_web.py            # FastAPI routes + WebSocket
uv run pytest tests/test_learner.py        # Conversation learning
uv run pytest tests/test_project_indexer.py # Project indexing
uv run pytest tests/test_chat.py           # Chat sender + disclaimer
uv run pytest tests/test_url_parser.py     # Meeting URL parsing
```

## Project Structure

```
zoom-auto/
├── src/zoom_auto/
│   ├── main.py                  # Entry point, CLI, orchestrator
│   ├── config.py                # Pydantic settings (TOML + .env)
│   ├── zoom/
│   │   ├── client.py            # SDK lifecycle (join/leave/auth)
│   │   ├── audio_capture.py     # Per-speaker PCM audio capture
│   │   ├── audio_sender.py      # Send TTS audio back to meeting
│   │   ├── chat_sender.py       # Meeting chat messages + disclaimer
│   │   ├── url_parser.py        # Parse meeting URLs and IDs
│   │   └── events.py            # Meeting event handlers
│   ├── stt/
│   │   ├── base.py              # Abstract STT interface
│   │   └── faster_whisper.py    # Faster Whisper (large-v3-turbo)
│   ├── tts/
│   │   ├── base.py              # Abstract TTS interface
│   │   ├── chatterbox.py        # Chatterbox Turbo voice cloning
│   │   └── voice_store.py       # Voice sample management
│   ├── llm/
│   │   ├── base.py              # Abstract LLM interface
│   │   ├── claude.py            # Claude Sonnet + Haiku
│   │   └── ollama.py            # Local Ollama fallback
│   ├── persona/
│   │   ├── builder.py           # Multi-source persona builder
│   │   ├── style_analyzer.py    # NLP communication patterns
│   │   ├── learner.py           # Real-time conversation learning
│   │   ├── knowledge_store.py   # Project knowledge persistence
│   │   └── sources/             # Transcript, Slack, email, project analyzers
│   ├── context/
│   │   ├── manager.py           # Sliding window + summarization
│   │   ├── transcript.py        # Live transcript accumulator
│   │   ├── speaker_tracker.py   # Speaker attribution
│   │   └── meeting_state.py     # Agenda, action items, decisions
│   ├── response/
│   │   ├── decision.py          # Trigger detection (should I speak?)
│   │   ├── generator.py         # Response generation (Claude + persona)
│   │   └── turn_manager.py      # Turn-taking + cooldowns
│   ├── pipeline/
│   │   ├── audio_pipeline.py    # Audio frame routing
│   │   ├── conversation.py      # Full real-time conversation loop
│   │   └── vad.py               # Silero VAD integration
│   └── web/
│       ├── app.py               # FastAPI server
│       └── routes/              # Voice, persona, meetings, dashboard
├── frontend/                    # React + Vite + TypeScript SPA
├── config/                      # TOML configuration
├── data/                        # Voice samples, learnings, knowledge (gitignored)
├── docker/                      # Dockerfile + docker-compose.yml
├── scripts/                     # CLI tools + demo pipeline
└── tests/                       # 600+ tests
```

## Estimated Costs

| Service | Cost | Notes |
|---------|------|-------|
| Claude API | ~$0.50-2.00/hr of meeting | Sonnet for responses, Haiku for decisions |
| Zoom | $0 | Works with free account |
| GPU | $0 | Local inference (Whisper, Chatterbox, VAD) |

## Security

- All secrets in `.env` (gitignored)
- Voice samples in `data/` (gitignored)
- Web UI bound to localhost only
- Option for fully local LLM via Ollama (no meeting content sent to cloud)
- Automatic chat disclaimer for transparency
- Transcripts configurable for auto-purge

## License

MIT License - Copyright 2026 Samuel French
