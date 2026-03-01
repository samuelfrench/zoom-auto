# Zoom Auto — Task Tracker

## Phase 1: Foundation
- [x] Initialize project + GitHub repo + core module scaffolding
- [ ] Configuration system (Pydantic Settings v2, TOML + .env loading)
- [ ] Zoom SDK integration (join/leave, audio capture, audio send)

## Phase 2: Audio Pipeline
- [ ] Silero VAD integration
- [ ] Faster Whisper STT (large-v3-turbo, int8)
- [ ] Chatterbox TTS (streaming synthesis with voice cloning)
- [ ] Audio pipeline routing (capture -> VAD -> STT -> LLM -> TTS -> send)

## Phase 3: Intelligence
- [ ] Claude LLM integration (Sonnet for responses, Haiku for decisions)
- [ ] Ollama local fallback
- [ ] Context manager (sliding window + summarization)
- [ ] Speaker tracking and diarization
- [ ] Meeting state tracking (agenda, action items, decisions)

## Phase 4: Persona System
- [ ] Persona builder (multi-source data aggregation)
- [ ] Communication style analyzer (NLP patterns)
- [ ] Transcript source analysis
- [ ] Slack export analysis
- [ ] Email/document analysis
- [ ] Test conversation analysis

## Phase 5: Response Engine
- [ ] Response trigger detection (should I speak?)
- [ ] Response generation (Claude + persona)
- [ ] Turn-taking manager (cooldowns, interruption avoidance)
- [ ] Full conversation loop integration

## Phase 6: Web Dashboard
- [ ] FastAPI server setup
- [ ] Voice sample upload/management endpoints
- [ ] Persona configuration endpoints
- [ ] Meeting join/monitor endpoints
- [ ] Live WebSocket dashboard

## Phase 7: Docker & Deployment
- [ ] Docker image build + test
- [ ] CUDA + Zoom SDK dependency validation
- [ ] End-to-end integration testing

## Completed
- 2026-03-01: Project initialization, repo structure, module scaffolding
