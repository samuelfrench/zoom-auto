"""Tests for the web API routes.

Tests cover:
- Health check endpoint
- Settings endpoint
- Persona routes (GET, PUT, POST)
- Meeting routes (POST join, POST leave, GET status)
- Dashboard routes (GET state)
- WebSocket dashboard connection
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from zoom_auto.config import PersonaConfig, Settings
from zoom_auto.persona.builder import PersonaProfile
from zoom_auto.web.app import create_app
from zoom_auto.web.routes import dashboard, meetings, persona

# --- Fixtures ---


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings()


@pytest.fixture
def app(settings: Settings) -> TestClient:
    """Create a FastAPI test client."""
    fastapi_app = create_app(settings=settings)
    return TestClient(fastapi_app)


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    """Create a temporary persona data directory."""
    d = tmp_path / "persona"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def reset_module_state() -> None:
    """Reset module-level state between tests."""
    persona._persona_config = None
    persona._persona_profile = None
    meetings._app_instance = None
    meetings._meeting_start_time = None
    dashboard._app_instance = None
    dashboard._clients.clear()
    dashboard._event_queue = None


# --- Health Check ---


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_ok(self, app: TestClient) -> None:
        """Health endpoint returns ok."""
        resp = app.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# --- Settings ---


class TestSettings:
    """Tests for the settings endpoint."""

    def test_get_settings(self, app: TestClient) -> None:
        """Settings endpoint returns config."""
        resp = app.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "zoom" in data
        assert "llm" in data
        assert "stt" in data
        assert "tts" in data
        assert "context" in data
        assert "response" in data
        assert "vad" in data
        assert data["llm"]["provider"] == "claude"

    def test_settings_contains_bot_name(self, app: TestClient) -> None:
        """Settings includes bot name."""
        resp = app.get("/api/settings")
        data = resp.json()
        assert "bot_name" in data["zoom"]


# --- Persona Routes ---


class TestPersonaRoutes:
    """Tests for persona configuration endpoints."""

    def test_get_config_default(self, app: TestClient) -> None:
        """GET /config returns default persona when no file exists."""
        resp = app.get("/api/persona/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "formality" in data
        assert "verbosity" in data
        assert "name" in data
        assert isinstance(data["formality"], float)

    def test_get_config_from_file(
        self, settings: Settings, persona_dir: Path
    ) -> None:
        """GET /config loads from TOML file."""
        # Write a profile
        profile = PersonaProfile(name="TestBot", formality=0.9)
        profile.to_toml(persona_dir / "profile.toml")

        # Configure persona to use this dir
        settings.persona = PersonaConfig(data_dir=str(persona_dir))
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        resp = client.get("/api/persona/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestBot"
        assert data["formality"] == 0.9

    def test_update_config(
        self, settings: Settings, persona_dir: Path
    ) -> None:
        """PUT /config updates persona fields."""
        # Write an initial profile
        profile = PersonaProfile(name="Original", formality=0.5)
        profile.to_toml(persona_dir / "profile.toml")

        settings.persona = PersonaConfig(data_dir=str(persona_dir))
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        resp = client.put(
            "/api/persona/config",
            json={"name": "Updated", "formality": 0.8},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated"
        assert data["formality"] == 0.8

        # Verify it persisted
        resp2 = client.get("/api/persona/config")
        assert resp2.json()["name"] == "Updated"

    def test_update_partial(
        self, settings: Settings, persona_dir: Path
    ) -> None:
        """PUT /config with partial fields only updates those fields."""
        profile = PersonaProfile(
            name="Partial", formality=0.5, verbosity=0.3
        )
        profile.to_toml(persona_dir / "profile.toml")

        settings.persona = PersonaConfig(data_dir=str(persona_dir))
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        # Only update formality
        resp = client.put(
            "/api/persona/config",
            json={"formality": 0.9},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["formality"] == 0.9
        assert data["name"] == "Partial"  # unchanged
        assert data["verbosity"] == 0.3  # unchanged

    def test_rebuild_no_sources(
        self, settings: Settings, persona_dir: Path
    ) -> None:
        """POST /rebuild with no source files saves default."""
        settings.persona = PersonaConfig(data_dir=str(persona_dir))
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        resp = client.post("/api/persona/rebuild")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "default" in data["message"].lower()

    def test_rebuild_with_sources(
        self, settings: Settings, persona_dir: Path
    ) -> None:
        """POST /rebuild with transcript files builds a profile."""
        # Create transcript sources
        transcripts = persona_dir / "transcripts"
        transcripts.mkdir()
        (transcripts / "meeting1.txt").write_text(
            "Alice: Let's discuss the deployment plan.\n"
            "Bob: I think we should deploy tomorrow.\n"
            "Alice: Sounds good, let me check the pipeline.\n"
        )

        settings.persona = PersonaConfig(data_dir=str(persona_dir))
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        resp = client.post("/api/persona/rebuild")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "1" in data["message"]


# --- Meeting Routes ---


class TestMeetingRoutes:
    """Tests for meeting endpoints."""

    def test_status_no_app(self, app: TestClient) -> None:
        """GET /status returns disconnected when no app."""
        resp = app.get("/api/meetings/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False

    def test_join_no_app(self, app: TestClient) -> None:
        """POST /join fails when no app instance."""
        resp = app.post(
            "/api/meetings/join",
            json={"meeting_id": "123456", "password": "pass"},
        )
        assert resp.status_code == 503

    def test_leave_no_app(self, app: TestClient) -> None:
        """POST /leave fails when no app instance."""
        resp = app.post("/api/meetings/leave")
        assert resp.status_code == 503

    def test_status_with_mock_app(self, settings: Settings) -> None:
        """GET /status with a mock app returns proper status."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = False
        mock_app.zoom_client.meeting_info = None

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.get("/api/meetings/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False

    def test_status_connected(self, settings: Settings) -> None:
        """GET /status when connected shows meeting info."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True
        mock_app.zoom_client.meeting_info.meeting_id = "987654"
        mock_app.context_manager.meeting_state.participants = {"Alice", "Bob"}
        mock_app.context_manager.transcript.entries = [MagicMock(), MagicMock()]

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        # Set meeting start time
        import time
        meetings._meeting_start_time = time.time() - 60

        resp = client.get("/api/meetings/status")
        data = resp.json()
        assert data["connected"] is True
        assert data["meeting_id"] == "987654"
        assert data["participants"] == 2
        assert data["duration_seconds"] >= 59
        assert data["utterances_count"] == 2

    def test_join_already_connected(self, settings: Settings) -> None:
        """POST /join fails when already connected."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/join",
            json={"meeting_id": "123"},
        )
        assert resp.status_code == 409

    def test_join_success(self, settings: Settings) -> None:
        """POST /join succeeds with mock app."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = False
        mock_app.join_meeting = AsyncMock()

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/join",
            json={"meeting_id": "123456", "password": "secret"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        mock_app.join_meeting.assert_called_once_with(
            meeting_id="123456", password="secret"
        )

    def test_leave_not_connected(self, settings: Settings) -> None:
        """POST /leave when not connected returns ok."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = False

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post("/api/meetings/leave")
        assert resp.status_code == 200
        data = resp.json()
        assert "not" in data["message"].lower()

    def test_leave_connected(self, settings: Settings) -> None:
        """POST /leave when connected calls zoom_client.leave."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True
        mock_app.zoom_client.leave = AsyncMock()

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post("/api/meetings/leave")
        assert resp.status_code == 200
        mock_app.zoom_client.leave.assert_called_once()


# --- Dashboard Routes ---


class TestDashboardRoutes:
    """Tests for dashboard endpoints."""

    def test_state_no_app(self, app: TestClient) -> None:
        """GET /state returns empty state when no app."""
        resp = app.get("/api/dashboard/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False
        assert data["transcript"] == []
        assert data["participants"] == []

    def test_state_with_mock_app(self, settings: Settings) -> None:
        """GET /state with mock app returns data."""
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True
        mock_app.zoom_client.meeting_info.meeting_id = "111222"
        mock_app.context_manager.meeting_state.participants = {"Alice"}
        mock_app.context_manager.meeting_state.decisions = ["Use React"]
        mock_app.context_manager.meeting_state.action_items = []
        mock_app.context_manager.transcript.entries = []

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.get("/api/dashboard/state")
        data = resp.json()
        assert data["connected"] is True
        assert data["meeting_id"] == "111222"
        assert "Alice" in data["participants"]

    def test_websocket_connect(self, settings: Settings) -> None:
        """WebSocket connects and receives initial state."""
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        with client.websocket_connect("/api/dashboard/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "state"
            assert "data" in data
            assert data["data"]["connected"] is False

    def test_websocket_ping_pong(self, settings: Settings) -> None:
        """WebSocket responds to ping with pong."""
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        with client.websocket_connect("/api/dashboard/ws") as ws:
            # Receive initial state
            ws.receive_json()
            # Send ping
            ws.send_text('{"type": "ping"}')
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_websocket_request_state(self, settings: Settings) -> None:
        """WebSocket responds to request_state."""
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        with client.websocket_connect("/api/dashboard/ws") as ws:
            # Receive initial state
            ws.receive_json()
            # Request state
            ws.send_text('{"type": "request_state"}')
            data = ws.receive_json()
            assert data["type"] == "state"


# --- CORS ---


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers(self, app: TestClient) -> None:
        """OPTIONS request should return CORS headers."""
        resp = app.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
