"""Live dashboard endpoints with WebSocket support.

Provides real-time meeting updates via WebSocket connection
for the web dashboard frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level reference to the app instance
_app_instance: object | None = None

# Connected WebSocket clients
_clients: list[WebSocket] = []

# Event queue for broadcasting
_event_queue: asyncio.Queue[dict] | None = None


def set_app_instance(app: object) -> None:
    """Set the ZoomAutoApp reference for dashboard data.

    Args:
        app: The ZoomAutoApp instance.
    """
    global _app_instance
    _app_instance = app


def get_event_queue() -> asyncio.Queue[dict]:
    """Get or create the event queue for broadcasting."""
    global _event_queue
    if _event_queue is None:
        _event_queue = asyncio.Queue()
    return _event_queue


async def broadcast_event(event: dict) -> None:
    """Broadcast an event to all connected WebSocket clients.

    Args:
        event: The event dict to broadcast.
    """
    disconnected: list[WebSocket] = []
    for ws in _clients:
        try:
            await ws.send_json(event)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in _clients:
            _clients.remove(ws)


class DashboardState(BaseModel):
    """Snapshot of the current dashboard state."""

    connected: bool = False
    meeting_id: str | None = None
    participants: list[str] = []
    duration_seconds: float = 0.0
    transcript: list[dict] = []
    decisions: list[str] = []
    action_items: list[str] = []
    bot_responses: list[dict] = []


@router.websocket("/ws")
async def dashboard_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for live dashboard updates.

    Streams real-time meeting data including:
    - Live transcript
    - Speaker activity
    - Bot response status
    - Meeting state (action items, decisions)
    """
    await websocket.accept()
    _clients.append(websocket)
    logger.info("Dashboard WebSocket client connected (total: %d)", len(_clients))

    try:
        # Send initial state
        state = _build_dashboard_state()
        await websocket.send_json({
            "type": "state",
            "data": state.model_dump(),
            "timestamp": time.time(),
        })

        # Listen for incoming messages and periodically push updates
        while True:
            try:
                # Wait for a message from the client (with timeout for periodic updates)
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=2.0
                )
                # Handle client messages
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "request_state":
                        state = _build_dashboard_state()
                        await websocket.send_json({
                            "type": "state",
                            "data": state.model_dump(),
                            "timestamp": time.time(),
                        })
                except json.JSONDecodeError:
                    pass
            except TimeoutError:
                # Timeout — send a periodic state update
                state = _build_dashboard_state()
                await websocket.send_json({
                    "type": "state_update",
                    "data": state.model_dump(),
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket client disconnected")
    except Exception:
        logger.debug("WebSocket connection error")
    finally:
        if websocket in _clients:
            _clients.remove(websocket)
        logger.info("WebSocket clients remaining: %d", len(_clients))


@router.get("/state")
async def get_dashboard_state() -> dict:
    """Get the current dashboard state as a snapshot.

    Returns the full current state for initial dashboard load.
    """
    state = _build_dashboard_state()
    return state.model_dump()


def _build_dashboard_state() -> DashboardState:
    """Build the current dashboard state from app components."""
    if _app_instance is None:
        return DashboardState()

    app = _app_instance
    connected = False
    meeting_id = None
    participants: list[str] = []
    transcript: list[dict] = []
    decisions: list[str] = []
    action_items: list[str] = []

    # Zoom connection
    if hasattr(app, "zoom_client"):
        connected = app.zoom_client.is_connected
        if connected and app.zoom_client.meeting_info:
            meeting_id = app.zoom_client.meeting_info.meeting_id

    # Context data
    if hasattr(app, "context_manager"):
        ctx = app.context_manager

        # Participants
        if hasattr(ctx, "meeting_state"):
            participants = list(ctx.meeting_state.participants)

            # Decisions and action items
            if hasattr(ctx.meeting_state, "decisions"):
                decisions = list(ctx.meeting_state.decisions)
            if hasattr(ctx.meeting_state, "action_items"):
                action_items = [
                    str(item) for item in ctx.meeting_state.action_items
                ]

        # Transcript entries (last 50)
        if hasattr(ctx, "transcript") and hasattr(ctx.transcript, "entries"):
            entries = ctx.transcript.entries[-50:]
            for entry in entries:
                transcript.append({
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": entry.timestamp.isoformat()
                    if hasattr(entry, "timestamp") and entry.timestamp
                    else "",
                })

    # Duration
    from zoom_auto.web.routes.meetings import _meeting_start_time

    duration = 0.0
    if connected and _meeting_start_time is not None:
        duration = round(time.time() - _meeting_start_time, 1)

    return DashboardState(
        connected=connected,
        meeting_id=meeting_id,
        participants=participants,
        duration_seconds=duration,
        transcript=transcript,
        decisions=decisions,
        action_items=action_items,
    )
