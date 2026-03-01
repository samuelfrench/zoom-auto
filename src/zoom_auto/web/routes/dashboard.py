"""Live dashboard endpoints with WebSocket support.

Provides real-time meeting updates via WebSocket connection
for the web dashboard frontend.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

router = APIRouter()


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
    try:
        while True:
            # Placeholder: will send real-time updates
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ping", "data": data})
    except Exception:
        pass
    finally:
        await websocket.close()


@router.get("/state")
async def get_dashboard_state() -> dict:
    """Get the current dashboard state as a snapshot.

    Returns the full current state for initial dashboard load.
    """
    raise NotImplementedError("Dashboard state not yet implemented")
