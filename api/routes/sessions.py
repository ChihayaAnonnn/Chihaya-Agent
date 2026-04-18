"""
Session management routes — full dual-LLM mode (persona + background agent).

Lifecycle:
  POST   /sessions                        — create & start a session
  POST   /sessions/{session_id}/messages  — send a message, get response
  GET    /sessions/{session_id}/events    — SSE stream of background agent events
  GET    /sessions                        — list active sessions
  DELETE /sessions/{session_id}           — close & stop a session
"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.deps import get_api_key, get_workspace
from api.session_registry import registry

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    user: str = "default"
    session_id: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    workspace: str


class MessageRequest(BaseModel):
    message: str
    timeout: float = 60.0


class MessageResponse(BaseModel):
    session_id: str
    response: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    req: SessionCreateRequest,
    api_key: str = Depends(get_api_key),
) -> SessionCreateResponse:
    """
    Start a persistent dual-LLM session (AgentLoop + BackgroundAgent).
    The background agent will analyze each turn, update MEMORY.md / USER.md,
    and feed ephemeral hints back to the persona model.
    """
    session_id = req.session_id or f"api:{req.user}"
    if registry.exists(session_id):
        raise HTTPException(
            status_code=409,
            detail=f"Session '{session_id}' already active. Send DELETE /sessions/{session_id} to close it first.",
        )

    workspace = get_workspace(req.user)
    registry.create(session_id=session_id, workspace=workspace, api_key=api_key)
    return SessionCreateResponse(session_id=session_id, workspace=str(workspace))


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(
    session_id: str,
    req: MessageRequest,
    api_key: str = Depends(get_api_key),
) -> MessageResponse:
    """
    Send a message to an active session and wait for the persona response.
    The background agent processes the turn asynchronously in parallel.
    """
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Create it first via POST /sessions.",
        )

    try:
        response = await session.send(req.message, timeout=req.timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent response timed out.")

    return MessageResponse(session_id=session_id, response=response)


@router.post("/sessions/{session_id}/keepalive")
async def session_keepalive(session_id: str) -> dict:
    """
    Reset the TTL clock for a session.
    Call periodically (e.g. every 5 minutes) to prevent automatic expiry.
    Returns the remaining TTL in seconds based on SESSION_TTL env var (default 1800s).
    """
    import os
    if not registry.touch(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    ttl = int(os.getenv("SESSION_TTL", "1800"))
    return {"session_id": session_id, "alive": True, "ttl_s": ttl}


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str, request: Request):
    """
    SSE stream of background agent activity for a session.

    Each event is a JSON object with a `type` field:
      - start       — background analysis begins for a user turn
      - tool_call   — a tool is being called (detail: tool name + arg)
      - tools_used  — summary of tools used in this analysis
      - hint        — ephemeral hint generated for the persona
      - done        — analysis complete
      - consolidate — memory consolidation triggered
      - keepalive   — heartbeat sent every 15s when queue is idle

    Connect with:
      const es = new EventSource('/sessions/{id}/events');
      es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    log_queue = session.log_queue

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(log_queue.get(), timeout=15.0)
                yield {"data": json.dumps(event, ensure_ascii=False)}
            except asyncio.TimeoutError:
                # Send keepalive to prevent connection timeout
                yield {"data": json.dumps({"type": "keepalive"})}

    return EventSourceResponse(event_generator())


@router.get("/sessions")
async def list_sessions() -> dict:
    """List all currently active sessions."""
    return {"sessions": registry.list_sessions()}


@router.delete("/sessions/{session_id}")
async def close_session(session_id: str) -> dict:
    """
    Stop the AgentLoop and BackgroundAgent for this session.
    Session history is persisted to the workspace automatically.
    """
    closed = await registry.close(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"closed": session_id}
