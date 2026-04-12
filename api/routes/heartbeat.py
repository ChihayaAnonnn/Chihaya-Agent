"""Heartbeat route."""

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.factory import make_agentic_runner
from api.deps import get_api_key, get_workspace
from heartbeat.service import HeartbeatService

router = APIRouter()


@router.post("/heartbeat")
async def heartbeat(
    user: str = Query("default"),
    api_key: str = Depends(get_api_key),
) -> dict:
    """Manually trigger a heartbeat check."""
    workspace = get_workspace(user)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Workspace for user '{user}' not found. Call POST /workspace/init first.")

    run_agent = make_agentic_runner(workspace, api_key)
    service = HeartbeatService(
        workspace=workspace,
        on_heartbeat=lambda prompt: run_agent(prompt, session_key="heartbeat"),
    )
    result = await service.trigger_now()
    return {"result": result or ""}
