"""Cron job management routes."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.deps import get_api_key, get_workspace
from agent.factory import make_agentic_runner
from cron.service import CronService
from cron.types import CronSchedule

router = APIRouter()


def _cron_store_path() -> Path:
    return Path.cwd() / "workspaces" / ".agent" / "cron" / "jobs.json"


class CronAddRequest(BaseModel):
    name: str
    message: str
    every: int | None = None
    at: str | None = None


@router.get("/jobs")
async def cron_list(include_disabled: bool = Query(False)) -> dict:
    """List scheduled jobs."""
    store_path = _cron_store_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    service = CronService(store_path)
    jobs = service.list_jobs(include_disabled=include_disabled)
    result = []
    for j in jobs:
        sched = (
            f"every {(j.schedule.every_ms or 0) // 1000}s"
            if j.schedule.kind == "every"
            else "one-time"
        )
        result.append({"id": j.id, "name": j.name, "schedule": sched, "enabled": j.enabled})
    return {"jobs": result}


@router.post("/jobs")
async def cron_add(req: CronAddRequest) -> dict:
    """Add a scheduled job."""
    if req.every:
        schedule = CronSchedule(kind="every", every_ms=req.every * 1000)
    elif req.at:
        import datetime
        dt = datetime.datetime.fromisoformat(req.at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        raise HTTPException(status_code=400, detail="Provide 'every' (seconds) or 'at' (ISO datetime)")

    store_path = _cron_store_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    service = CronService(store_path)
    job = service.add_job(name=req.name, schedule=schedule, message=req.message)
    return {"id": job.id, "name": job.name}


@router.delete("/jobs/{job_id}")
async def cron_remove(job_id: str) -> dict:
    """Remove a scheduled job."""
    service = CronService(_cron_store_path())
    if not service.remove_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"removed": job_id}


@router.post("/jobs/{job_id}/run")
async def cron_run(
    job_id: str,
    force: bool = Query(False),
    user: str = Query("default"),
    api_key: str = Depends(get_api_key),
) -> dict:
    """Manually execute a scheduled job."""
    workspace = get_workspace(user)
    store_path = _cron_store_path()
    run_agent = make_agentic_runner(workspace, api_key)
    service = CronService(store_path)
    result_holder: list[str] = []

    async def on_job(job) -> str | None:
        r = await run_agent(job.payload.message, session_key=f"cron:{job.id}")
        result_holder.append(r)
        return r

    service.on_job = on_job
    ok = await service.run_job(job_id, force=force)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not due")
    return {"result": result_holder[0] if result_holder else ""}
