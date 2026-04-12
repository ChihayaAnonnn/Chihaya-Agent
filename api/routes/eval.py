"""Evaluation routes."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from api.deps import get_api_key
from providers.qwen import QwenProvider

router = APIRouter()

_EVAL_RESULTS_DIR = Path.cwd() / "eval" / "results"

# In-process task state: task_id -> {"status": str, "summary": dict|None, "error": str|None}
_eval_tasks: dict[str, dict] = {}


class EvalRunRequest(BaseModel):
    dataset: str = "longmemeval"
    entry_id: str | None = None
    filter_type: str | None = None
    single_session: bool = False
    limit: int | None = None
    judge_model: str = "qwen-turbo"
    keep_workspaces: bool = True


async def _run_eval_task(task_id: str, req: EvalRunRequest, api_key: str) -> None:
    from eval.dataset import load_longmemeval
    from eval.runner import run_batch, summarize

    try:
        entries = load_longmemeval(
            entry_id=req.entry_id,
            filter_type=req.filter_type,
            single_session_only=req.single_session,
            limit=req.limit,
        )
        if not entries:
            _eval_tasks[task_id] = {"status": "completed", "summary": {"total": 0}, "error": None}
            return

        provider = QwenProvider(api_key=api_key)
        eval_base = Path.cwd() / "eval" / ".workspaces"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_file = _EVAL_RESULTS_DIR / f"{req.dataset}_{ts}.jsonl"

        results = await run_batch(
            entries,
            provider,
            eval_workspace_base=eval_base,
            results_file=results_file,
            judge_model=req.judge_model,
            keep_workspace=req.keep_workspaces,
            verbose=False,
        )
        summary = summarize(results)
        _eval_tasks[task_id] = {
            "status": "completed",
            "summary": summary,
            "results_file": str(results_file),
            "error": None,
        }
    except Exception as e:
        _eval_tasks[task_id] = {"status": "failed", "summary": None, "error": str(e)}


@router.post("/run")
async def eval_run(
    req: EvalRunRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
) -> dict:
    """Start an evaluation run asynchronously. Poll GET /eval/tasks/{task_id} for status."""
    if req.dataset != "longmemeval":
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {req.dataset}")

    task_id = str(uuid4())
    _eval_tasks[task_id] = {"status": "running", "summary": None, "error": None}
    background_tasks.add_task(_run_eval_task, task_id, req, api_key)
    return {"task_id": task_id, "status": "running"}


@router.get("/tasks/{task_id}")
async def eval_task_status(task_id: str) -> dict:
    """Poll evaluation task progress."""
    if task_id not in _eval_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **_eval_tasks[task_id]}


@router.get("/results")
async def eval_results(latest: bool = Query(True)) -> dict:
    """List evaluation results."""
    if not _EVAL_RESULTS_DIR.exists():
        return {"results": []}

    files = sorted(_EVAL_RESULTS_DIR.glob("*.jsonl"))
    if not files:
        return {"results": []}

    from eval.runner import EvalResult, summarize

    targets = [files[-1]] if latest else files
    output = []
    for f in targets:
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        results = [EvalResult(**json.loads(line)) for line in lines if line.strip()]
        summary = summarize(results)
        output.append({"file": f.name, "summary": summary})

    return {"results": output}
