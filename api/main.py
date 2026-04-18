"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, cron, eval, heartbeat, sessions, workspace
from api.session_registry import registry
from cli.commands import _setup_logging, _run_log_file


@asynccontextmanager
async def lifespan(app: FastAPI):
    verbose = os.getenv("LOG_VERBOSE", "").lower() in ("1", "true")
    _setup_logging(verbose=verbose, log_file=_run_log_file("api"))
    registry.start_cleanup_task()
    yield
    # Gracefully stop cleanup task and close all active sessions on shutdown
    registry.stop_cleanup_task()
    for session_id in list(registry.list_sessions()):
        await registry.close(session_id)


app = FastAPI(
    title="CLI Async Agent API",
    description="HTTP interface for the CLI agent",
    version="0.1.0",
    lifespan=lifespan,
)

_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workspace.router, prefix="/workspace", tags=["workspace"])
app.include_router(chat.router, tags=["chat"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(cron.router, prefix="/cron", tags=["cron"])
app.include_router(heartbeat.router, tags=["heartbeat"])
app.include_router(eval.router, prefix="/eval", tags=["eval"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
