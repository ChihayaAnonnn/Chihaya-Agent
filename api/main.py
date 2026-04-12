"""FastAPI application entry point."""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, cron, eval, heartbeat, workspace

app = FastAPI(
    title="CLI Async Agent API",
    description="HTTP interface for the CLI agent",
    version="0.1.0",
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
app.include_router(cron.router, prefix="/cron", tags=["cron"])
app.include_router(heartbeat.router, tags=["heartbeat"])
app.include_router(eval.router, prefix="/eval", tags=["eval"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
