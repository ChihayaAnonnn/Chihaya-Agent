"""Chat route — single-turn conversation (roleplay -m equivalent)."""

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from agent.background import BackgroundAgent, PromptHolder
from agent.loop import AgentLoop
from agent.factory import make_agentic_runner
from api.deps import get_api_key, get_workspace
from bus.events import ChatHistorySnapshot
from bus.queue import MessageBus
from providers.qwen import QwenProvider
from session.manager import SessionManager

import asyncio

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user: str = "default"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    api_key: str = Depends(get_api_key),
) -> ChatResponse:
    """Single-turn agent chat."""
    workspace = get_workspace(req.user)
    session_id = req.session_id or f"api:{req.user}"

    background_queue: asyncio.Queue[ChatHistorySnapshot] = asyncio.Queue()
    prompt_holder = PromptHolder()
    bus = MessageBus()
    persona_provider = QwenProvider(api_key=api_key)
    background_provider = QwenProvider(api_key=api_key)
    session_manager = SessionManager(workspace)

    agent_loop = AgentLoop(
        bus=bus,
        provider=persona_provider,
        workspace=workspace,
        session_manager=session_manager,
        persona_provider=persona_provider,
        background_queue=background_queue,
        prompt_holder=prompt_holder,
        ephemeral=False,
    )

    response = await agent_loop.process_direct(
        content=req.message,
        session_key=session_id,
        channel="api",
        chat_id=session_id,
    )

    return ChatResponse(response=response, session_id=session_id)
