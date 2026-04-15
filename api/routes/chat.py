"""
Chat route — stateless single-turn conversation (roleplay -m equivalent).

The persona model responds immediately. After the response is sent,
a background task runs the BackgroundAgent once to analyse the turn and
update MEMORY.md / USER.md if warranted.

For multi-turn conversations with persistent background agent use /sessions.
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from agent.background import BackgroundAgent, PromptHolder
from agent.loop import AgentLoop
from api.deps import get_api_key, get_workspace
from bus.events import ChatHistorySnapshot
from bus.queue import MessageBus
from providers.qwen import QwenProvider
from session.manager import SessionManager

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user: str = "default"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


async def _run_background_once(
    background_agent: BackgroundAgent,
    background_queue: asyncio.Queue[ChatHistorySnapshot],
) -> None:
    """Consume exactly one snapshot from the queue and run background analysis."""
    try:
        snapshot = await asyncio.wait_for(background_queue.get(), timeout=5.0)
        update = await background_agent._analyze(snapshot)
        if update:
            await background_agent.prompt_holder.write(update)
        await background_agent._maybe_consolidate(snapshot)
        logger.info("[CHAT] background analysis complete for session: %s", snapshot.session_key)
    except asyncio.TimeoutError:
        logger.debug("[CHAT] background queue empty, skipping analysis")
    except Exception as e:
        logger.error("[CHAT] background analysis error: %s", e)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
) -> ChatResponse:
    """
    Single-turn agent chat. Returns persona response immediately.
    Background memory analysis runs asynchronously after the response is sent.
    """
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
    background_agent = BackgroundAgent(
        queue=background_queue,
        provider=background_provider,
        prompt_holder=prompt_holder,
        workspace=workspace,
        session_manager=session_manager,
    )

    # Persona response (synchronous from the caller's perspective)
    response = await agent_loop.process_direct(
        content=req.message,
        session_key=session_id,
        channel="api",
        chat_id=session_id,
    )

    # Schedule background analysis to run after the HTTP response is sent
    background_tasks.add_task(_run_background_once, background_agent, background_queue)

    return ChatResponse(response=response, session_id=session_id)
