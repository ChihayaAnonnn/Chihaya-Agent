"""
Global registry of active agent sessions.

Each session runs AgentLoop (persona lane) + BackgroundAgent (memory/tool lane)
as persistent asyncio tasks, mirroring the CLI `roleplay` interactive mode.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.background import BackgroundAgent, PromptHolder
from agent.loop import AgentLoop
from bus.events import ChatHistorySnapshot, InboundMessage
from bus.queue import MessageBus
from providers.qwen import QwenProvider
from session.manager import SessionManager

# Maximum events buffered per session before oldest are dropped
_LOG_QUEUE_MAXSIZE = 200

logger = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    session_id: str
    bus: MessageBus
    agent_loop: AgentLoop
    background_agent: BackgroundAgent
    loop_task: asyncio.Task
    bg_task: asyncio.Task
    workspace: Path
    log_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=_LOG_QUEUE_MAXSIZE))

    async def send(self, message: str, timeout: float = 60.0) -> str:
        """Publish a message and wait for the agent response."""
        channel, chat_id = "api", self.session_id
        await self.bus.publish_inbound(
            InboundMessage(
                channel=channel,
                sender_id="user",
                chat_id=chat_id,
                content=message,
            )
        )
        # Drain progress messages; return first content response
        while True:
            msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=timeout)
            if not msg.metadata.get("_progress"):
                return msg.content

    async def close(self) -> None:
        """Stop both agent tasks gracefully."""
        self.agent_loop.stop()
        self.background_agent.stop()
        await asyncio.gather(self.loop_task, self.bg_task, return_exceptions=True)
        logger.info("[REGISTRY] session closed: %s", self.session_id)


class SessionRegistry:
    """Process-level registry of active dual-LLM sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ActiveSession] = {}

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get(self, session_id: str) -> ActiveSession | None:
        return self._sessions.get(session_id)

    def create(self, session_id: str, workspace: Path, api_key: str) -> ActiveSession:
        """Create and start a new dual-LLM session. Replaces any existing session with the same id."""
        if session_id in self._sessions:
            raise ValueError(f"Session '{session_id}' already exists. Close it first.")

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
        log_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_LOG_QUEUE_MAXSIZE)

        async def _on_event(event: dict[str, Any]) -> None:
            if log_queue.full():
                try:
                    log_queue.get_nowait()  # drop oldest to make room
                except asyncio.QueueEmpty:
                    pass
            await log_queue.put(event)

        background_agent = BackgroundAgent(
            queue=background_queue,
            provider=background_provider,
            prompt_holder=prompt_holder,
            workspace=workspace,
            session_manager=session_manager,
            on_event=_on_event,
        )

        loop_task = asyncio.create_task(agent_loop.run(), name=f"loop:{session_id}")
        bg_task = asyncio.create_task(background_agent.run(), name=f"bg:{session_id}")

        session = ActiveSession(
            session_id=session_id,
            bus=bus,
            agent_loop=agent_loop,
            background_agent=background_agent,
            loop_task=loop_task,
            bg_task=bg_task,
            workspace=workspace,
            log_queue=log_queue,
        )
        self._sessions[session_id] = session
        logger.info("[REGISTRY] session created: %s (workspace=%s)", session_id, workspace)
        return session

    async def close(self, session_id: str) -> bool:
        """Stop and remove a session. Returns True if it existed."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.close()
        return True

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# Process-level singleton
registry = SessionRegistry()
