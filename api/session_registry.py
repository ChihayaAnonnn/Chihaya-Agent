"""
Global registry of active agent sessions.

Each session runs AgentLoop (persona lane) + BackgroundAgent (memory/tool lane)
as persistent asyncio tasks, mirroring the CLI `roleplay` interactive mode.

TTL:
  Sessions that have not received a message or keepalive within SESSION_TTL_S
  (default 1800s / 30 min, configurable via SESSION_TTL env var) are automatically
  closed by a background cleanup task started in the FastAPI lifespan.
"""

import asyncio
import logging
import os
import time
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

# Default session TTL in seconds (overridable via SESSION_TTL env var)
_DEFAULT_TTL_S = 1800

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
    last_active: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Reset the TTL clock."""
        self.last_active = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_active

    async def send(self, message: str, timeout: float = 60.0) -> str:
        """Publish a message, wait for the agent response, and update last_active."""
        self.touch()
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
        self._cleanup_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    def start_cleanup_task(
        self,
        ttl_s: int | None = None,
        interval_s: int = 60,
    ) -> None:
        """Start the background task that evicts idle sessions.

        Args:
            ttl_s: Seconds of inactivity before a session is closed.
                   Defaults to SESSION_TTL env var or _DEFAULT_TTL_S.
            interval_s: How often to scan for expired sessions (default 60s).
        """
        if self._cleanup_task is not None:
            return  # already running

        resolved_ttl = ttl_s or int(os.getenv("SESSION_TTL", str(_DEFAULT_TTL_S)))

        async def _cleanup_loop() -> None:
            logger.info("[REGISTRY] cleanup task started (ttl=%ds, interval=%ds)", resolved_ttl, interval_s)
            while True:
                await asyncio.sleep(interval_s)
                await self._evict_expired(resolved_ttl)

        self._cleanup_task = asyncio.create_task(_cleanup_loop(), name="session-cleanup")

    async def _evict_expired(self, ttl_s: int) -> None:
        expired = [
            sid for sid, s in self._sessions.items()
            if s.idle_seconds() > ttl_s
        ]
        for sid in expired:
            logger.info("[REGISTRY] evicting idle session: %s (idle=%.0fs)", sid, self._sessions[sid].idle_seconds())
            await self.close(sid)

    def stop_cleanup_task(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get(self, session_id: str) -> ActiveSession | None:
        return self._sessions.get(session_id)

    def touch(self, session_id: str) -> bool:
        """Reset the TTL clock for a session. Returns False if not found."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.touch()
        return True

    def create(self, session_id: str, workspace: Path, api_key: str) -> ActiveSession:
        """Create and start a new dual-LLM session."""
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
