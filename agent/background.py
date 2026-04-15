"""Background agent: heavy lane — tools, memory, planning, persona guidance."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from bus.events import ChatHistorySnapshot, PersonaPromptUpdate
from providers.base import LLMProvider
from session.manager import SessionManager

from agent.context import ContextBuilder
from agent.memory import MemoryStore
from agent.runner import run_agentic_loop
from agent.subagent import SubagentManager
from agent.tools.filesystem import ReadFileTool, WriteFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.spawn import SpawnTool

# Callback type: receives an event dict, fires-and-forgets
OnEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None

logger = logging.getLogger(__name__)

BACKGROUND_SYSTEM_PROMPT = """You are the background reasoning agent in a dual-LLM roleplay system.

A separate persona model handles every user turn. You run concurrently and have two
ways to influence it — use either, both, or neither depending on the conversation:

1. **Update context files** (via read_file / write_file)
   - USER.md — learned preferences, ongoing projects, things the user cares about
   - memory/MEMORY.md — important facts, decisions, long-term knowledge
   Only write when the conversation reveals something worth persisting.
   Read a file first to avoid overwriting existing content.

2. **Return an ephemeral hint** for the persona's very next turn.
   Useful for short-lived steering (tone shift, something to mention once).

After using tools (or choosing not to), finish with a JSON object:
{"ephemeral_hint": "<one sentence or empty string>"}

If nothing needs updating or hinting, return {"ephemeral_hint": ""}."""

MEMORY_CONSOLIDATION_THRESHOLD = 20


class PromptHolder:
    """Shared state for PersonaPromptUpdate. Background writes, persona reads."""

    def __init__(self) -> None:
        self._value: PersonaPromptUpdate | None = None
        self._lock = asyncio.Lock()

    async def write(self, update: PersonaPromptUpdate) -> None:
        async with self._lock:
            self._value = update

    def read(self) -> PersonaPromptUpdate | None:
        return self._value


class BackgroundAgent:
    """
    Heavy lane: consumes ChatHistorySnapshot from queue, runs an agentic loop
    with tools and memory access, writes PersonaPromptUpdate to shared state.
    """

    def __init__(
        self,
        queue: asyncio.Queue[ChatHistorySnapshot],
        provider: LLMProvider,
        prompt_holder: PromptHolder,
        workspace: Path | None = None,
        model: str | None = None,
        session_manager: SessionManager | None = None,
        on_event: "OnEventCallback" = None,
    ) -> None:
        self.queue = queue
        self.provider = provider
        self.prompt_holder = prompt_holder
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.on_event = on_event
        self._running = False
        self._turns_since_consolidation: dict[str, int] = {}

        if workspace is not None:
            self.context = ContextBuilder(workspace)
            self.memory = MemoryStore(workspace)
            self.sessions = session_manager or SessionManager(workspace)
            self.tools = self._build_tools(workspace)
        else:
            self.context = None
            self.memory = None
            self.sessions = None
            self.tools = ToolRegistry()

    def _build_tools(self, workspace: Path) -> ToolRegistry:
        tools = ToolRegistry()
        tools.register(ReadFileTool(workspace=workspace))
        tools.register(WriteFileTool(workspace=workspace))
        # SpawnTool excluded: background agent does not route through bus
        return tools

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("[BACKGROUND] agent started")
        while self._running:
            try:
                snapshot = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                update = await self._analyze(snapshot)
                if update:
                    await self.prompt_holder.write(update)
                await self._maybe_consolidate(snapshot)
            except Exception as e:
                logger.error("[BACKGROUND] error: %s", e)

    async def _emit(self, event: dict[str, Any]) -> None:
        """Fire-and-forget event to the registered callback."""
        if self.on_event is not None:
            try:
                await self.on_event(event)
            except Exception:
                pass

    async def _analyze(self, snapshot: ChatHistorySnapshot) -> PersonaPromptUpdate | None:
        logger.info("[BACKGROUND] analyzing snapshot: session=%s", snapshot.session_key)
        await self._emit({"type": "start", "session": snapshot.session_key,
                          "message": snapshot.current_user_message[:120]})

        system_prompt = BACKGROUND_SYSTEM_PROMPT
        if self.context is not None:
            mem = self.memory.get_memory_context() if self.memory else ""
            if mem:
                system_prompt = f"{system_prompt}\n\n{mem}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in snapshot.messages[-20:]:
            messages.append({"role": m["role"], "content": m.get("content", "")})
        messages.append({
            "role": "user",
            "content": (
                f"Current user message: {snapshot.current_user_message}\n\n"
                "Analyze the conversation. Update context files if warranted, "
                "then return the JSON with ephemeral_hint."
            ),
        })

        logger.info("[BACKGROUND] running agentic loop")

        async def _on_progress(progress: str) -> None:
            await self._emit({"type": "tool_call", "detail": progress})

        final_content, tools_used = await run_agentic_loop(
            provider=self.provider,
            tools=self.tools,
            messages=messages,
            model=self.model,
            max_iterations=10,
            temperature=0.3,
            max_tokens=1024,
            on_progress=_on_progress,
        )
        if tools_used:
            logger.info("[BACKGROUND] tools used: %s", ", ".join(tools_used))
            await self._emit({"type": "tools_used", "tools": tools_used})

        hint = self._extract_hint(final_content or "")
        if hint:
            await self._emit({"type": "hint", "hint": hint})
        await self._emit({"type": "done", "tools_used": tools_used})
        return PersonaPromptUpdate(ephemeral_hint=hint)

    @staticmethod
    def _extract_hint(content: str) -> str:
        """Best-effort extraction of ephemeral_hint from the model's final output."""
        content = content.strip()
        if not content:
            return ""
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(l for l in lines if not l.startswith("```")).strip()
        try:
            data = json.loads(content)
            return str(data.get("ephemeral_hint", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            return ""

    async def _maybe_consolidate(self, snapshot: ChatHistorySnapshot) -> None:
        """Consolidate session memory when message count crosses the threshold."""
        if self.memory is None or self.sessions is None:
            return
        session_key = snapshot.session_key
        count = self._turns_since_consolidation.get(session_key, 0) + 1
        self._turns_since_consolidation[session_key] = count

        if count < MEMORY_CONSOLIDATION_THRESHOLD:
            return

        self._turns_since_consolidation[session_key] = 0
        session = self.sessions.get_or_create(session_key)
        if len(session.messages) > 0:
            logger.info("[BACKGROUND] consolidating memory for session: %s", session_key)
            await self._emit({"type": "consolidate", "session": session_key})
            try:
                await self.memory.consolidate(session, archive_all=False)
                self.sessions.save(session)
            except Exception as e:
                logger.error("[BACKGROUND] memory consolidation error: %s", e)
