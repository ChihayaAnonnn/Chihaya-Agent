"""Heartbeat service - periodic agent wake-up to check for tasks."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_S = 60  # 60s for demo

HEARTBEAT_PROMPT = """Read HEARTBEAT.md in your workspace (if it exists).
Follow any instructions or tasks listed there.
If nothing needs attention, reply with just: HEARTBEAT_OK"""

HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"


def _is_heartbeat_empty(content: str | None) -> bool:
    """Check if HEARTBEAT.md has no actionable content."""
    if not content:
        return True
    skip_patterns = {"- [ ]", "* [ ]", "- [x]", "* [x]"}
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--") or line in skip_patterns:
            continue
        return False
    return True


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.
    Reads HEARTBEAT.md and executes tasks via on_heartbeat callback.
    """

    def __init__(
        self,
        workspace: Path,
        on_heartbeat: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
    ) -> None:
        self.workspace = workspace
        self.on_heartbeat = on_heartbeat
        self.interval_s = interval_s
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def start(self) -> None:
        if not self.enabled:
            logger.info("[HEARTBEAT] disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[HEARTBEAT] started (every %ds)", self.interval_s)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[HEARTBEAT] error: %s", e)

    async def _tick(self) -> None:
        content = self._read_heartbeat_file()
        if _is_heartbeat_empty(content):
            logger.debug("[HEARTBEAT] tick: no tasks (HEARTBEAT.md empty)")
            return
        logger.info("[HEARTBEAT] tick: checking for tasks...")
        if self.on_heartbeat:
            try:
                response = await self.on_heartbeat(HEARTBEAT_PROMPT)
                if HEARTBEAT_OK_TOKEN.replace("_", "") in response.upper().replace("_", ""):
                    logger.info("[HEARTBEAT] OK (no action needed)")
                else:
                    logger.info("[HEARTBEAT] completed task")
            except Exception as e:
                logger.error("[HEARTBEAT] execution failed: %s", e)

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        if self.on_heartbeat:
            return await self.on_heartbeat(HEARTBEAT_PROMPT)
        return None
