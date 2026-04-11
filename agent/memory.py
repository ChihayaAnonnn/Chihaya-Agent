"""Memory system for persistent agent memory."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from utils import ensure_dir

if TYPE_CHECKING:
    from session.manager import Session

logger = logging.getLogger(__name__)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path) -> None:
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def consolidate(
        self,
        session: "Session",
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> None:
        """
        Consolidate old messages into HISTORY.md.
        Mock mode: no LLM call, just append summary to HISTORY.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("[MEMORY] consolidation (archive_all): %d messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return
            if len(session.messages) - session.last_consolidated <= 0:
                return
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(
                "[MEMORY] consolidation: %d to consolidate, %d keep",
                len(old_messages),
                keep_count,
            )

        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            entry = f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content'][:200]}"
            self.append_history(entry)

        session.last_consolidated = (
            0 if archive_all else len(session.messages) - keep_count
        )
        logger.info(
            "[MEMORY] consolidation done: last_consolidated=%d",
            session.last_consolidated,
        )
