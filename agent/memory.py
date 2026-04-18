"""Memory system for persistent agent memory.

Two-layer design:
- MEMORY.md: long-term facts injected into every prompt (token-budgeted).
- HISTORY.md: compressed logs, NOT injected; keeps full trail for audit.

Compression is LLM-driven (not mechanical truncation) so information density
stays high as conversations grow.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from providers.base import LLMProvider
from utils import ensure_dir

if TYPE_CHECKING:
    from session.manager import Session

logger = logging.getLogger(__name__)

# token budget for MEMORY.md; when exceeded, archive_old_memory is triggered
MEMORY_MAX_TOKENS = 1000
# soft warning threshold below the hard cap
MEMORY_SOFT_WARN_TOKENS = 800
# lower bound for how much context compress_session preserves raw
DEFAULT_KEEP_MESSAGES = 10
# rough char→token estimator fallback when tiktoken isn't available
_CHARS_PER_TOKEN_FALLBACK = 2.5


def _estimate_tokens(text: str) -> int:
    """Best-effort token count.

    Uses tiktoken's cl100k_base when installed (accurate for GPT/Qwen family);
    otherwise falls back to a simple char-based heuristic. Never raises.
    """
    if not text:
        return 0
    try:  # lazy import so tiktoken stays optional
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text) / _CHARS_PER_TOKEN_FALLBACK)


_COMPRESS_SYSTEM_PROMPT = """You compress conversation segments into durable notes.

Keep:
- User background, preferences, and any *changes* to them
- Decisions that were made (don't drop them)
- Unresolved tasks and follow-ups
- Technical constraints (stack, scope, deadlines)

Drop:
- Tactical code details unless they encode a core architectural choice
- Failed attempts and retries
- Pleasantries

Output: Markdown with `##` section headers. Hard cap ≈300 tokens."""


_ARCHIVE_SYSTEM_PROMPT = """You prune a long-lived MEMORY.md file.

Goal: keep the file under {budget} tokens while preserving the facts the
persona needs to answer future questions.

Rules:
- Return two sections separated by the literal line `===ARCHIVE===`:
  1. The trimmed MEMORY.md (Markdown, within budget)
  2. What was removed (Markdown, free-form; will be appended to HISTORY.md)
- Preserve sections: `## 基本信息`, `## 重要偏好`, `## 待跟进` (never drop non-empty items)
- Safe to drop: stale items in `## 最近关键决策` older than ~6 months,
  `## 当前进行中的项目` entries marked done.
- Never invent new facts.
"""


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (compressed log)."""

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

    def long_term_token_count(self) -> int:
        return _estimate_tokens(self.read_long_term())

    # ------------------------------------------------------------------
    # LLM-driven compression
    # ------------------------------------------------------------------

    async def compress_session(
        self,
        session: "Session",
        provider: LLMProvider,
        *,
        model: str | None = None,
        keep_count: int = DEFAULT_KEEP_MESSAGES,
    ) -> bool:
        """LLM-summarize old messages into HISTORY.md and trim session in place.

        Returns True if compression happened. Safe to call repeatedly.
        """
        messages = session.messages
        if len(messages) <= keep_count:
            return False

        start = max(0, session.last_consolidated)
        end = len(messages) - keep_count
        segment = messages[start:end]
        if not segment:
            return False

        transcript = _format_segment(segment)
        summary = await _summarize_with_llm(
            transcript=transcript,
            provider=provider,
            model=model,
        )
        if not summary:
            logger.warning("[MEMORY] compress_session: empty summary, skipping")
            return False

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"## Session {session.key} @ {stamp} "
            f"(messages {start}..{end - 1})\n\n{summary.strip()}"
        )
        session.last_consolidated = end
        logger.info(
            "[MEMORY] compressed %d messages → HISTORY.md (session=%s)",
            len(segment),
            session.key,
        )
        return True

    async def archive_old_memory(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        budget_tokens: int = MEMORY_MAX_TOKENS,
    ) -> bool:
        """If MEMORY.md exceeds budget, let an LLM prune it and archive the cut-out.

        The pruned version is written back to MEMORY.md; removed content is
        appended to HISTORY.md for auditability. Returns True if pruning ran.
        """
        current = self.read_long_term()
        if _estimate_tokens(current) <= budget_tokens:
            return False

        logger.info(
            "[MEMORY] archiving: current=%d tokens budget=%d",
            _estimate_tokens(current),
            budget_tokens,
        )
        system = _ARCHIVE_SYSTEM_PROMPT.format(budget=budget_tokens)
        response = await provider.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Current MEMORY.md:\n\n{current}"},
            ],
            model=model or provider.get_default_model(),
            temperature=0.2,
            max_tokens=1536,
        )
        content = (response.content or "").strip()
        if "===ARCHIVE===" not in content:
            logger.warning("[MEMORY] archive output missing marker, aborting")
            return False

        trimmed, _, archived = content.partition("===ARCHIVE===")
        trimmed = trimmed.strip()
        archived = archived.strip()

        # Guard: never write an empty MEMORY.md
        if not trimmed:
            logger.warning("[MEMORY] archive produced empty MEMORY.md, aborting")
            return False
        # Guard: still over budget → refuse, keep original
        if _estimate_tokens(trimmed) > budget_tokens:
            logger.warning(
                "[MEMORY] archive result still over budget (%d > %d), aborting",
                _estimate_tokens(trimmed),
                budget_tokens,
            )
            return False

        self.write_long_term(trimmed)
        if archived:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.append_history(f"## MEMORY archive @ {stamp}\n\n{archived}")
        logger.info("[MEMORY] archive ok: new size=%d tokens", _estimate_tokens(trimmed))
        return True

    # ------------------------------------------------------------------
    # Back-compat: old mechanical consolidate (still used by tests/eval)
    # ------------------------------------------------------------------

    async def consolidate(
        self,
        session: "Session",
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> None:
        """Legacy mechanical truncation path (no LLM).

        Retained for eval-harness replay and scripts that must not spend tokens.
        New code should use :meth:`compress_session` instead.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(
                "[MEMORY] legacy consolidate (archive_all): %d messages",
                len(session.messages),
            )
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
                "[MEMORY] legacy consolidate: %d to consolidate, %d keep",
                len(old_messages),
                keep_count,
            )

        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            entry = (
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: "
                f"{m['content'][:200]}"
            )
            self.append_history(entry)

        session.last_consolidated = (
            0 if archive_all else len(session.messages) - keep_count
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _format_segment(messages: list[dict[str, Any]]) -> str:
    """Turn raw session messages into a readable transcript for the summarizer."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        ts = (m.get("timestamp") or "")[:16]
        header = f"[{ts}] {role.upper()}" if ts else role.upper()
        lines.append(f"{header}:\n{content}")
    return "\n\n".join(lines)


async def _summarize_with_llm(
    transcript: str,
    provider: LLMProvider,
    model: str | None,
) -> str:
    """Call the provider to compress a transcript segment. Never raises."""
    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            model=model or provider.get_default_model(),
            temperature=0.2,
            max_tokens=512,
        )
        return (response.content or "").strip()
    except Exception as e:
        logger.error("[MEMORY] summarize failed: %s", e)
        return ""
