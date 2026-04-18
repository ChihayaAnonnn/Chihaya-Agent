"""Tests for LLM-driven memory compression and MEMORY.md budget enforcement."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agent.memory import (
    MEMORY_MAX_TOKENS,
    MemoryStore,
    _estimate_tokens,
)
from providers.base import LLMResponse
from session.manager import Session


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], finish_reason="stop")


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(_estimate_tokens(""), 0)

    def test_monotonic(self) -> None:
        self.assertLess(_estimate_tokens("short"), _estimate_tokens("short " * 100))


class TestCompressSession(unittest.TestCase):
    def test_compress_trims_and_appends_history(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                memory = MemoryStore(ws)

                session = Session(key="cli:test")
                for i in range(30):
                    role = "user" if i % 2 == 0 else "assistant"
                    session.add_message(role, f"turn-{i}: " + ("x" * 200))

                provider = MagicMock()
                provider.get_default_model.return_value = "mock-model"
                provider.chat = AsyncMock(
                    return_value=_resp("## Summary\nUser talked about many turns.")
                )

                changed = await memory.compress_session(
                    session, provider=provider, keep_count=10
                )
                self.assertTrue(changed)
                self.assertGreater(session.last_consolidated, 0)
                self.assertTrue(memory.history_file.exists())
                self.assertIn(
                    "Summary", memory.history_file.read_text(encoding="utf-8")
                )
                provider.chat.assert_called_once()

        asyncio.run(run())

    def test_compress_noop_when_too_few_messages(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                memory = MemoryStore(ws)
                session = Session(key="cli:test")
                for i in range(5):
                    session.add_message("user", f"turn-{i}")

                provider = MagicMock()
                provider.get_default_model.return_value = "mock"
                provider.chat = AsyncMock(return_value=_resp("summary"))

                changed = await memory.compress_session(
                    session, provider=provider, keep_count=10
                )
                self.assertFalse(changed)
                provider.chat.assert_not_called()

        asyncio.run(run())


class TestArchiveOldMemory(unittest.TestCase):
    def test_archive_noop_when_under_budget(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                memory = MemoryStore(ws)
                memory.write_long_term("## 基本信息\n短")
                provider = MagicMock()
                provider.get_default_model.return_value = "mock"
                provider.chat = AsyncMock()
                changed = await memory.archive_old_memory(provider=provider)
                self.assertFalse(changed)
                provider.chat.assert_not_called()

        asyncio.run(run())

    def test_archive_trims_when_over_budget(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                memory = MemoryStore(ws)
                # make it over budget by padding
                big = "## 最近关键决策\n" + ("- 某次决策。\n" * 2000)
                memory.write_long_term(big)
                self.assertGreater(memory.long_term_token_count(), MEMORY_MAX_TOKENS)

                pruned = "## 基本信息\n精简的画像。"
                archived = "(dropped stale decisions)"
                provider = MagicMock()
                provider.get_default_model.return_value = "mock"
                provider.chat = AsyncMock(
                    return_value=_resp(f"{pruned}\n===ARCHIVE===\n{archived}")
                )

                changed = await memory.archive_old_memory(
                    provider=provider, budget_tokens=MEMORY_MAX_TOKENS
                )
                self.assertTrue(changed)
                self.assertEqual(memory.read_long_term(), pruned)
                self.assertIn(archived, memory.history_file.read_text(encoding="utf-8"))

        asyncio.run(run())

    def test_archive_refuses_still_oversized_output(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                memory = MemoryStore(ws)
                big = "x" * 20000
                memory.write_long_term(big)

                # LLM returns something that's STILL oversized → refuse
                still_big = "## 基本信息\n" + "y" * 20000
                provider = MagicMock()
                provider.get_default_model.return_value = "mock"
                provider.chat = AsyncMock(
                    return_value=_resp(f"{still_big}\n===ARCHIVE===\nnope")
                )
                changed = await memory.archive_old_memory(
                    provider=provider, budget_tokens=MEMORY_MAX_TOKENS
                )
                self.assertFalse(changed)
                # original preserved
                self.assertEqual(memory.read_long_term(), big)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
