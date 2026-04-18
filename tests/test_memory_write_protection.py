"""Tests for WriteFileTool's MEMORY.md token guard."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent.tools.filesystem import MEMORY_WRITE_HARD_LIMIT_TOKENS, WriteFileTool


class TestMemoryWriteProtection(unittest.TestCase):
    def test_oversized_write_rejected(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                tool = WriteFileTool(workspace=ws)
                # ~4x the hard limit in tokens
                oversized = "a" * (MEMORY_WRITE_HARD_LIMIT_TOKENS * 5 * 3)
                result = await tool.execute(
                    path="memory/MEMORY.md", content=oversized
                )
                self.assertIn("rejected", result)
                self.assertFalse((ws / "memory" / "MEMORY.md").exists())

        asyncio.run(run())

    def test_within_budget_writes_through(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                tool = WriteFileTool(workspace=ws)
                payload = "## 基本信息\n正常小文件。\n"
                result = await tool.execute(
                    path="memory/MEMORY.md", content=payload
                )
                self.assertIn("Successfully wrote", result)
                self.assertEqual(
                    (ws / "memory" / "MEMORY.md").read_text(encoding="utf-8"),
                    payload,
                )

        asyncio.run(run())

    def test_guard_only_targets_memory_md(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                ws = Path(tmp)
                tool = WriteFileTool(workspace=ws)
                big = "a" * (MEMORY_WRITE_HARD_LIMIT_TOKENS * 5 * 3)
                result = await tool.execute(
                    path="notes/BIGFILE.md", content=big
                )
                self.assertIn("Successfully wrote", result)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
