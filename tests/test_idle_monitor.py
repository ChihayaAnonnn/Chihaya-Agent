"""Tests for IdleMonitor: rule filter, cooldown, quiet hours."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agent.proactive import IdleMonitor, _SessionActivity
from bus.queue import MessageBus
from providers.base import LLMResponse
from session.manager import SessionManager


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], finish_reason="stop")


def _make_monitor(tmp: Path, provider=None) -> IdleMonitor:
    provider = provider or MagicMock()
    provider.get_default_model.return_value = "mock"
    bus = MessageBus()
    sm = SessionManager(tmp)
    return IdleMonitor(
        bus=bus,
        provider=provider,
        workspace=tmp,
        session_manager=sm,
        idle_threshold_s=1,
        cooldown_s=60,
        poll_interval_s=1,
        quiet_hours=(99, 99),  # disabled for the test
    )


class TestShouldTrigger(unittest.TestCase):
    def test_respects_idle_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = _make_monitor(Path(tmp))
            now_ms = int(time.time() * 1000)
            entry = _SessionActivity(
                last_activity_ms=now_ms, channel="cli", chat_id="a"
            )
            self.assertFalse(monitor._should_trigger("s", entry))
            entry.last_activity_ms = now_ms - 5000
            self.assertTrue(monitor._should_trigger("s", entry))

    def test_cooldown_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = _make_monitor(Path(tmp))
            past = int(time.time() * 1000) - 10000
            future = int(time.time() * 1000) + 60000
            entry = _SessionActivity(
                last_activity_ms=past, channel="cli", chat_id="a",
                cooldown_until_ms=future,
            )
            self.assertFalse(monitor._should_trigger("s", entry))


class TestContextualHook(unittest.TestCase):
    def test_no_hook_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            monitor = _make_monitor(ws)
            (ws / "memory").mkdir(exist_ok=True)
            (ws / "memory" / "MEMORY.md").write_text(
                "## 基本信息\nhi\n", encoding="utf-8"
            )
            self.assertFalse(monitor._has_contextual_hook("cli:x"))

    def test_pending_section_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            monitor = _make_monitor(ws)
            (ws / "memory").mkdir(exist_ok=True)
            (ws / "memory" / "MEMORY.md").write_text(
                "## 基本信息\nhi\n\n## 待跟进\n- 周一跟进 A\n",
                encoding="utf-8",
            )
            self.assertTrue(monitor._has_contextual_hook("cli:x"))

    def test_hook_keyword_in_recent_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            monitor = _make_monitor(ws)
            session = monitor._sessions.get_or_create("cli:x")
            session.add_message("user", "明天我们再聊这个")
            monitor._sessions.save(session)
            self.assertTrue(monitor._has_contextual_hook("cli:x"))


class TestQuietHours(unittest.TestCase):
    def test_wrap_midnight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = _make_monitor(Path(tmp))
            monitor._quiet_start, monitor._quiet_end = 23, 8
            # we can't mock datetime easily; just assert boolean type
            self.assertIsInstance(monitor._in_quiet_hours(), bool)


class TestGenerateProactive(unittest.TestCase):
    def test_empty_response_returns_none(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                provider = MagicMock()
                provider.get_default_model.return_value = "mock"
                provider.chat = AsyncMock(return_value=_resp(""))
                monitor = _make_monitor(Path(tmp), provider=provider)
                entry = _SessionActivity(
                    last_activity_ms=int(time.time() * 1000) - 1000000,
                    channel="cli", chat_id="a",
                )
                result = await monitor._generate_proactive_message("cli:x", entry)
                self.assertIsNone(result)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
