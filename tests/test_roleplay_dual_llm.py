"""Tests for dual-LLM roleplay flow."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from bus.events import ChatHistorySnapshot, PersonaPromptUpdate
from providers.base import LLMResponse
from agent.background import BackgroundAgent, PromptHolder


def _make_llm_response(content: str) -> LLMResponse:
    """Return a proper LLMResponse with no tool calls."""
    return LLMResponse(content=content, tool_calls=[], finish_reason="stop")


class TestPromptHolder(unittest.TestCase):
    def test_write_and_consume(self) -> None:
        async def run() -> None:
            holder = PromptHolder()
            self.assertIsNone(await holder.peek())

            update = PersonaPromptUpdate(ephemeral_hint="Be more concise.")
            await holder.write(update)
            self.assertEqual(await holder.peek(), update)

            update2 = PersonaPromptUpdate(ephemeral_hint="New guidance.", metadata={"v": 1})
            await holder.write(update2)
            peeked = await holder.peek()
            self.assertEqual(peeked.ephemeral_hint, "New guidance.")
            self.assertEqual(peeked.metadata, {"v": 1})

        asyncio.run(run())

    def test_consume_clears_value(self) -> None:
        """read_and_consume() must return the stored update exactly once."""
        async def run() -> None:
            holder = PromptHolder()
            update = PersonaPromptUpdate(ephemeral_hint="hint")
            await holder.write(update)
            self.assertEqual(await holder.read_and_consume(), update)
            self.assertIsNone(await holder.read_and_consume(), "value should be cleared after consume")

        asyncio.run(run())


class TestChatHistorySnapshot(unittest.TestCase):
    def test_required_fields(self) -> None:
        snap = ChatHistorySnapshot(
            session_key="cli:direct",
            messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}],
            current_user_message="hello again",
        )
        self.assertEqual(snap.session_key, "cli:direct")
        self.assertEqual(len(snap.messages), 2)
        self.assertEqual(snap.current_user_message, "hello again")


class TestBackgroundAgent(unittest.TestCase):
    def test_consumes_and_writes(self) -> None:
        async def run() -> None:
            queue: asyncio.Queue[ChatHistorySnapshot] = asyncio.Queue()
            holder = PromptHolder()
            mock_provider = MagicMock()
            mock_provider.get_default_model.return_value = "mock"
            mock_provider.chat = AsyncMock(
                return_value=_make_llm_response(
                    '{"ephemeral_hint": "Use a friendly tone."}'
                )
            )
            agent = BackgroundAgent(queue=queue, provider=mock_provider, prompt_holder=holder)
            agent._running = True
            snapshot = ChatHistorySnapshot(
                session_key="test",
                messages=[{"role": "user", "content": "hi"}],
                current_user_message="hi",
            )
            await queue.put(snapshot)
            task = asyncio.create_task(agent.run())
            await asyncio.sleep(0.5)
            agent.stop()
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            update = await holder.peek()
            self.assertIsNotNone(update)
            self.assertIn("friendly", update.ephemeral_hint)
            mock_provider.chat.assert_called()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
