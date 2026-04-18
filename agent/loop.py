"""Agent loop: persona lane — single-shot LLM call, no tools, no iteration."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from bus.events import ChatHistorySnapshot, InboundMessage, OutboundMessage
from bus.queue import MessageBus
from providers.base import LLMProvider

from agent.background import PromptHolder
from session.manager import SessionManager

from agent.context import ContextBuilder
from agent.proactive import IdleMonitor
from utils import log_event, new_trace_id

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    Persona lane of the dual-loop agent.

    Responsibilities:
    - Receive inbound messages from the bus
    - Push ChatHistorySnapshot to the background queue (fire-and-forget)
    - Read latest PersonaPromptUpdate from the shared prompt holder
    - Make a single LLM call using persona context
    - Emit response immediately (no tools, no iteration)
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        memory_window: int = 50,
        session_manager: SessionManager | None = None,
        *,
        persona_provider: LLMProvider | None = None,
        background_queue: asyncio.Queue[ChatHistorySnapshot] | None = None,
        prompt_holder: PromptHolder | None = None,
        idle_monitor: IdleMonitor | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.bus = bus
        self.ephemeral = ephemeral
        self.provider = persona_provider or provider
        self.background_queue = background_queue
        self.prompt_holder = prompt_holder
        self.idle_monitor = idle_monitor
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)

        self._running = False

    async def run(self) -> None:
        """Run the persona loop, processing messages from the bus."""
        self._running = True
        logger.info("[LOOP] loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0,
                )
                try:
                    response = await self._process_message(msg)
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="",
                                metadata=msg.metadata or {},
                            )
                        )
                except Exception as e:
                    logger.error("[LOOP] error processing message: %s", e)
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"Sorry, I encountered an error: {str(e)}",
                        )
                    )
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the persona loop."""
        self._running = False
        logger.info("[LOOP] loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message with a single LLM call."""
        trace_id = new_trace_id()
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("[LOOP] processing message from %s:%s: %s", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key, ephemeral=self.ephemeral)

        if self.idle_monitor is not None:
            self.idle_monitor.record_user_activity(key, msg.channel, msg.chat_id)

        if msg.content.strip().lower() == "/help":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Commands:\n/help — Show commands",
            )

        history = session.get_history(max_messages=self.memory_window)

        # Fire-and-forget: background agent receives snapshot for async analysis
        if self.background_queue is not None:
            snapshot = ChatHistorySnapshot(
                session_key=key,
                messages=history,
                current_user_message=msg.content,
                trace_id=trace_id,
            )
            self.background_queue.put_nowait(snapshot)

        ephemeral_hint = ""
        if self.prompt_holder is not None:
            if update := await self.prompt_holder.read_and_consume():
                ephemeral_hint = update.ephemeral_hint or ""

        messages = self.context.build_persona_messages(
            history=history,
            current_message=msg.content,
            channel=msg.channel,
            chat_id=msg.chat_id,
            ephemeral_hint=ephemeral_hint,
        )
        logger.info(
            "[PERSONA] sending messages to provider:\n%s",
            json.dumps(messages, ensure_ascii=False, indent=2),
        )
        # Single shot — no tools, no iteration
        t0 = time.monotonic()
        response = await self.provider.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PERSONA] response from provider: %s", response)
        final_content = (response.content or "").strip()

        if not final_content:
            final_content = "..."

        log_event(
            "persona_turn",
            trace_id=trace_id,
            session=key,
            channel=msg.channel,
            latency_ms=latency_ms,
            usage=response.usage or None,
            ephemeral_hint=ephemeral_hint or None,
            history_len=len(history),
        )

        logger.info("[LOOP] response to %s:%s", msg.channel, msg.sender_id)
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata={**(msg.metadata or {}), "trace_id": trace_id},
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI usage)."""
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
        )
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress
        )
        return response.content if response else ""
