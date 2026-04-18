"""IdleMonitor: triggers agent-initiated messages after user inactivity.

Design constraints:
- All cooldown / quiet-hour state lives inside IdleMonitor. The ProactiveMessage
  event crossing the bus carries only content that the outbound consumer needs.
- Before spending an LLM call, a cheap rule filter decides whether the session
  even has a plausible reason to speak up (open follow-ups, delayed items). This
  keeps the idle-loop API cost sub-linear in active-session count.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.context import ContextBuilder
from agent.memory import MemoryStore
from bus.events import OutboundMessage, ProactiveMessage
from bus.queue import MessageBus
from providers.base import LLMProvider
from session.manager import SessionManager
from utils import log_event, new_trace_id

logger = logging.getLogger(__name__)

HOOK_KEYWORDS: tuple[str, ...] = (
    "明天", "下次", "等我", "稍后", "回头", "待定", "跟进", "后续",
    "later", "tomorrow", "follow up", "follow-up", "ping me",
)


@dataclass
class _SessionActivity:
    last_activity_ms: int
    channel: str
    chat_id: str
    cooldown_until_ms: int = 0
    proactive_count: int = 0
    tags: dict[str, Any] = field(default_factory=dict)


class IdleMonitor:
    """Watches per-session idle time and may push proactive messages to outbound.

    Typical wiring (CLI):

        monitor = IdleMonitor(bus=bus, provider=persona_provider,
                              workspace=ws, session_manager=sm)
        ...
        asyncio.create_task(monitor.run())

        # On every inbound user message:
        monitor.record_user_activity(session_key, channel, chat_id)
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        session_manager: SessionManager,
        model: str | None = None,
        idle_threshold_s: int = 300,
        cooldown_s: int = 600,
        quiet_hours: tuple[int, int] = (23, 8),
        poll_interval_s: int = 30,
        max_proactive_per_day: int = 5,
    ) -> None:
        self._bus = bus
        self._provider = provider
        self._workspace = workspace
        self._sessions = session_manager
        self._memory = MemoryStore(workspace)
        self._context = ContextBuilder(workspace)
        self._model = model or provider.get_default_model()

        self._idle_threshold_ms = idle_threshold_s * 1000
        self._cooldown_s = cooldown_s
        self._quiet_start, self._quiet_end = quiet_hours
        self._poll_interval_s = poll_interval_s
        self._max_proactive_per_day = max_proactive_per_day

        self._activity: dict[str, _SessionActivity] = {}
        self._running = False

    # ---------------- public API ----------------

    def record_user_activity(
        self, session_key: str, channel: str, chat_id: str
    ) -> None:
        entry = self._activity.get(session_key)
        now = _now_ms()
        if entry is None:
            self._activity[session_key] = _SessionActivity(
                last_activity_ms=now, channel=channel, chat_id=chat_id
            )
        else:
            entry.last_activity_ms = now
            entry.channel = channel
            entry.chat_id = chat_id

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        """Main loop: periodically check each session for idle triggering."""
        self._running = True
        logger.info(
            "[IDLE] monitor started threshold=%ds cooldown=%ds",
            self._idle_threshold_ms // 1000,
            self._cooldown_s,
        )
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval_s)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("[IDLE] tick error: %s", e)

    # ---------------- internals ----------------

    async def _tick(self) -> None:
        if self._in_quiet_hours():
            return
        for session_key, entry in list(self._activity.items()):
            if not self._should_trigger(session_key, entry):
                continue
            if not self._has_contextual_hook(session_key):
                continue
            await self._decide_and_send(session_key, entry)

    def _should_trigger(self, session_key: str, entry: _SessionActivity) -> bool:
        now = _now_ms()
        if now < entry.cooldown_until_ms:
            return False
        if now - entry.last_activity_ms < self._idle_threshold_ms:
            return False
        if entry.proactive_count >= self._max_proactive_per_day:
            return False
        return True

    def _in_quiet_hours(self) -> bool:
        h = datetime.now().hour
        start, end = self._quiet_start, self._quiet_end
        if start < end:
            return start <= h < end
        # wraps midnight, e.g. 23→8
        return h >= start or h < end

    def _has_contextual_hook(self, session_key: str) -> bool:
        """Cheap text-only precheck: does this session have something to follow up on?"""
        mem = self._memory.read_long_term()
        if "## 待跟进" in mem:
            tail = mem.split("## 待跟进", 1)[-1]
            # next ## header terminates the section
            section = tail.split("\n## ", 1)[0]
            if section.strip():
                return True

        session = self._sessions.get_or_create(session_key)
        for msg in session.messages[-6:]:
            content = (msg.get("content") or "")
            if any(k in content for k in HOOK_KEYWORDS):
                return True
        return False

    async def _decide_and_send(
        self, session_key: str, entry: _SessionActivity
    ) -> None:
        content = await self._generate_proactive_message(session_key, entry)
        if not content:
            return

        trace_id = new_trace_id()
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=entry.channel,
                chat_id=entry.chat_id,
                content=content,
                metadata={
                    "trace_id": trace_id,
                    "proactive": True,
                    "trigger_reason": "idle",
                },
            )
        )
        # also emit typed event for subscribers that care
        event = ProactiveMessage(
            session_key=session_key,
            channel=entry.channel,
            chat_id=entry.chat_id,
            content=content,
            trigger_reason="idle",
            metadata={"trace_id": trace_id},
        )
        log_event(
            "proactive_fired",
            trace_id=trace_id,
            session=session_key,
            trigger=event.trigger_reason,
        )

        entry.cooldown_until_ms = _now_ms() + self._cooldown_s * 1000
        entry.proactive_count += 1

    async def _generate_proactive_message(
        self, session_key: str, entry: _SessionActivity
    ) -> str | None:
        """Ask the persona LLM whether to speak up; empty → stay silent."""
        idle_minutes = max(
            1, (_now_ms() - entry.last_activity_ms) // 60_000
        )
        session = self._sessions.get_or_create(session_key)
        recent = session.get_history(max_messages=6)

        system = (
            "You may proactively initiate a message to the user, based on the "
            "persona and memory you were given. Only speak if you have a "
            "concrete, natural reason (an open follow-up, a relevant thought, "
            "a light check-in). If in doubt, stay silent.\n\n"
            "Output the message text only, no quotes, no JSON, no explanation. "
            "Return an EMPTY string if it's better to say nothing."
        )

        messages = self._context.build_persona_messages(
            history=recent,
            current_message=(
                f"[SYSTEM NOTICE] User has been idle for ~{idle_minutes} minutes. "
                f"Decide whether to proactively send a message now."
            ),
            channel=entry.channel,
            chat_id=entry.chat_id,
            ephemeral_hint=system,
        )
        try:
            response = await self._provider.chat(
                messages=messages,
                model=self._model,
                temperature=0.5,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning("[IDLE] proactive LLM call failed: %s", e)
            return None

        text = (response.content or "").strip()
        # Defensive: many models like to wrap refusal in JSON or brackets
        if text in {"", '""', "''", "null", "None"}:
            return None
        return text


def _now_ms() -> int:
    return int(time.time() * 1000)
