"""Subagent manager for background task execution."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import time as _time

from bus.events import InboundMessage
from bus.queue import MessageBus
from providers.base import LLMProvider

from agent.tools.filesystem import ReadFileTool, WriteFileTool
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SubagentManager:
    """
    Manages background subagent execution.
    Subagents run in the background and announce results via the message bus.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        restrict_to_workspace: bool = False,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info("[SUBAGENT] spawn id=%s label=%s", task_id, display_label)
        return (
            f"Subagent [{display_label}] started (id: {task_id}). "
            "I'll notify you when it completes."
        )

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("[SUBAGENT] starting id=%s task=%s", task_id, label[:50])

        try:
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))

            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1
                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })
                    for tool_call in response.tool_calls:
                        result = await tools.execute(
                            tool_call.name, tool_call.arguments
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info("[SUBAGENT] complete id=%s", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            logger.error("[SUBAGENT] failed id=%s error=%s", task_id, e)
            await self._announce_result(
                task_id, label, task, f"Error: {str(e)}", origin, "error"
            )

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"
        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize briefly for the user (1-2 sentences)."""

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )
        await self.bus.publish_inbound(msg)
        logger.debug("[SUBAGENT] announced id=%s to %s:%s", task_id, origin["channel"], origin["chat_id"])

    def _build_subagent_prompt(self, task: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task
2. Your final response will be reported back to the main agent
3. Be concise but informative

## Tools
- read_file: Read file contents
- write_file: Write content to files

## Workspace
{self.workspace}

When done, provide a clear summary of your findings or actions."""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
