"""Shared agentic runner factory for CLI and API usage."""

from pathlib import Path
from typing import Any

from agent.context import ContextBuilder
from agent.runner import run_agentic_loop
from agent.tools.filesystem import ReadFileTool, WriteFileTool
from agent.tools.registry import ToolRegistry
from providers.qwen import QwenProvider
from session.manager import SessionManager


def make_agentic_runner(workspace: Path, api_key: str):
    """
    Build a standalone agentic runner (tools + context) for cron/heartbeat/API.

    Returns an async callable: run(content, session_key) -> str
    """
    provider = QwenProvider(api_key=api_key)
    context = ContextBuilder(workspace)
    session_manager = SessionManager(workspace)
    tools = ToolRegistry()
    tools.register(ReadFileTool(workspace=workspace))
    tools.register(WriteFileTool(workspace=workspace))

    async def run(content: str, session_key: str = "cli:direct") -> str:
        session = session_manager.get_or_create(session_key)
        history = session.get_history(max_messages=20)
        messages = context.build_messages(
            history=history,
            current_message=content,
        )
        final_content, _ = await run_agentic_loop(
            provider=provider,
            tools=tools,
            messages=messages,
            model=provider.get_default_model(),
        )
        result = final_content or ""
        session.add_message("user", content)
        session.add_message("assistant", result)
        session_manager.save(session)
        return result

    return run
