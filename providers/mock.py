"""Mock LLM provider - deterministic, reproducible, no external dependencies."""

import logging
import uuid
from typing import Any

from providers.base import LLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message content."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            return ""
    return ""


def _has_tool_result(messages: list[dict[str, Any]], tool_name: str) -> bool:
    """Check if the last assistant tool call was followed by a tool result."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("function", {}).get("name") == tool_name:
                    # Check if next message is tool result for this call
                    if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                        return True
    return False


class MockProvider(LLMProvider):
    """
    Deterministic mock LLM for reproducible demos.
    Rule-based responses - no external API calls.
    """

    DEFAULT_MODEL = "mock/deterministic"

    def __init__(self) -> None:
        super().__init__()

    def get_default_model(self) -> str:
        return self.DEFAULT_MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        content = _last_user_content(messages)
        content_lower = content.lower().strip()

        tool_names = []
        if tools:
            tool_names = [
                t.get("function", {}).get("name")
                for t in tools
                if t.get("type") == "function"
            ]

        # Rule 1: "hello" / "你好" -> direct text
        if any(x in content_lower for x in ("hello", "你好", "hi", "hey")):
            logger.info("[MOCK] direct reply: greeting")
            return LLMResponse(content="Hello! I'm the mock agent. How can I help?")

        # Rule 2: "spawn" / "子任务" -> spawn tool call (if spawn available and no prior spawn result)
        if "spawn" in content_lower or "子任务" in content:
            if "spawn" in tool_names:
                task = content.split("spawn")[-1].strip() or "background task"
                if "子任务" in content:
                    task = content.split("子任务")[-1].strip() or "background task"
                logger.info("[MOCK] tool_call: spawn task=%r", task[:50])
                return LLMResponse(
                    content="I'll spawn a subagent for that.",
                    tool_calls=[
                        ToolCallRequest(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="spawn",
                            arguments={"task": task[:200], "label": task[:30]},
                        )
                    ],
                )

        # Rule 3: "read" / "读取" -> read_file tool call
        if "read" in content_lower or "读取" in content:
            if "read_file" in tool_names:
                path = "workspace/AGENTS.md"
                if "path" in content_lower or "文件" in content:
                    for word in content.split():
                        if "/" in word or "." in word:
                            path = word
                            break
                logger.info("[MOCK] tool_call: read_file path=%s", path)
                return LLMResponse(
                    content="Reading the file.",
                    tool_calls=[
                        ToolCallRequest(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name="read_file",
                            arguments={"path": path},
                        )
                    ],
                )

        # Rule 4: After tool result - return final summary
        if messages and messages[-1].get("role") == "tool":
            result = messages[-1].get("content", "")[:100]
            logger.info("[MOCK] final reply after tool result")
            return LLMResponse(
                content=f"Task completed. Result preview: {result}..."
            )

        # Default: echo
        logger.info("[MOCK] direct reply: echo")
        return LLMResponse(
            content=f"Echo: {content[:100]}{'...' if len(content) > 100 else ''}"
        )
