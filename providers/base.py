"""Base LLM provider interface."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str | None = None, api_base: str | None = None) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self._trace_file: Path | None = None
        self._trace_seq: int = 0

    def set_trace_file(self, path: Path | None) -> None:
        """Set the JSONL file to record all LLM inputs and outputs. Pass None to disable."""
        self._trace_file = path
        self._trace_seq = 0

    def _write_trace(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        response: LLMResponse,
        elapsed_ms: float,
    ) -> None:
        """Append a full LLM call record to the trace file."""
        if not self._trace_file:
            return
        self._trace_seq += 1
        entry = {
            "seq": self._trace_seq,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "elapsed_ms": round(elapsed_ms, 1),
            "input": {
                "messages": messages,
                "tools": tools or [],
            },
            "output": {
                "content": response.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            },
        }
        try:
            with open(self._trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # tracing must never break the main flow

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
