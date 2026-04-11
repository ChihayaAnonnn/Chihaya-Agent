"""Agentic runner: reusable iterative LLM + tool-call loop."""

import json
import logging
from typing import Any, Awaitable, Callable

from providers.base import LLMProvider
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_IS_ERROR = ("Error:", "Error ")


def _is_error_result(result: str) -> bool:
    return result.startswith(_IS_ERROR)


def _tool_hint(tool_calls: list) -> str:
    def _fmt(tc):
        val = next(iter(tc.arguments.values()), None) if tc.arguments else None
        if not isinstance(val, str):
            return tc.name
        return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
    return ", ".join(_fmt(tc) for tc in tool_calls)


async def run_agentic_loop(
    provider: LLMProvider,
    tools: ToolRegistry,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_iterations: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str | None, list[str]]:
    """
    Iterative LLM + tool-call loop.

    Returns (final_content, tools_used).
    Callers own message list construction; this function mutates `messages` in-place
    with assistant and tool-result turns so callers can inspect the full trace.
    """
    iteration = 0
    final_content: str | None = None
    tools_used: list[str] = []

    while iteration < max_iterations:
        iteration += 1
        logger.debug(
            "[PROMPT] iter=%d msgs=%d\n%s",
            iteration,
            len(messages),
            json.dumps(messages, ensure_ascii=False, indent=2),
        )
        response = await provider.chat(
            messages=messages,
            tools=tools.get_definitions(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(
            "[LLM_RESPONSE] iter=%d finish=%s usage=%s content_len=%s tool_calls=%d",
            iteration,
            response.finish_reason,
            response.usage,
            len(response.content or ""),
            len(response.tool_calls),
        )

        if response.has_tool_calls:
            if on_progress:
                if response.content:
                    await on_progress(response.content)
                await on_progress(_tool_hint(response.tool_calls))

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
                tools_used.append(tool_call.name)
                logger.info(
                    "[TOOL_CALL] %s(%s)",
                    tool_call.name,
                    json.dumps(tool_call.arguments, ensure_ascii=False)[:300],
                )
                result = await tools.execute(tool_call.name, tool_call.arguments)
                if _is_error_result(result):
                    logger.warning(
                        "[TOOL_ERROR] %s → %s",
                        tool_call.name,
                        result[:500],
                    )
                else:
                    logger.debug("[TOOL_RESULT] %s → %s", tool_call.name, result[:300])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": result,
                })
        else:
            final_content = (response.content or "").strip()
            break

    return final_content, tools_used
