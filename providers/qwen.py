"""Qwen LLM provider via DashScope OpenAI-compatible API."""

import asyncio
import json
import logging
import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from providers.base import LLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 1.0
RETRY_MAX_DELAY_S = 8.0

_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class QwenProvider(LLMProvider):
    """Qwen models via DashScope (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        api_base: str = DASHSCOPE_BASE_URL,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base)
        self._default_model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    def get_default_model(self) -> str:
        return self._default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        effective_model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("[QWEN] chat model=%s msgs=%d", effective_model, len(messages))
        t0 = time.monotonic()
        completion = await self._chat_with_retry(kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000

        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[ToolCallRequest] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCallRequest(id=tc.id, name=tc.function.name, arguments=arguments)
                )

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        response = LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
        logger.debug(
            "[QWEN] done model=%s finish=%s tokens=%s elapsed=%.0fms",
            effective_model, response.finish_reason, response.usage, elapsed_ms,
        )
        self._write_trace(messages, tools, effective_model, response, elapsed_ms)
        return response

    async def _chat_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Call the OpenAI client with retry + exponential backoff on transient errors.

        Retries on rate limits, connection/timeout errors and 5xx.
        Non-retryable errors (auth, bad request) propagate immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt == MAX_RETRIES - 1:
                    break
                delay = min(
                    RETRY_BASE_DELAY_S * (2 ** attempt),
                    RETRY_MAX_DELAY_S,
                )
                delay += random.uniform(0, delay * 0.1)
                logger.warning(
                    "[QWEN] retryable error (%s) attempt=%d/%d sleep=%.2fs: %s",
                    type(exc).__name__, attempt + 1, MAX_RETRIES, delay, exc,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc
