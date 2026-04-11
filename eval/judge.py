"""Answer correctness evaluation via LLM-as-judge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from providers.base import LLMProvider

DEFAULT_JUDGE_MODEL = "qwen-turbo"


@dataclass
class JudgeResult:
    correct: bool
    expected: str
    actual: str
    method: str = "llm_judge"
    detail: str = ""


_LLM_JUDGE_PROMPT = """\
You are an impartial evaluator. Given a question, the expected answer (which may be a literal value OR a rubric describing what a good answer should contain), and the model's actual response, decide whether the response is semantically correct.

A response is correct if it conveys the same essential information as the expected answer, even if the wording differs significantly.

Question: {question}

Expected answer:
{expected}

Model response:
{actual}

Reply with ONLY a JSON object (no markdown fences):
{{"verdict": "pass" or "fail", "reason": "<one sentence>"}}"""


async def llm_judge(
    question: str,
    expected: str,
    actual: str,
    provider: LLMProvider,
    model: str | None = None,
) -> JudgeResult:
    """Use an LLM to judge whether the response is semantically correct."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _LLM_JUDGE_PROMPT.format(
                question=question,
                expected=expected,
                actual=actual,
            ),
        }
    ]
    response = await provider.chat(
        messages=messages,
        model=model or DEFAULT_JUDGE_MODEL,
        temperature=0.0,
        max_tokens=256,
    )
    content = (response.content or "").strip()

    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(l for l in lines if not l.startswith("```")).strip()

    try:
        data = json.loads(content)
        verdict = str(data.get("verdict", "")).lower().strip()
        reason = str(data.get("reason", ""))
        return JudgeResult(
            correct=verdict == "pass",
            expected=expected,
            actual=actual,
            detail=reason,
        )
    except (json.JSONDecodeError, AttributeError):
        return JudgeResult(
            correct=False,
            expected=expected,
            actual=actual,
            detail=f"judge parse error: {content[:200]}",
        )
