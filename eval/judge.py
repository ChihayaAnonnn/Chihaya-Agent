"""Answer correctness evaluation via LLM-as-judge.

Two-stage design to control cost:

1. ``keyword_judge`` — deterministic, zero-API. If every expected keyword
   appears verbatim (case-insensitive) in the response, we mark it correct
   without paying for an LLM call.
2. ``llm_judge`` — fallback semantic check when keyword coverage is partial
   or the question is a rubric/style check rather than a fact retrieval.
"""

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


def keyword_judge(
    actual: str,
    expected_keywords: list[str],
) -> JudgeResult | None:
    """Fast zero-cost judge.

    Returns a JudgeResult if all keywords match (correct=True); returns None
    when keywords are empty or incomplete so callers can fall back to LLM.
    """
    if not expected_keywords:
        return None
    actual_lc = (actual or "").lower()
    missing = [k for k in expected_keywords if k.lower() not in actual_lc]
    if not missing:
        return JudgeResult(
            correct=True,
            expected=", ".join(expected_keywords),
            actual=actual,
            method="keyword",
            detail=f"all {len(expected_keywords)} keywords matched",
        )
    # Partial match → defer to LLM for semantic call
    return None


async def judge_response(
    question: str,
    expected: str,
    actual: str,
    provider: LLMProvider,
    *,
    expected_keywords: list[str] | None = None,
    model: str | None = None,
) -> JudgeResult:
    """Unified entrypoint: try keyword judge first, fall back to LLM."""
    if expected_keywords:
        kw = keyword_judge(actual, expected_keywords)
        if kw is not None:
            return kw
    return await llm_judge(
        question=question,
        expected=expected,
        actual=actual,
        provider=provider,
        model=model,
    )


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
