"""Eval runner: replays dataset conversations through the background agent,
then tests persona recall on the final question."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bus.events import ChatHistorySnapshot, PersonaPromptUpdate
from providers.base import LLMProvider

from agent.background import BackgroundAgent, PromptHolder
from agent.context import ContextBuilder
from session.manager import SessionManager

from eval.dataset import EvalEntry, EvalSession, iter_turn_pairs
from eval.judge import DEFAULT_JUDGE_MODEL, JudgeResult, llm_judge
from eval.workspace import (
    cleanup_eval_workspace,
    create_eval_workspace,
    read_memory_snapshot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    question_id: str
    question_type: str
    question: str
    expected_answer: str
    model_answer: str
    correct: bool
    judge_detail: str
    memory_snapshot: str
    turns_replayed: int
    sessions_count: int
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Single-entry runner
# ---------------------------------------------------------------------------

async def run_single_entry(
    entry: EvalEntry,
    provider: LLMProvider,
    *,
    eval_workspace_base: Path,
    model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    clear_history_between_sessions: bool = True,
    keep_workspace: bool = True,
    verbose: bool = False,
) -> EvalResult:
    """Run evaluation on a single dataset entry.

    1. Create a fresh workspace
    2. Replay all turns through the background agent
    3. Ask the persona the final question
    4. Judge the answer
    """
    t0 = time.monotonic()
    workspace = create_eval_workspace(eval_workspace_base, entry.question_id)
    model_name = model or provider.get_default_model()

    try:
        result = await _replay_and_answer(
            entry=entry,
            provider=provider,
            workspace=workspace,
            model=model_name,
            judge_model=judge_model,
            clear_history_between_sessions=clear_history_between_sessions,
            verbose=verbose,
        )
    except Exception as exc:
        logger.error("entry %s failed: %s", entry.question_id, exc, exc_info=True)
        result = EvalResult(
            question_id=entry.question_id,
            question_type=entry.question_type,
            question=entry.question,
            expected_answer=entry.answer,
            model_answer="",
            correct=False,
            judge_detail="",
            memory_snapshot=read_memory_snapshot(workspace),
            turns_replayed=entry.total_turns,
            sessions_count=len(entry.sessions),
            elapsed_seconds=time.monotonic() - t0,
            error=str(exc),
        )
    finally:
        if not keep_workspace:
            cleanup_eval_workspace(workspace)
        else:
            logger.info("[EVAL] workspace kept at %s", workspace)

    result.elapsed_seconds = time.monotonic() - t0
    return result


async def _replay_and_answer(
    entry: EvalEntry,
    provider: LLMProvider,
    workspace: Path,
    model: str,
    judge_model: str,
    clear_history_between_sessions: bool,
    verbose: bool,
) -> EvalResult:
    prompt_holder = PromptHolder()
    session_manager = SessionManager(workspace)
    background = BackgroundAgent(
        queue=asyncio.Queue(),
        provider=provider,
        prompt_holder=prompt_holder,
        workspace=workspace,
        model=model,
        session_manager=session_manager,
    )

    history: list[dict[str, Any]] = []
    turns_replayed = 0

    for session_idx, session in enumerate(entry.sessions):
        if clear_history_between_sessions and session_idx > 0:
            history = []

        for user_turn, assistant_turn in iter_turn_pairs(session):
            snapshot = ChatHistorySnapshot(
                session_key=f"eval:{entry.question_id}",
                messages=list(history),
                current_user_message=user_turn.content,
            )
            update = await background._analyze(snapshot)
            if update:
                await prompt_holder.write(update)

            history.append({"role": "user", "content": user_turn.content})
            history.append({"role": "assistant", "content": assistant_turn.content})
            turns_replayed += 2

            if verbose:
                marker = " [EVIDENCE]" if user_turn.has_answer or assistant_turn.has_answer else ""
                logger.info(
                    "[EVAL] replayed turn %d/%d%s",
                    turns_replayed,
                    entry.total_turns,
                    marker,
                )

    # --- Final question: persona answers using accumulated memory ---
    # Always pass the last session's history so the question has conversational
    # context, mirroring production where a question arrives mid-session.
    # When clear_history_between_sessions is True the `history` list already
    # contains only the most recent session's turns (prior sessions were
    # cleared at session boundaries above).
    context = ContextBuilder(workspace)
    ephemeral_hint = ""
    if update := prompt_holder.read():
        ephemeral_hint = update.ephemeral_hint or ""

    messages = context.build_persona_messages(
        history=history,
        current_message=entry.question,
        ephemeral_hint=ephemeral_hint,
    )
    response = await provider.chat(
        messages=messages,
        model=model,
        temperature=0.3,
        max_tokens=1024,
    )
    model_answer = (response.content or "").strip()

    # --- Judge ---
    memory_snap = read_memory_snapshot(workspace)
    judge_result = await llm_judge(
        question=entry.question,
        expected=entry.answer,
        actual=model_answer,
        provider=provider,
        model=judge_model,
    )

    return EvalResult(
        question_id=entry.question_id,
        question_type=entry.question_type,
        question=entry.question,
        expected_answer=entry.answer,
        model_answer=model_answer,
        correct=judge_result.correct,
        judge_detail=judge_result.detail,
        memory_snapshot=memory_snap,
        turns_replayed=turns_replayed,
        sessions_count=len(entry.sessions),
        elapsed_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    entries: list[EvalEntry],
    provider: LLMProvider,
    *,
    eval_workspace_base: Path,
    results_file: Path | None = None,
    model: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    clear_history_between_sessions: bool = True,
    keep_workspace: bool = True,
    verbose: bool = False,
) -> list[EvalResult]:
    """Run evaluation on multiple entries sequentially."""
    results: list[EvalResult] = []

    for i, entry in enumerate(entries):
        logger.info(
            "[EVAL] (%d/%d) %s [%s]",
            i + 1, len(entries), entry.question_id, entry.question_type,
        )
        result = await run_single_entry(
            entry,
            provider,
            eval_workspace_base=eval_workspace_base,
            model=model,
            judge_model=judge_model,
            clear_history_between_sessions=clear_history_between_sessions,
            keep_workspace=keep_workspace,
            verbose=verbose,
        )
        results.append(result)

        status = "PASS" if result.correct else "FAIL"
        logger.info(
            "[EVAL] %s %s (%.1fs) — Q: %s",
            status, entry.question_id, result.elapsed_seconds,
            entry.question[:60],
        )

        if results_file:
            _append_result(results_file, result)

    return results


def _append_result(path: Path, result: EvalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: list[EvalResult]) -> dict[str, Any]:
    """Produce an aggregate summary from eval results."""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.error)

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_type.setdefault(r.question_type, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if r.correct:
            bucket["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
        "by_type": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for k, v in sorted(by_type.items())
        },
    }
