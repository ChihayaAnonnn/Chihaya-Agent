"""Dataset loaders for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class EvalTurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass
class EvalSession:
    turns: list[EvalTurn] = field(default_factory=list)
    date: str | None = None


@dataclass
class EvalEntry:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str | None = None
    sessions: list[EvalSession] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    @property
    def is_single_session(self) -> bool:
        return len(self.sessions) == 1


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

_LONGMEMEVAL_PATH = Path(__file__).parent / "thirdparty" / "LongMemEval" / "data" / "longmemeval_oracle.json"


def _load_longmemeval_entry(raw: dict) -> EvalEntry:
    dates = raw.get("haystack_dates", [])
    sessions: list[EvalSession] = []
    for idx, raw_session in enumerate(raw.get("haystack_sessions", [])):
        turns = [
            EvalTurn(
                role=t["role"],
                content=t["content"],
                has_answer=t.get("has_answer", False),
            )
            for t in raw_session
        ]
        sessions.append(EvalSession(
            turns=turns,
            date=dates[idx] if idx < len(dates) else None,
        ))
    return EvalEntry(
        question_id=raw["question_id"],
        question_type=raw["question_type"],
        question=raw["question"],
        answer=raw["answer"],
        question_date=raw.get("question_date"),
        sessions=sessions,
    )


def load_longmemeval(
    path: Path | None = None,
    *,
    filter_type: str | None = None,
    single_session_only: bool = False,
    entry_id: str | None = None,
    limit: int | None = None,
) -> list[EvalEntry]:
    """Load LongMemEval oracle dataset with optional filters."""
    data_path = path or _LONGMEMEVAL_PATH
    with open(data_path, encoding="utf-8") as f:
        raw_data: list[dict] = json.load(f)

    entries: list[EvalEntry] = []
    for raw in raw_data:
        entry = _load_longmemeval_entry(raw)

        if entry_id and entry.question_id != entry_id:
            continue
        if filter_type and entry.question_type != filter_type:
            continue
        if single_session_only and not entry.is_single_session:
            continue

        entries.append(entry)
        if limit and len(entries) >= limit:
            break

    return entries


def iter_turn_pairs(session: EvalSession) -> Iterator[tuple[EvalTurn, EvalTurn]]:
    """Yield (user, assistant) turn pairs from a session."""
    turns = session.turns
    for i in range(0, len(turns) - 1, 2):
        if turns[i].role == "user" and turns[i + 1].role == "assistant":
            yield turns[i], turns[i + 1]
