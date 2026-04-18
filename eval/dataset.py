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


# ---------------------------------------------------------------------------
# Custom memory_recall dataset (hand-authored, JSONL)
# ---------------------------------------------------------------------------

_MEMORY_RECALL_PATH = Path(__file__).parent / "datasets" / "memory_recall.jsonl"

_GAP_FILLER = {
    "user": "随便再聊聊吧，最近有什么新技术值得关注？",
    "assistant": "最近 WASI 和 MCP 都挺热。",
}


def load_memory_recall(
    path: Path | None = None,
    *,
    limit: int | None = None,
) -> list[EvalEntry]:
    """Load the self-authored memory_recall.jsonl dataset.

    Each line becomes a single-session EvalEntry whose session is:
    ``setup_turns + gap_turns*N filler turns``. The ``expected_keywords``
    field is stashed in ``question_type`` payload via a tagging convention:
    callers that want the keywords should use :func:`load_memory_recall_raw`.
    """
    data_path = path or _MEMORY_RECALL_PATH
    entries: list[EvalEntry] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            entries.append(_build_memory_recall_entry(raw))
            if limit and len(entries) >= limit:
                break
    return entries


def load_memory_recall_raw(path: Path | None = None) -> list[dict]:
    """Return raw dicts preserving ``expected_keywords`` / ``judge_prompt``."""
    data_path = path or _MEMORY_RECALL_PATH
    out: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_memory_recall_entry(raw: dict) -> EvalEntry:
    setup = raw.get("setup_turns", [])
    gap = int(raw.get("gap_turns", 0))

    turns: list[EvalTurn] = [
        EvalTurn(role=t["role"], content=t["content"], has_answer=True)
        for t in setup
    ]
    for _ in range(max(0, gap // 2)):
        turns.append(EvalTurn(role="user", content=_GAP_FILLER["user"]))
        turns.append(EvalTurn(role="assistant", content=_GAP_FILLER["assistant"]))

    return EvalEntry(
        question_id=raw["id"],
        question_type=raw.get("type", "memory_recall"),
        question=raw["question"],
        answer=", ".join(raw.get("expected_keywords", []))
        or raw.get("judge_prompt", ""),
        sessions=[EvalSession(turns=turns)],
    )
