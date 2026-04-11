"""Session management for conversation history."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from utils import ensure_dir, safe_filename

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    A conversation session.
    Stores messages in JSONL format for persistence.
    """

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Get recent messages in LLM format."""
        out: list[dict[str, Any]] = []
        for m in self.messages[-max_messages:]:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """Manages conversation sessions stored as JSONL files."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.sessions_dir = ensure_dir(workspace / "sessions")
        self._cache: dict[str, Session] = {}
        self._ephemeral_keys: set[str] = set()

    def _get_session_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str, *, ephemeral: bool = False) -> Session:
        """Get an existing session or create a new one."""
        if key in self._cache:
            return self._cache[key]
        if ephemeral:
            session = Session(key=key)
            self._ephemeral_keys.add(key)
        else:
            session = self._load(key)
            if session is None:
                session = Session(key=key)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        path = self._get_session_path(key)
        if not path.exists():
            return None
        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = (
                            datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else None
                        )
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as e:
            logger.warning("Failed to load session %s: %s", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk. Skips ephemeral sessions."""
        if session.key in self._ephemeral_keys:
            return
        path = self._get_session_path(session.key)
        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)
        self._ephemeral_keys.discard(key)

    def get_cached(self, key: str) -> Session | None:
        """Return cached session if present, else None."""
        return self._cache.get(key)

    def archive_session(self, session: Session) -> Path | None:
        """Archive session to sessions/{key}_{timestamp}.jsonl. Returns path or None if no messages."""
        if not session.messages:
            return None
        safe_key = safe_filename(session.key.replace(":", "_"))
        ts = session.updated_at.strftime("%Y%m%d_%H%M%S")
        path = self.sessions_dir / f"{safe_key}_{ts}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        logger.info("[SESSION] archived to %s", path)
        return path
