"""Utility functions."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

OBS_LOGGER = logging.getLogger("agent.obs")


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        name = name.replace(char, "_")
    return name.strip()


def new_trace_id() -> str:
    """Generate a short trace id for a single turn."""
    return uuid.uuid4().hex[:12]


def now_ms() -> int:
    """Current time in milliseconds (monotonic is not used: events are wall-clock)."""
    return int(time.time() * 1000)


def log_event(event: str, **fields: Any) -> None:
    """Emit a single structured JSON log line.

    Use for metrics / cross-cutting observability (latency, tokens, trace_id).
    The main application loggers keep their existing human-readable format.
    """
    payload: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
    }
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    try:
        OBS_LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except (TypeError, ValueError):
        OBS_LOGGER.info(json.dumps({"event": event, "error": "unserializable_fields"}))
