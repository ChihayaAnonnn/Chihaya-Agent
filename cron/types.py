"""Cron types."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """Schedule definition for a cron job."""
    kind: Literal["at", "every"]
    at_ms: int | None = None  # For "at": timestamp in ms
    every_ms: int | None = None  # For "every": interval in ms


@dataclass
class CronPayload:
    """What to do when the job runs."""
    kind: Literal["agent_turn"] = "agent_turn"
    message: str = ""
    deliver: bool = False
    channel: str | None = None
    to: str | None = None


@dataclass
class CronJobState:
    """Runtime state of a job."""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None


@dataclass
class CronJob:
    """A scheduled job."""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass
class CronStore:
    """Persistent store for cron jobs."""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
