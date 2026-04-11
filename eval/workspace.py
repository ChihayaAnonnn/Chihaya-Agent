"""Disposable eval workspace factory."""

from __future__ import annotations

import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _PROJECT_ROOT / "context_file_template"

CLEAN_MEMORY = "# Long-term Memory\n\n"


def create_eval_workspace(
    base_dir: Path,
    name: str,
    *,
    template_dir: Path | None = None,
) -> Path:
    """Create a fresh, isolated workspace for one eval entry.

    Returns the workspace path. Deletes any pre-existing workspace with the
    same name to ensure a clean slate.
    """
    workspace = base_dir / name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    src = template_dir or _TEMPLATE_DIR
    if src.exists():
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, workspace / f.name)

    for sub in ("sessions", "memory", "skills"):
        (workspace / sub).mkdir(exist_ok=True)

    (workspace / "memory" / "MEMORY.md").write_text(CLEAN_MEMORY, encoding="utf-8")
    (workspace / "memory" / "HISTORY.md").write_text("", encoding="utf-8")

    return workspace


def cleanup_eval_workspace(workspace: Path) -> None:
    """Remove an eval workspace."""
    if workspace.exists():
        shutil.rmtree(workspace)


def read_memory_snapshot(workspace: Path) -> str:
    """Read the current MEMORY.md contents from a workspace."""
    mem = workspace / "memory" / "MEMORY.md"
    return mem.read_text(encoding="utf-8") if mem.exists() else ""
