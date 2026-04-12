"""FastAPI shared dependencies."""

import os
import shutil
from pathlib import Path

from fastapi import Header, HTTPException


def get_api_key(x_dashscope_key: str = Header(None)) -> str:
    key = x_dashscope_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise HTTPException(status_code=401, detail="Provide X-Dashscope-Key header")
    return key


def get_workspace(user: str = "default") -> Path:
    """Return workspaces/{user}/, bootstrapping from templates if new."""
    workspace = Path.cwd() / "workspaces" / user
    if workspace.exists():
        return workspace

    workspace.mkdir(parents=True)
    template_dir = Path.cwd() / "context_file_template"
    if template_dir.exists():
        for f in template_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, workspace / f.name)
    for sub in ("sessions", "memory", "skills"):
        (workspace / sub).mkdir(exist_ok=True)
    (workspace / "memory" / "MEMORY.md").write_text("# Long-term Memory\n\n", encoding="utf-8")
    (workspace / "memory" / "HISTORY.md").write_text("", encoding="utf-8")
    return workspace
