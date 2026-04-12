"""Workspace management routes."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.deps import get_workspace

router = APIRouter()

# File extensions allowed for content preview
_READABLE_SUFFIXES = {".md", ".txt", ".json", ".jsonl"}
# Directories to skip when listing files
_SKIP_DIRS = {"__pycache__", ".git"}


@router.post("/init")
async def workspace_init(user: str = Query("default")) -> dict:
    """Initialize workspace for a user (equivalent to `agent onboard`)."""
    workspace = get_workspace(user)

    templates = {
        "AGENTS.md": "# Agent Instructions\n\nYou are a helpful AI assistant. Be concise and friendly.\n\n## Guidelines\n- Use tools when needed (read_file, write_file, spawn)\n- Remember important info in memory/MEMORY.md\n",
        "SOUL.md": "# Soul\n\nI am a minimal AI assistant for learning agent concepts.\n",
        "USER.md": "# User\n\nUser preferences go here.\n",
        "HEARTBEAT.md": "# Heartbeat Tasks\n\nAdd tasks here for the agent to check periodically.\n",
    }
    created: list[str] = []
    for name, content in templates.items():
        p = workspace / name
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            created.append(name)

    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("# Long-term Memory\n\n", encoding="utf-8")
        created.append("memory/MEMORY.md")
    history_file = memory_dir / "HISTORY.md"
    if not history_file.exists():
        history_file.write_text("", encoding="utf-8")
        created.append("memory/HISTORY.md")
    (workspace / "skills").mkdir(exist_ok=True)

    return {"workspace": str(workspace), "created": created}


@router.get("/files")
async def workspace_files(user: str = Query("default")) -> dict:
    """
    Return a recursive file tree of the user's workspace.

    Each node: {"name": str, "path": str, "type": "file"|"dir", "children": [...]}
    path is relative to workspace root, using forward slashes.
    """
    workspace = get_workspace(user)
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Workspace for user '{user}' not found.")

    def _build_tree(directory: Path, relative: Path) -> list[dict]:
        nodes: list[dict] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return nodes
        for entry in entries:
            if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                continue
            rel_path = (relative / entry.name).as_posix()
            if entry.is_dir():
                nodes.append({
                    "name": entry.name,
                    "path": rel_path,
                    "type": "dir",
                    "children": _build_tree(entry, relative / entry.name),
                })
            else:
                nodes.append({
                    "name": entry.name,
                    "path": rel_path,
                    "type": "file",
                    "readable": entry.suffix in _READABLE_SUFFIXES,
                })
        return nodes

    tree = _build_tree(workspace, Path(""))
    return {"user": user, "tree": tree}


@router.get("/files/{file_path:path}")
async def workspace_file_content(file_path: str, user: str = Query("default")) -> dict:
    """
    Return the content of a single file in the user's workspace.

    file_path is relative to workspace root (e.g. "memory/MEMORY.md").
    """
    workspace = get_workspace(user)
    target = (workspace / file_path).resolve()

    # Prevent path traversal outside workspace
    try:
        target.relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_path}' not found.")
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"'{file_path}' is a directory.")
    if target.suffix not in _READABLE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"File type '{target.suffix}' is not readable.")
    if target.stat().st_size > 512 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>512KB).")

    content = target.read_text(encoding="utf-8", errors="replace")
    return {"path": file_path, "content": content, "suffix": target.suffix}
