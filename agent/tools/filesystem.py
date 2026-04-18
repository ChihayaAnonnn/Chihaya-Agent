"""File system tools: read, write."""

import difflib
import logging
from pathlib import Path
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

MEMORY_WRITE_HARD_LIMIT_TOKENS = 1500


def _is_memory_md(path: Path) -> bool:
    """True when the resolved path points at memory/MEMORY.md inside a workspace."""
    parts = [p.lower() for p in path.parts]
    return (
        path.name.upper() == "MEMORY.MD"
        and len(parts) >= 2
        and parts[-2] == "memory"
    )


def _roughly_count_tokens(text: str) -> int:
    """Token estimator shared with memory store; tiktoken when available."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text) / 2.5)


def _resolve_path(
    path: str, workspace: Path | None = None, allowed_dir: Path | None = None
) -> Path:
    """Resolve path against workspace and enforce directory restriction."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


class ReadFileTool(Tool):
    """Tool to read file contents."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None) -> None:
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The file path to read"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            return file_path.read_text(encoding="utf-8")
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Tool to write content to a file."""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None) -> None:
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)

            # Guard: MEMORY.md has a hard token budget. Refuse oversized writes
            # and push the model to regenerate a leaner version.
            if _is_memory_md(file_path):
                tokens = _roughly_count_tokens(content)
                if tokens > MEMORY_WRITE_HARD_LIMIT_TOKENS:
                    return (
                        f"Error: MEMORY.md write rejected — content is "
                        f"{tokens} tokens, hard limit is "
                        f"{MEMORY_WRITE_HARD_LIMIT_TOKENS}. Trim outdated items "
                        f"(especially `## 最近关键决策`) and try again."
                    )

            file_path.parent.mkdir(parents=True, exist_ok=True)

            old_content = ""
            if file_path.exists():
                try:
                    old_content = file_path.read_text(encoding="utf-8")
                except Exception:
                    pass

            file_path.write_text(content, encoding="utf-8")

            if old_content != content:
                diff_lines = list(difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    lineterm="",
                ))
                if diff_lines:
                    logger.info("[FILE_CHANGE] %s\n%s", path, "".join(diff_lines))

            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
