"""Context builder for assembling agent prompts."""

import platform
from pathlib import Path
from typing import Any

from agent.memory import MemoryStore
from agent.skills import SkillsLoader


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from bootstrap files, memory, and skills."""
        parts = []
        parts.append(self._get_identity())
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(
                f"# Skills\n\nTo use a skill, read its SKILL.md using read_file.\n\n{skills_summary}"
            )
        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        from datetime import datetime
        import time as _time

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = (
            f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, "
            f"Python {platform.python_version()}"
        )
        return f"""# GeneralAgentTemplate

You are a helpful AI assistant with access to tools:
- read_file: Read file contents
- write_file: Write content to files
- spawn: Spawn subagents for background tasks

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
{workspace_path}
- Memory: {workspace_path}/memory/MEMORY.md
- History: {workspace_path}/memory/HISTORY.md
- Skills: {workspace_path}/skills/{{name}}/SKILL.md

Be concise and helpful. Use tools when needed."""

    def _load_bootstrap_files(self) -> str:
        parts = []
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def apply_persona_addendum(base_prompt: str, addendum: str) -> str:
        """Append persona guidance from background agent to system prompt."""
        if not addendum or not addendum.strip():
            return base_prompt
        return f"{base_prompt}\n\n## Background Guidance\n{addendum.strip()}"

    # ------------------------------------------------------------------
    # Persona lane: SOUL.md + USER.md + memory + history + ephemeral hint
    # ------------------------------------------------------------------

    PERSONA_CONTEXT_FILES = ["SOUL.md", "USER.md"]

    def build_persona_prompt(self) -> str:
        """
        System prompt for the persona lane.
        Loads SOUL.md, USER.md, and long-term memory so the persona
        benefits from context files the background agent maintains.
        """
        parts = []
        for filename in self.PERSONA_CONTEXT_FILES:
            path = self.workspace / filename
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
        if not parts:
            parts.append("You are a helpful AI assistant. Be concise and friendly.")

        mem = self.memory.get_memory_context()
        if mem:
            parts.append(mem)

        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        parts.append(f"Current time: {now} ({tz})")
        return "\n\n".join(parts)

    def build_persona_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        channel: str | None = None,
        chat_id: str | None = None,
        ephemeral_hint: str = "",
    ) -> list[dict[str, Any]]:
        """
        Build message list for the persona lane's single LLM call.
        Applies optional ephemeral hint from the background agent.
        """
        system_prompt = self.build_persona_prompt()
        if ephemeral_hint and ephemeral_hint.strip():
            system_prompt += f"\n\n## Background Hint\n{ephemeral_hint.strip()}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_message})
        return messages

    # ------------------------------------------------------------------
    # Background lane (full): tools, memory, skills, bootstrap files
    # ------------------------------------------------------------------

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        persona_addendum: str = "",
    ) -> list[dict[str, Any]]:
        """Build the complete message list for the background agent's LLM call."""
        messages = []
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        system_prompt = self.apply_persona_addendum(system_prompt, persona_addendum)
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": current_message})
        return messages

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)
        return messages
