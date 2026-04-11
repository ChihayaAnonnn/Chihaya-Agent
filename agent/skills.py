"""Skills loader for agent capabilities."""

from pathlib import Path


class SkillsLoader:
    """
    Loader for agent skills.
    Skills are markdown files (SKILL.md) that teach the agent how to use tools.
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None) -> None:
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir

    def list_skills(self) -> list[dict[str, str]]:
        """List all available skills (name, path, source)."""
        skills = []
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({
                            "name": skill_dir.name,
                            "path": str(skill_file),
                            "source": "workspace",
                        })
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(
                        s["name"] == skill_dir.name for s in skills
                    ):
                        skills.append({
                            "name": skill_dir.name,
                            "path": str(skill_file),
                            "source": "builtin",
                        })
        return skills

    def load_skill(self, name: str) -> str | None:
        """Load a skill by name."""
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")
        return None

    def build_skills_summary(self) -> str:
        """Build a summary of all skills (name, path)."""
        all_skills = self.list_skills()
        if not all_skills:
            return ""
        lines = ["<skills>"]
        for s in all_skills:
            lines.append(f'  <skill name="{s["name"]}" path="{s["path"]}"/>')
        lines.append("</skills>")
        return "\n".join(lines)
