"""Integration tests for roleplay CLI."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

# Ensure workspace exists before importing cli (which may resolve paths)
_workspace = Path(__file__).resolve().parent.parent / "workspace"


class TestRoleplayCLI(unittest.TestCase):
    """CLI integration tests."""

    def setUp(self) -> None:
        self.orig_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        Path("workspace").mkdir(exist_ok=True)
        (Path("workspace") / "AGENTS.md").write_text("# Agent\n")
        (Path("workspace") / "SOUL.md").write_text("# Soul\n")
        (Path("workspace") / "USER.md").write_text("# User\n")
        (Path("workspace") / "memory").mkdir(exist_ok=True)
        (Path("workspace") / "memory" / "MEMORY.md").write_text("# Memory\n")
        (Path("workspace") / "memory" / "HISTORY.md").write_text("")
        (Path("workspace") / "skills").mkdir(exist_ok=True)
        (Path("workspace") / "sessions").mkdir(exist_ok=True)

    def tearDown(self) -> None:
        os.chdir(self.orig_cwd)

    def test_roleplay_single_message(self) -> None:
        """Roleplay command returns response for single message."""
        from typer.testing import CliRunner
        from cli.commands import app
        runner = CliRunner()
        result = runner.invoke(app, ["roleplay", "-m", "hello"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Agent", result.output)
        self.assertIn("Hello", result.output)
