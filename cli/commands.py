"""CLI commands for GeneralAgentTemplate."""

import asyncio
from datetime import datetime
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.rule import Rule

from bus.events import ChatHistorySnapshot, InboundMessage
from bus.queue import MessageBus
from cron.service import CronService
from cron.types import CronSchedule
from heartbeat.service import HeartbeatService
from dotenv import load_dotenv

from providers.qwen import QwenProvider

load_dotenv()


def _qwen_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        console.print("[red]DASHSCOPE_API_KEY not set. Copy .env.example to .env and fill it in.[/red]")
        raise typer.Exit(1)
    return key
from session.manager import SessionManager

from agent.background import BackgroundAgent, PromptHolder
from agent.proactive import IdleMonitor
from agent.context import ContextBuilder
from agent.loop import AgentLoop
from agent.memory import MemoryStore
from agent.runner import run_agentic_loop
from agent.tools.filesystem import ReadFileTool, WriteFileTool
from agent.tools.registry import ToolRegistry

app = typer.Typer(name="agent", help="GeneralAgentTemplate - CLI Agent")
console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def _setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
    ]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        handlers.append(file_handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def _run_log_file(mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "logs" / f"{ts}_{mode}.log"


def _get_workspace() -> Path:
    return Path.cwd() / "workspace"


def _get_user_workspace(user_name: str) -> Path:
    """Return ``workspaces/{user_name}/``, bootstrapping from templates if new."""
    workspace = Path.cwd() / "workspaces" / user_name
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
    console.print(f"[green]✓ Created workspace for user [bold]{user_name}[/bold][/green]")
    return workspace


# ---------------------------------------------------------------------------
# Onboard
# ---------------------------------------------------------------------------


@app.command()
def onboard():
    """Initialize workspace."""
    workspace = _get_workspace()
    workspace.mkdir(parents=True, exist_ok=True)

    templates = {
        "AGENTS.md": """# Agent Instructions

You are a helpful AI assistant. Be concise and friendly.

## Guidelines
- Use tools when needed (read_file, write_file, spawn)
- Remember important info in memory/MEMORY.md
""",
        "SOUL.md": """# Soul

I am a minimal AI assistant for learning agent concepts.
""",
        "USER.md": """# User

User preferences go here.
""",
        "HEARTBEAT.md": """# Heartbeat Tasks

Add tasks here for the agent to check periodically.
""",
    }
    for name, content in templates.items():
        p = workspace / name
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            console.print(f"  [green]✓[/green] [dim]{name}[/dim]")

    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    (memory_dir / "MEMORY.md").write_text("# Long-term Memory\n\n", encoding="utf-8")
    (memory_dir / "HISTORY.md").write_text("", encoding="utf-8")
    (workspace / "skills").mkdir(exist_ok=True)
    console.print(f"\n[bold green]✓[/bold green] Workspace ready at [cyan]{workspace}[/cyan]")


# ---------------------------------------------------------------------------
# Roleplay (dual-LLM: persona + background)
# ---------------------------------------------------------------------------


@app.command()
def roleplay(
    message: str = typer.Option(None, "--message", "-m", help="Single message"),
    user: str = typer.Option("default", "--user", "-u", help="User name (workspace per user)"),
    session_id: str = typer.Option(None, "--session", "-s", help="Session ID"),
    logs: bool = typer.Option(False, "--logs", help="Show runtime logs"),
    proactive: bool = typer.Option(
        True, "--proactive/--no-proactive", help="Enable idle-triggered proactive messages"
    ),
    idle_seconds: int = typer.Option(
        300, "--idle-seconds", help="Seconds of inactivity before the agent may speak up"
    ),
):
    """Interact with the agent (dual-LLM: fast persona + heavy background)."""
    workspace = _get_user_workspace(user)
    _setup_logging(verbose=logs, log_file=_run_log_file("roleplay"))
    if session_id is None:
        session_id = f"cli:{user}"

    bus = MessageBus()
    persona_provider = QwenProvider(api_key=_qwen_api_key())
    background_provider = QwenProvider(api_key=_qwen_api_key())
    background_queue: asyncio.Queue[ChatHistorySnapshot] = asyncio.Queue()
    prompt_holder = PromptHolder()
    session_manager = SessionManager(workspace)

    idle_monitor: IdleMonitor | None = None
    if proactive and not message:  # only in interactive mode
        idle_monitor = IdleMonitor(
            bus=bus,
            provider=persona_provider,
            workspace=workspace,
            session_manager=session_manager,
            idle_threshold_s=idle_seconds,
        )

    agent_loop = AgentLoop(
        bus=bus,
        provider=persona_provider,
        workspace=workspace,
        session_manager=session_manager,
        persona_provider=persona_provider,
        background_queue=background_queue,
        prompt_holder=prompt_holder,
        idle_monitor=idle_monitor,
        ephemeral=True,
    )
    background_agent = BackgroundAgent(
        queue=background_queue,
        provider=background_provider,
        prompt_holder=prompt_holder,
        workspace=workspace,
        session_manager=session_manager,
    )

    def _print_response(content: str) -> None:
        console.print()
        console.rule("[bold cyan]Agent[/bold cyan]", style="cyan dim")
        console.print(Markdown(content))
        console.print()

    if message:
        async def run_once() -> None:
            resp = await agent_loop.process_direct(message, session_id)
            _print_response(resp)
            if session := session_manager.get_cached(session_id):
                session_manager.archive_session(session)

        asyncio.run(run_once())
    else:
        console.print(Rule("[bold blue]Roleplay[/bold blue]  [dim]dual-LLM · fast persona + deep background[/dim]", style="blue dim"))
        console.print("[dim]  type [bold]exit[/bold] or Ctrl+C to quit[/dim]\n")
        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        async def run_interactive() -> None:
            loop_task = asyncio.create_task(agent_loop.run())
            bg_task = asyncio.create_task(background_agent.run())
            idle_task: asyncio.Task | None = None
            if idle_monitor is not None:
                idle_task = asyncio.create_task(idle_monitor.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []

            async def consume_outbound() -> None:
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            console.print(f"  [yellow]⟳[/yellow] [dim]{msg.content}[/dim]")
                        elif msg.metadata.get("proactive"):
                            _print_response(f"*[主动消息]* {msg.content}")
                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            _print_response(msg.content)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(consume_outbound())
            try:
                while True:
                    try:
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: console.input("[bold green]You[/bold green] › ")
                        )
                        user_input = user_input.strip()
                        if not user_input:
                            continue
                        if user_input.lower() in EXIT_COMMANDS:
                            console.print("\n[bold yellow]Goodbye![/bold yellow]")
                            break
                        turn_done.clear()
                        turn_response.clear()
                        await bus.publish_inbound(
                            InboundMessage(
                                channel=cli_channel,
                                sender_id="user",
                                chat_id=cli_chat_id,
                                content=user_input,
                            )
                        )
                        await turn_done.wait()
                        if turn_response:
                            _print_response(turn_response[0])
                    except (KeyboardInterrupt, EOFError):
                        console.print("\n[bold yellow]Goodbye![/bold yellow]")
                        break
            finally:
                if session := session_manager.get_cached(session_id):
                    session_manager.archive_session(session)
                agent_loop.stop()
                background_agent.stop()
                if idle_monitor is not None:
                    idle_monitor.stop()
                outbound_task.cancel()
                tasks: list[asyncio.Task] = [loop_task, bg_task, outbound_task]
                if idle_task is not None:
                    idle_task.cancel()
                    tasks.append(idle_task)
                await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(run_interactive())


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------


def _make_agentic_runner(workspace: Path) -> Any:
    """Build a standalone agentic runner (tools + context) for cron/heartbeat."""
    provider = QwenProvider(api_key=_qwen_api_key())
    context = ContextBuilder(workspace)
    session_manager = SessionManager(workspace)
    tools = ToolRegistry()
    tools.register(ReadFileTool(workspace=workspace))
    tools.register(WriteFileTool(workspace=workspace))

    async def run(content: str, session_key: str = "cli:direct") -> str:
        session = session_manager.get_or_create(session_key)
        history = session.get_history(max_messages=20)
        messages = context.build_messages(
            history=history,
            current_message=content,
        )
        final_content, _ = await run_agentic_loop(
            provider=provider,
            tools=tools,
            messages=messages,
            model=provider.get_default_model(),
        )
        result = final_content or ""
        session.add_message("user", content)
        session.add_message("assistant", result)
        session_manager.save(session)
        return result

    return run


cron_app = typer.Typer(help="Scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all_jobs: bool = typer.Option(False, "--all", "-a", help="Include disabled"),
):
    """List scheduled jobs."""
    store_path = _get_workspace().parent / ".agent" / "cron" / "jobs.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    service = CronService(store_path)
    jobs = service.list_jobs(include_disabled=all_jobs)
    if not jobs:
        console.print("No scheduled jobs.")
        return
    for j in jobs:
        sched = f"every {(j.schedule.every_ms or 0) // 1000}s" if j.schedule.kind == "every" else "one-time"
        console.print(f"  [bold yellow]{j.id}[/bold yellow]  [white]{j.name}[/white]  [cyan dim]{sched}[/cyan dim]")
        if j.state.next_run_at_ms:
            import time
            ts = j.state.next_run_at_ms / 1000
            console.print(f"    [dim]next: {time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))}[/dim]")


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n"),
    message: str = typer.Option(..., "--message", "-m"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    at: str = typer.Option(None, "--at", help="Run once at ISO time"),
):
    """Add a scheduled job."""
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Specify --every or --at[/red]")
        raise typer.Exit(1)
    store_path = _get_workspace().parent / ".agent" / "cron" / "jobs.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    service = CronService(store_path)
    job = service.add_job(name=name, schedule=schedule, message=message)
    console.print(f"[green]Added job '{job.name}' ({job.id})[/green]")


@cron_app.command("remove")
def cron_remove(job_id: str = typer.Argument(..., help="Job ID")):
    """Remove a job."""
    store_path = _get_workspace().parent / ".agent" / "cron" / "jobs.json"
    service = CronService(store_path)
    if service.remove_job(job_id):
        console.print(f"[green]Removed {job_id}[/green]")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Manually run a job."""
    _setup_logging()
    workspace = _get_workspace()
    store_path = workspace.parent / ".agent" / "cron" / "jobs.json"
    run_agent = _make_agentic_runner(workspace)
    service = CronService(store_path)
    result_holder: list[str] = []

    async def on_job(job) -> str | None:
        r = await run_agent(job.payload.message, session_key=f"cron:{job.id}")
        result_holder.append(r)
        return r

    service.on_job = on_job
    if asyncio.run(service.run_job(job_id, force=force)):
        console.print("[green]Job executed[/green]")
        if result_holder:
            console.print(Markdown(result_holder[0]))
    else:
        console.print(f"[red]Failed to run {job_id}[/red]")


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


@app.command()
def heartbeat():
    """Manually trigger a heartbeat."""
    _setup_logging()
    workspace = _get_workspace()
    if not workspace.exists():
        console.print("[red]Run 'agent onboard' first[/red]")
        raise typer.Exit(1)
    run_agent = _make_agentic_runner(workspace)
    service = HeartbeatService(
        workspace=workspace,
        on_heartbeat=lambda prompt: run_agent(prompt, session_key="heartbeat"),
    )
    console.print(Rule("[bold magenta]Heartbeat[/bold magenta]", style="magenta dim"))
    result = asyncio.run(service.trigger_now())
    if result:
        console.print(Markdown(result))
    else:
        console.print("[dim]No heartbeat handler or HEARTBEAT.md empty.[/dim]")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

eval_app = typer.Typer(help="Evaluation commands")
app.add_typer(eval_app, name="eval")

_EVAL_RESULTS_DIR = Path.cwd() / "eval" / "results"


@eval_app.command("run")
def eval_run(
    dataset: str = typer.Option("longmemeval", "--dataset", "-d", help="Dataset name"),
    entry_id: str = typer.Option(None, "--entry-id", "-e", help="Run a single entry by ID"),
    filter_type: str = typer.Option(None, "--filter", "-f", help="Filter by question type"),
    single_session: bool = typer.Option(False, "--single-session", help="Only single-session entries"),
    limit: int = typer.Option(None, "--limit", "-n", help="Max entries to run"),
    judge_model: str = typer.Option("qwen-turbo", "--judge-model", "-j", help="Model for LLM judge"),
    keep_workspaces: bool = typer.Option(True, "--keep/--no-keep", help="Keep eval workspaces for inspection (default: keep)"),
    logs: bool = typer.Option(False, "--logs", help="Show runtime logs"),
):
    """Run evaluation against a dataset."""
    from eval.dataset import load_longmemeval, load_memory_recall
    from eval.runner import run_batch, summarize

    if dataset == "longmemeval":
        entries = load_longmemeval(
            entry_id=entry_id,
            filter_type=filter_type,
            single_session_only=single_session,
            limit=limit,
        )
    elif dataset == "memory_recall":
        entries = load_memory_recall(limit=limit)
        if entry_id:
            entries = [e for e in entries if e.question_id == entry_id]
        if filter_type:
            entries = [e for e in entries if e.question_type == filter_type]
    else:
        _setup_logging(verbose=logs)
        console.print(
            f"[red]Unknown dataset: {dataset}. "
            f"Supported: longmemeval, memory_recall[/red]"
        )
        raise typer.Exit(1)
    if not entries:
        _setup_logging(verbose=logs)
        console.print("[yellow]No entries matched the filters.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Loaded {len(entries)} entries[/bold] from {dataset}")

    provider = QwenProvider(api_key=_qwen_api_key())
    eval_base = Path.cwd() / "eval" / ".workspaces"

    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    results_file = _EVAL_RESULTS_DIR / f"{dataset}_{ts}.jsonl"
    _EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _run_log_file("eval")
    _setup_logging(verbose=logs, log_file=log_file)

    results = asyncio.run(run_batch(
        entries,
        provider,
        eval_workspace_base=eval_base,
        results_file=results_file,
        judge_model=judge_model,
        keep_workspace=keep_workspaces,
        verbose=logs,
    ))

    summary = summarize(results)
    console.print()
    console.print(Rule("[bold]Results[/bold]"))
    console.print(f"  Total: {summary['total']}  Correct: {summary['correct']}  "
                  f"Accuracy: {summary['accuracy']:.1%}  Errors: {summary['errors']}")
    if summary["by_type"]:
        console.print()
        for qtype, stats in summary["by_type"].items():
            console.print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.0%})")
    console.print(f"\n  Results saved to [cyan]{results_file}[/cyan]")


@eval_app.command("results")
def eval_results(
    latest: bool = typer.Option(True, "--latest/--all", help="Show only latest run"),
    compare: int = typer.Option(
        0, "--compare", "-c",
        help="Compare the latest N runs side-by-side (e.g. --compare 2)",
    ),
):
    """Show evaluation results."""
    if not _EVAL_RESULTS_DIR.exists():
        console.print("[dim]No results yet. Run 'agent eval run' first.[/dim]")
        return

    files = sorted(_EVAL_RESULTS_DIR.glob("*.jsonl"))
    if not files:
        console.print("[dim]No results yet.[/dim]")
        return

    import json as _json
    from eval.runner import EvalResult, summarize

    def _load(path: Path) -> list[EvalResult]:
        items: list[EvalResult] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            items.append(EvalResult(**_json.loads(line)))
        return items

    if compare and compare >= 2:
        runs = files[-compare:]
        console.print(Rule(f"[bold]Comparing {len(runs)} runs[/bold]"))
        summaries: list[tuple[Path, dict[str, Any]]] = []
        for f in runs:
            summaries.append((f, summarize(_load(f))))

        header = f"{'Run':<32}{'Total':>8}{'Correct':>10}{'Acc':>8}{'Errors':>8}"
        console.print(f"[bold]{header}[/bold]")
        for f, s in summaries:
            console.print(
                f"{f.name:<32}{s['total']:>8}{s['correct']:>10}"
                f"{s['accuracy']:>7.1%}{s['errors']:>8}"
            )

        # per-type delta between first and last run
        first, last = summaries[0][1], summaries[-1][1]
        types = sorted(set(first["by_type"]) | set(last["by_type"]))
        if types:
            console.print("\n[bold]Per-type Δ (last − first)[/bold]")
            for t in types:
                a = first["by_type"].get(t, {"accuracy": 0.0})["accuracy"]
                b = last["by_type"].get(t, {"accuracy": 0.0})["accuracy"]
                delta = b - a
                sign = "+" if delta >= 0 else ""
                color = "green" if delta >= 0 else "red"
                console.print(
                    f"  {t:<20} {a:>6.1%} → {b:>6.1%}  "
                    f"[{color}]{sign}{delta*100:.1f}pp[/{color}]"
                )
        return

    targets = [files[-1]] if latest else files
    for f in targets:
        console.print(Rule(f"[bold]{f.name}[/bold]"))
        results = _load(f)

        summary = summarize(results)
        console.print(f"  Total: {summary['total']}  Correct: {summary['correct']}  "
                      f"Accuracy: {summary['accuracy']:.1%}")
        for qtype, stats in summary["by_type"].items():
            console.print(f"    {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.0%})")

        # Show failures
        failures = [r for r in results if not r.correct]
        if failures:
            console.print(f"\n  [yellow]Failures ({len(failures)}):[/yellow]")
            for r in failures[:10]:
                console.print(f"    {r.question_id}: Q={r.question[:50]}...")
                console.print(f"      Expected: {r.expected_answer}")
                console.print(f"      Got: {r.model_answer[:80]}")


if __name__ == "__main__":
    app()
