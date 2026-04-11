# GeneralAgentTemplate — Dual-LLM Roleplay Agent

A learning project exploring agent architecture patterns. Built around a **dual-LLM roleplay system**: a fast persona model handles every turn; a background reasoning model runs concurrently, enriches context, and steers the persona's tone and behavior over time.

---

## How It Works

```
User input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  AgentLoop  (persona lane)                              │
│                                                         │
│  1. reads PromptHolder → ## Background Guidance         │
│  2. builds messages (SOUL + history + guidance)         │
│  3. single provider.chat() call → response              │
│  4. put_nowait(snapshot) → background queue             │
└────────────────────────┬────────────────────────────────┘
                         │ ChatHistorySnapshot (non-blocking)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  BackgroundAgent  (background lane)                     │
│                                                         │
│  run_agentic_loop() with tools (read_file, write_file)  │
│  → produces PersonaPromptUpdate                         │
│  → writes to PromptHolder (replace-on-write)            │
└─────────────────────────────────────────────────────────┘
```

### Concurrency model

Both lanes run as asyncio tasks in the same event loop. The persona never waits for the background — it fires a snapshot into a queue and responds immediately. The background processes concurrently and writes its result to `PromptHolder`. The persona picks it up on the **next** turn.

CLI `input()` runs via `run_in_executor` so the event loop stays live between user turns, allowing the background agent's HTTP callbacks to fire without waiting for the next keystroke.

### Turn-offset guidance flow

```
Turn N   → persona responds (uses N-1 background result)
           background analyzes turn N concurrently
Turn N+1 → persona injects N result as ## Background Guidance
```

### `PersonaPromptUpdate` fields

| Field | Purpose |
|---|---|
| `system_prompt_addendum` | Tone hints, character guidance — appended to system prompt |
| `injected_context` | Retrieved facts/data — prepended to user message |
| `conversation_directives` | Behavioral steering — what to emphasize or avoid |

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set your API key:

```bash
export QWEN_API_KEY=sk-...
```

## Usage

| Command | Description |
|---|---|
| `agent onboard` | Initialize shared workspace |
| `agent roleplay` | Interactive dual-LLM session (default user) |
| `agent roleplay -u alice` | Session under `workspaces/alice/` |
| `agent roleplay -u alice -m "hello"` | Single-message mode for a named user |
| `agent roleplay --logs` | Show runtime logs (`[AGENT]`, `[PERSONA]`, `[BACKGROUND]`, `[RUNNER]`) |
| `agent chat` | Simple single-lane chat (mock provider) |
| `agent cron add -n demo -m "Check status" -e 30` | Add cron job (every 30s) |
| `agent heartbeat` | Trigger heartbeat manually |

### Per-user workspaces

Each user gets an isolated workspace at `workspaces/{user_name}/`. On first use, files are copied from `context_file_template/` (SOUL.md, AGENTS.md, USER.md, etc.). Context loading, memory, and sessions are all scoped to the user's folder.

```
workspaces/
└── alice/
    ├── SOUL.md, AGENTS.md, USER.md, ...   # Context files (from template)
    ├── memory/
    │   ├── MEMORY.md                       # Long-term facts
    │   └── HISTORY.md                      # Consolidated history
    ├── sessions/
    │   ├── cli_alice.jsonl                 # Live session
    │   └── cli_alice_20260227_153000.jsonl # Archived session
    └── skills/
```

## Project Structure

```
agent/
├── loop.py          # Persona lane: single-shot LLM call, no tools
├── background.py    # Background lane: agentic loop with tools and memory
├── runner.py        # Reusable run_agentic_loop() (tool-call iteration)
├── context.py       # build_persona_messages() / build_system_prompt()
├── memory.py        # MemoryStore: MEMORY.md + HISTORY.md consolidation
├── tools/           # Tool registry, read_file, write_file, spawn
bus/
├── events.py        # ChatHistorySnapshot, PersonaPromptUpdate, InboundMessage
├── queue.py         # MessageBus
providers/
├── qwen.py          # DashScope / Qwen (OpenAI-compatible)
├── mock.py          # Deterministic mock for testing
session/
└── manager.py       # SessionManager: JSONL persistence, archiving
cli/
└── commands.py      # roleplay, chat, cron, heartbeat commands
context_file_template/  # Templates copied into new user workspaces
```

## Docs

- [`docs/PLAN_CLI_ROLEPLAY_DUAL_LLM.md`](docs/PLAN_CLI_ROLEPLAY_DUAL_LLM.md) — full architecture, data flow, concurrency design
- [`docs/README_TEMPLATE_ORIGINAL.md`](docs/README_TEMPLATE_ORIGINAL.md) — original template README (mock provider, core concepts)
