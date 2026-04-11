# GeneralAgentTemplate (original template README)

> Archived from README.md before roleplay system documentation was added.

Minimal CLI nanobot for learning agent core concepts. No real LLM — uses a deterministic Mock provider for reproducible demos.

## Topics Covered

1. **Python async & agent core**
   - Agent loop (LLM ↔ tool execution)
   - Message bus (inbound/outbound queues)
   - Subagent concurrency & management

2. **Tools, skills, context**
   - Tool registry, read_file, write_file, spawn
   - Skills loader, context builder

3. **Memory**
   - MEMORY.md (long-term), HISTORY.md (grep-searchable)
   - Consolidation (simplified, no LLM)

4. **Proactive & long-term tasks**
   - Heartbeat (periodic HEARTBEAT.md check)
   - Cron (scheduled jobs: every N seconds, or at time)

## Install

```bash
cd GeneralAgentTemplate
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

Or run without activating (use venv python directly):

```bash
cd GeneralAgentTemplate
.venv/bin/agent onboard
.venv/bin/agent chat -m "hello"
```

## Usage

| Command | Description |
|--------|-------------|
| `agent onboard` | Initialize workspace |
| `agent chat` | Interactive chat |
| `agent chat -m "hello"` | Single message |
| `agent chat --logs` | Show runtime logs |
| `agent cron add -n demo -m "Check status" -e 30` | Add job (every 30s) |
| `agent cron list` | List jobs |
| `agent cron remove <id>` | Remove job |
| `agent cron run <id>` | Run job manually |
| `agent heartbeat` | Trigger heartbeat manually |

## Mock LLM Behavior

- `hello` / `你好` → greeting
- `spawn` / `子任务` → spawn tool call
- `read` / `读取` → read_file tool call
- After tool result → final summary
- Default → echo

## Project Structure

```
GeneralAgentTemplate/
├── agent/          # Core loop, context, memory, skills, subagent, tools
├── bus/            # Message bus (events, queue)
├── providers/      # Mock LLM
├── session/       # Session manager
├── heartbeat/     # Heartbeat service
├── cron/          # Cron service
├── cli/           # CLI commands
└── workspace/     # Created by onboard
```

## Logging

Use `--logs` to see `[BUS]`, `[LOOP]`, `[SUBAGENT]`, `[HEARTBEAT]`, `[CRON]` logs.
