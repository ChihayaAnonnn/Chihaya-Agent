# cli-async-agent — Dual-LLM Roleplay Agent

Low-latency persona + heavy background reasoning with structured memory,
proactive engagement, and an eval harness for regression tracking.

This is not a wrapper over a chat API: it is an architectural experiment in
how to keep a single-process agent feeling fast while still giving it the
time to think, remember, and act.

---

## Why it exists

A single LLM call sitting in the hot path makes every feature a latency
trade-off. The more you ask the model to do (retrieve memory, plan, call
tools), the longer the user waits. Teams either keep the agent shallow and
amnesiac, or build a slow agent that users stop trusting.

This project splits responsibilities across two lanes running on the same
event loop:

| Lane | Model | Role | Latency budget |
|---|---|---|---|
| **Persona** | fast (e.g. `qwen-plus`) | one-shot reply, no tools | ≈1–2s |
| **Background** | heavier (e.g. `qwen-max`) | tools, memory writes, planning | async, ≤30s |

The persona never waits for the background. It fires a snapshot into a
queue, reads any steering hint left by the prior background turn, and
replies. The background picks the snapshot up, spends as long as it needs,
and writes a one-shot hint + durable memory for the *next* turn.

---

## Architecture at a glance

```
User input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ AgentLoop  (persona lane — every turn)                      │
│  1. record_user_activity → IdleMonitor                      │
│  2. ephemeral_hint = prompt_holder.read_and_consume()       │
│  3. snapshot → background queue (non-blocking)              │
│  4. provider.chat(SOUL + history + hint)  → reply           │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
                │ ChatHistorySnapshot           │ OutboundMessage
                ▼                              │
┌─────────────────────────────────────────────┐│
│ BackgroundAgent  (reasoning lane)           ││
│  run_agentic_loop (read/write tools, ≤30s)  ││
│  → compress_session → HISTORY.md            ││
│  → archive_old_memory → MEMORY.md ≤1000 tok ││
│  → PersonaPromptUpdate → PromptHolder       ││
└─────────────────────────────────────────────┘│
                                               │
┌─────────────────────────────────────────────┐│
│ IdleMonitor  (proactive lane)               ││
│  rule-filter: MEMORY.md `## 待跟进` / hooks  ││
│  LLM-gate: speak only if worthwhile         ││
│  cooldown + quiet hours + daily cap         │▼
│  → ProactiveMessage → outbound queue ───────→ CLI render
└─────────────────────────────────────────────┘
```

### Turn-offset guidance flow

```
Turn N   → persona replies (uses N-1 background hint)
           background starts analyzing N concurrently
Turn N+1 → persona consumes N's hint exactly once
```

The hint is consumed, not merely read, so a stale hint can never leak
across two turns.

---

## Memory model

Two-layer, both plain Markdown. No vector DB.

| Layer | File | Injected into prompt? | Budget |
|---|---|---|---|
| Long-term user profile | `memory/MEMORY.md` | yes, every turn | ≤1000 tokens |
| Compressed logs | `memory/HISTORY.md` | no, audit only | unbounded |

How it stays small:

- **`compress_session`** — every N user turns, the background LLM summarizes
  the oldest chunk of the session into `HISTORY.md` (LLM-driven, not
  mechanical truncation) and trims the session file.
- **`archive_old_memory`** — when `MEMORY.md` exceeds 1000 tokens, an LLM
  prunes stale items, writes back the trimmed version, and appends what it
  removed to `HISTORY.md`.
- **`WriteFileTool` hard guard** — rejects writes to `MEMORY.md` over 1500
  tokens and tells the model to retry with a leaner version.
- **Structured prompt** — the background system prompt defines the exact
  section layout (`基本信息` / `当前进行中的项目` / `重要偏好` /
  `最近关键决策` / `待跟进`) and tells the model to read-then-patch rather
  than rewrite.

Why not RAG? Conversational memory ≠ knowledge retrieval. Vector similarity
is unreliable for "what did the user tell me about themselves?"; missing
recall is silent, and the file can't be hand-edited. Keeping memory as a
Markdown file keeps it debuggable, auditable, and cheap.

---

## Proactive engagement

The agent may speak first after idle, but only when a rule filter decides
it is worth spending a token on:

1. **`## 待跟进` non-empty** in `MEMORY.md`, or
2. **hook keywords** (`明天`, `下次`, `等我`, `later`, `tomorrow`, …) in
   the last few messages.

Only then does it call the LLM to decide *whether* to speak, and if so
*what* to say. Silence is an explicit valid output — the persona prompt
tells the model that staying quiet is a feature, not a failure. A cooldown
(default 10 min), quiet hours, and a daily cap bound the total cost
regardless of how many sessions are active.

---

## Reliability

- **Retry + exponential backoff** on transient LLM errors (rate limit,
  connection, 5xx) with jitter. Auth/4xx propagate immediately.
- **Background timeout** — `_analyze` runs under `asyncio.wait_for`
  (30s default). A timeout logs a structured event and drops the turn;
  persona is never blocked.
- **`PromptHolder` atomicity** — `read_and_consume()` is `async` under the
  same lock as `write()`, closing the stale-hint race.
- **MEMORY.md write protection** — oversized writes are rejected with a
  user-visible error the model can act on.

---

## Observability

Every turn emits a single structured JSON line tagged with a `trace_id`
shared by the persona and background events:

```json
{"event":"persona_turn","trace_id":"a3f1…","latency_ms":1240,"usage":{"prompt_tokens":820,"completion_tokens":145}}
{"event":"background_turn","trace_id":"a3f1…","latency_ms":3800,"ephemeral_hint":"..."}
{"event":"proactive_fired","trace_id":"b8e2…","session":"cli:alice","trigger":"idle"}
```

The logger is `agent.obs`; hook it into your preferred sink.

---

## Eval harness

```bash
agent eval run --dataset memory_recall     # self-authored, 15 entries
agent eval run --dataset longmemeval       # oracle subset
agent eval results --compare 2             # diff the latest two runs
```

Judging is two-stage: `keyword_judge` decides correctness for free when all
expected keywords appear verbatim; otherwise an LLM judge is asked for a
rubric verdict. Results land in `eval/results/*.jsonl` with timestamps so
`--compare N` can show the delta across runs.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

Set your API key:

```bash
export DASHSCOPE_API_KEY=sk-...  # or QWEN_API_KEY
```

## Usage

| Command | Description |
|---|---|
| `agent onboard` | Initialize shared workspace |
| `agent roleplay -u alice` | Interactive dual-LLM session |
| `agent roleplay -u alice -m "hello"` | Single-message mode |
| `agent roleplay --no-proactive` | Disable IdleMonitor |
| `agent roleplay --idle-seconds 60` | Speak up after 60s of silence |
| `agent roleplay --logs` | Stream runtime + structured logs |
| `agent cron add -n demo -m "Check status" -e 30` | Add cron job |
| `agent heartbeat` | Trigger heartbeat manually |
| `agent eval run -d memory_recall` | Run the recall eval |
| `agent eval results --compare 2` | Compare the last two eval runs |

### Per-user workspaces

Each user has an isolated workspace at `workspaces/{user}/`. On first use
the files under `context_file_template/` (SOUL.md, AGENTS.md, USER.md) are
copied in. Memory and sessions are always scoped to the user folder.

```
workspaces/
└── alice/
    ├── SOUL.md  AGENTS.md  USER.md
    ├── memory/
    │   ├── MEMORY.md      # long-term (≤1000 tokens)
    │   └── HISTORY.md     # compressed logs
    ├── sessions/
    │   ├── cli_alice.jsonl
    │   └── cli_alice_20260227_153000.jsonl
    └── skills/
```

---

## Project layout

```
agent/
├── loop.py          persona lane, trace-id + structured logs
├── background.py    heavy lane, wait_for(30s), compress+archive hooks
├── memory.py        MemoryStore: compress_session / archive_old_memory
├── proactive.py     IdleMonitor: rule filter → LLM → ProactiveMessage
├── runner.py        reusable tool-call loop
├── context.py       prompt assembly (SOUL/USER/MEMORY/hint)
├── tools/           read_file / write_file (MEMORY.md guarded)
bus/
├── events.py        InboundMessage / OutboundMessage / ChatHistorySnapshot
├── events.py        PersonaPromptUpdate / ProactiveMessage
├── queue.py         MessageBus
providers/
├── qwen.py          DashScope / Qwen, retry + backoff
├── base.py          LLMProvider interface, trace file
session/
└── manager.py       JSONL session persistence, ephemeral mode
eval/
├── runner.py        replay + judge + aggregate
├── judge.py         keyword fast path → llm_judge fallback
├── dataset.py       LongMemEval + custom memory_recall loader
└── datasets/
    └── memory_recall.jsonl
cli/
└── commands.py      roleplay / cron / heartbeat / eval
```

---

## Docs

- [`docs/UPGRADE_PLAN.md`](docs/UPGRADE_PLAN.md) — the roadmap that produced
  the current architecture (memory, proactive, eval, observability).
- [`docs/PLAN_CLI_ROLEPLAY_DUAL_LLM.md`](docs/PLAN_CLI_ROLEPLAY_DUAL_LLM.md)
  — original dual-LLM design notes.
