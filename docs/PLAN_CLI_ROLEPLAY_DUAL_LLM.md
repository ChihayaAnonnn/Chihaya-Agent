# Plan: CLI Roleplay with Dual-LLM Lanes

## Overview

CLI roleplay with two parallel LLM lanes:
- **Persona lane** (`AgentLoop`): smaller, faster model — single-shot response, no tools, no iteration
- **Background lane** (`BackgroundAgent`): larger, more powerful model — agentic loop with tools, memory management, produces guidance for persona

Communication: `asyncio.Queue` for background input; `PromptHolder` shared state for `PersonaPromptUpdate` (each update replaces the previous).

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI Roleplay Agent                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Input ──► Bus.inbound ──► AgentLoop (persona) ──► Response       │
│                        │                    ▲                          │
│                        │                    │ latest_persona_prompt     │
│                        │                    │ (PromptHolder, replace)   │
│                        ▼                    │                          │
│                 background_inbound ──► BackgroundAgent ──► write       │
│                        (Queue)         tools, memory,                   │
│                                        planning                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Lane Responsibilities

| Lane | Model | Infrastructure | Output |
|------|-------|----------------|--------|
| Persona (`AgentLoop`) | Small/fast | Single `provider.chat()` call, lightweight persona context (SOUL.md + history + guidance) | Direct response |
| Background (`BackgroundAgent`) | Large/powerful | `run_agentic_loop()` with tools, memory, full context (bootstrap files + memory + skills) | `PersonaPromptUpdate` → shared state |

---

## 2. Data Exchange Format

### 2.1 Background → Persona: `PersonaPromptUpdate` (`bus/events.py`)

```python
@dataclass
class PersonaPromptUpdate:
    system_prompt_addendum: str   # tone hints, character guidance, situational context
    injected_context: str = ""    # facts/data retrieved by tools → prepended to user message
    conversation_directives: str = ""  # behavioral steering (emphasize, avoid, steer)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 2.2 Background Input: `ChatHistorySnapshot` (`bus/events.py`)

```python
@dataclass
class ChatHistorySnapshot:
    session_key: str
    messages: list[dict[str, Any]]   # LLM format
    current_user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## 3. Shared State and Queues

| Channel | Type | Producer | Consumer |
|---------|------|----------|----------|
| `bus.inbound` | `InboundMessage` | CLI | `AgentLoop` |
| `background_inbound` | `asyncio.Queue[ChatHistorySnapshot]` | `AgentLoop` | `BackgroundAgent` |
| `latest_persona_prompt` | `PersonaPromptUpdate \| None` (`PromptHolder`) | `BackgroundAgent` | `AgentLoop` |

- Persona prompt: **replace-on-write** (not queue). Latest background update wins.
- No throttling: background processes every snapshot; persona never waits.

---

## 4. Context Profiles (`agent/context.py`)

### Persona (lightweight)
- `build_persona_prompt()` — loads only SOUL.md + current time
- `build_persona_messages()` — applies `system_prompt_addendum`, `conversation_directives`, prepends `injected_context` to user message

### Background (full)
- `build_system_prompt()` — bootstrap files (AGENTS.md, SOUL.md, USER.md) + memory + skills
- `build_messages()` — full context used by background's agentic loop

---

## 5. Agentic Runner (`agent/runner.py`)

Standalone `run_agentic_loop()` function. Used by:
- `BackgroundAgent._analyze()` — with tools (read_file, write_file)
- `cli/commands.py:_make_agentic_runner()` — for cron and heartbeat jobs

---

## 6. Memory Management

Memory consolidation is owned by the background lane:
- `BackgroundAgent._maybe_consolidate()` runs after every N turns per session
- Uses `MemoryStore.consolidate()` to flush old messages to HISTORY.md
- Persona lane is always `ephemeral=True` — no disk writes from persona

---

## 7. Concurrency

```python
background_queue = asyncio.Queue[ChatHistorySnapshot]()
prompt_holder = PromptHolder()

agent_loop = AgentLoop(
    ...,
    persona_provider=fast_model,
    background_queue=background_queue,
    prompt_holder=prompt_holder,
    ephemeral=True,
)
background_agent = BackgroundAgent(
    queue=background_queue,
    provider=large_model,
    prompt_holder=prompt_holder,
    workspace=workspace,
)

await asyncio.gather(agent_loop.run(), background_agent.run())
```

### Non-blocking CLI input (critical)

The interactive CLI must use `run_in_executor` for `input()`:

```python
user_input = await asyncio.get_event_loop().run_in_executor(
    None, lambda: input("You: ")
)
```

`input()` is a blocking syscall. If called directly inside `async` code it **freezes the entire event loop thread**, preventing any asyncio callbacks — including HTTP response delivery — from firing. The background agent's LLM HTTP response would arrive at the OS level but remain unprocessed until the next `input()` call returned, creating an artificial N-second delay equal to the user's think time.

With `run_in_executor`, `input()` runs in a thread pool. The event loop remains free and the background response is processed as soon as the API responds, independent of user activity.

### Turn-offset behavior (by design)

The background result for turn N is applied to the persona on turn N+1:

```
Turn N:  persona fires snapshot → background queue
         persona reads PromptHolder (still contains N-1 result) → responds immediately
         background processes snapshot concurrently → writes N result to PromptHolder

Turn N+1: persona reads PromptHolder → gets N result as ## Background Guidance
```

This is intentional: the persona never waits for the background. The tradeoff is that background guidance lags one turn. For long-running sessions this is negligible; guidance accumulates and steers tone/context progressively.

---

## 8. File Layout

```
agent/runner.py        # Reusable run_agentic_loop()
agent/loop.py          # Persona lane: single-shot LLM call
agent/background.py    # Background lane: tools, memory, planning
agent/context.py       # build_persona_prompt/messages + build_system_prompt/messages
bus/events.py          # PersonaPromptUpdate (3 fields), ChatHistorySnapshot
cli/commands.py        # roleplay command; cron/heartbeat use _make_agentic_runner()
```

---

## 9. Dependencies

- No new external deps; reuses `providers/base.LLMProvider`
- Requires two provider instances (or one provider with different `model` params)
