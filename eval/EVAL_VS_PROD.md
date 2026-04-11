# Evaluation vs Production: Dual-LLM Roleplay System

This document describes the difference between **evaluation mode** (for benchmarking with fixed datasets) and **production mode** (for real user interactions).

---

## System Overview

The dual-LLM roleplay system consists of two concurrent lanes:

| Lane | Model | Role | Latency Constraint |
|------|-------|------|-------------------|
| **Persona** | Fast, long-context LLM (256k) | Responds to user immediately | Must be low-latency |
| **Background** | Powerful reasoning LLM | Analyzes conversation, updates memory/guidance | Can be async, no strict deadline |

**Turn-offset guidance flow:**
- Turn N: Persona responds using guidance from Turn N-1's background analysis
- Turn N: Background analyzes concurrently, writes guidance for Turn N+1

---

## Production Mode

In production, the system engages in open-ended conversation with real users.

### Per-Turn Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  PRODUCTION TURN N                                              │
│                                                                 │
│  INPUT: User message U_n + conversation history (H_1...H_{n-1}) │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PERSONA LANE (synchronous, must be fast)               │   │
│  │                                                         │   │
│  │  Messages: [system] + [history] + [guidance_{n-1}]      │   │
│  │           + [U_n]                                       │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Persona LLM ──► Response P_n ──► RETURN TO USER        │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         │ ChatHistorySnapshot (async)           │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  BACKGROUND LANE (asynchronous)                         │   │
│  │                                                         │   │
│  │  Input: [history] + [U_n] + [P_n]                       │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Background LLM ──► PersonaPromptUpdate                 │   │
│  │         (with tools: read_file, write_file)             │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Write to PromptHolder ──► Available for Turn N+1       │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  STATE AFTER TURN N:                                            │
│    - Conversation history: [..., U_n, P_n]                      │
│    - Memory/guidance: Updated by background                     │
│    - Ready for Turn N+1                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties

1. **Persona's response IS the conversation**: What persona says becomes part of history
2. **Background sees real conversation**: Background analyzes actual user-persona dialogue
3. **Memory builds from real interactions**: Guidance reflects genuine conversation patterns
4. **Open-ended**: No predetermined conversation trajectory

---

## Evaluation Mode

In evaluation, we benchmark against datasets with **predefined conversations** (e.g., LongMemEval). The dataset contains fixed user-assistant turns, some of which contain evidence needed to answer a final question.

### The Problem

```
Dataset conversation:
  [user] "I graduated with Business Administration"
  [assistant] "Congratulations! What are your plans?"    ← Evidence here
  [user] "I'm job hunting now"
  ...

If we let persona generate:
  [user] "I graduated with Business Administration"
  [assistant] "Nice! What field?"                        ← Different response!
  ...

Result: Evidence is lost. Question becomes unanswerable.
```

### Solution: Override Persona Response with Dataset Response

We preserve evidence integrity by **injecting the dataset's assistant responses** into history, while still allowing the background agent to analyze incrementally.

### Per-Turn Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION TURN N (Replay)                                     │
│                                                                 │
│  INPUT: Dataset turn pair (U_n, A_n) from haystack_sessions     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PERSONA LANE                                           │   │
│  │                                                         │   │
│  │  Option A: Skip persona generation entirely             │   │
│  │  Option B: Generate but DISCARD response                │   │
│  │                                                         │   │
│  │  Either way: Inject A_n (dataset's assistant)           │   │
│  │  into conversation history                              │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         │ ChatHistorySnapshot (triggered)       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  BACKGROUND LANE                                        │   │
│  │                                                         │   │
│  │  Input: [history] + [U_n] + [A_n_dataset]               │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Background LLM ──► PersonaPromptUpdate                 │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Write to PromptHolder ──► Available for Turn N+1       │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  STATE AFTER TURN N:                                            │
│    - Conversation history: [..., U_n, A_n_dataset]              │
│    - Memory/guidance: Updated by background                     │
│    - Evidence: Preserved                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

... after all N replay turns ...

┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION TURN N+1 (Final Question)                           │
│                                                                 │
│  INPUT: Target question Q from dataset                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PERSONA LANE (full invocation)                         │   │
│  │                                                         │   │
│  │  Messages: [system] + [full history] + [guidance_N]     │   │
│  │           + [Q]                                         │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Persona LLM ──► Response R ──► EVALUATED               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Response R is compared against dataset's expected answer       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Side-by-Side Comparison

| Aspect | Production | Evaluation |
|--------|------------|------------|
| **User message source** | Real user input | Dataset's predefined turns |
| **Assistant response source** | Persona generates | Dataset's predefined turns (injected) |
| **Persona invocation** | Every turn | Only for final question |
| **Background sees** | Real persona responses | Dataset's assistant responses |
| **Memory builds from** | Genuine conversation | Dataset conversation |
| **What's tested** | Full interactive experience | Memory extraction & retention |
| **Conversation trajectory** | Open-ended | Fixed, predetermined |

---

## What Evaluation Tests

### ✅ Tested

1. **Incremental memory extraction**: Can background identify and retain key information from each turn?
2. **Memory accumulation**: Does guidance improve over multiple turns?
3. **Turn-offset recall**: Can persona use N-1 guidance to answer correctly?
4. **Long-context utilization**: Can persona leverage full history when answering?

### ❌ Not Tested

1. **Persona response quality during replay**: We skip/discard persona's replay responses
2. **Coherence of real conversation**: Dataset conversations may differ from real patterns
3. **User follow-up patterns**: Real users react to persona's actual responses
4. **Background's handling of persona-generated content**: Background sees dataset responses, not persona's

---

## Validity of the Approximation

The evaluation mode makes a **controlled approximation**:

> We assume that if the memory system can extract and retain information from the dataset's conversations, it can also do so from real conversations.

This is valid for benchmarking **memory capability** because:

1. The background agent's extraction logic is the same
2. The memory update mechanism is the same
3. The guidance injection mechanism is the same
4. The persona's final-answer generation is the same

The only difference is **what conversation** the background analyzes — and for memory benchmarking, that's acceptable.

---

## Implications for Evaluation Design

### Dataset Requirements

- Dataset must contain complete user-assistant turn pairs
- Evidence must be embedded in these turns
- Final question must be answerable from accumulated evidence

### Metrics

- **Answer correctness**: Does persona's final response contain the expected answer?
- **Per-question-type breakdown**: Different memory abilities (temporal, multi-session, etc.)
- **Optional**: Background recall accuracy (did background identify evidence turns?)

### Future Extensions

When testing with **explicit memory mechanisms** (e.g., MEMORY.md consolidation):

1. Background writes to persistent memory during replay
2. Persona reads consolidated memory as part of context
3. Measure: Does explicit memory improve recall vs. implicit guidance?

---

## Summary

| Mode | Persona | Background | History Source |
|------|---------|------------|----------------|
| **Production** | Generates every turn | Analyzes real responses | Real conversation |
| **Evaluation** | Only generates final answer | Analyzes dataset responses | Dataset turns |

Evaluation mode sacrifices **interaction fidelity** to preserve **evidence integrity**. This trade-off is appropriate for benchmarking memory systems against fixed datasets like LongMemEval.
