# Agent 升级开发文档

> 基于现有 `cli-async-agent` 双 LLM 架构，三个方向的工业化扩展：
> 1. **Proactive Engagement** — 主动交互
> 2. **Memory 增强** — 压缩上下文 + memory.md 双层记忆
> 3. **Eval Harness** — 可量化评估体系
>
> 开发者：1 人；目标：可放简历的独立工业化项目。

---

## 目录

1. [现有架构速览](#现有架构速览)
2. [Memory 方案决策](#memory-方案决策)
3. [Feature 1：Proactive Engagement（主动交互）](#feature-1-proactive-engagement)
4. [Feature 2：Memory 增强（双层记忆）](#feature-2-memory-增强)
5. [Feature 3：Eval Harness](#feature-3-eval-harness)
6. [横切关注点：可观测性 & 稳定性](#横切关注点)
7. [依赖变更](#依赖变更)
8. [执行步骤清单](#执行步骤清单)

---

## 现有架构速览

```
CLI input
    │
    ▼
MessageBus.inbound
    │
    ▼
AgentLoop (persona)               ← 单次 LLM call，无工具
    │ put_nowait(snapshot)         ← fire-and-forget
    ▼
BackgroundAgent (reasoning)       ← run_agentic_loop，最多10轮 tool call
    │ write(PersonaPromptUpdate)   ← ephemeral_hint
    ▼
PromptHolder                      ← 前台下一轮 read()
```

**现有关键限制（待解决）**
- `PromptHolder` 是 sticky，不是 truly ephemeral
- `MemoryStore.consolidate()` 是机械截断归档，无 LLM 摘要压缩
- `MEMORY.md` 无大小控制，长期使用后会膨胀至 token 上限
- 后台只能被动分析，无主动触发路径
- `ephemeral_hint` 无消费确认，可能跨多轮泄漏

---

## Memory 方案决策

### 为什么不用 RAG 做记忆

对话记忆 ≠ 知识检索，两者需求截然不同：

| 维度 | RAG | memory.md + 压缩摘要 |
|------|-----|----------------------|
| 设计目标 | 查询外部静态知识库 | 保留对话中积累的动态上下文 |
| 召回机制 | 语义相似度（不稳定） | 确定性全文注入（无遗漏） |
| 典型失败场景 | 用户说"我有5年Python经验"，RAG 无法在问 Rust 问题时召回 | 无此问题，关键事实始终在 prompt 中 |
| 基础设施依赖 | 向量数据库 + embedding 模型版本管理 | 普通文本文件 |
| 可调试性 | 召回结果难以预测 | 全透明，用户可直接编辑 |

**结论**：当前 `memory.md` 方向正确，问题在于缺少**主动的 LLM 摘要压缩**机制来控制文件大小和信息密度。需要增强的是压缩质量，而不是引入向量检索。

### 双层记忆架构

```
短期记忆（会话内）          长期记忆（跨会话）
─────────────────          ──────────────────
sessions/*.jsonl            workspaces/{user}/
  最近 N 条原始消息   ──压缩──→  memory/MEMORY.md
                                  用户长期档案
                                  ≤ 1000 token
                             memory/HISTORY.md
                                  归档摘要（不注入 prompt）
```

---

## Feature 1: Proactive Engagement

### 目标

用户停止输入超过阈值后，后台自主判断是否主动发起话题，从而实现拟人化的主动交互体验。

### 新增文件

```
agent/proactive.py          # IdleMonitor：监测 idle + 触发主动消息
```

`bus/events.py` 新增 `ProactiveMessage` dataclass。

### 数据结构

```python
# bus/events.py 新增
@dataclass
class ProactiveMessage:
    """由 IdleMonitor 触发的主动消息，直接写入 bus outbound。"""
    session_key: str
    channel: str
    chat_id: str
    trigger_reason: str        # "idle" | "memory_recall" | "scheduled"
    # 注意：cooldown 是 IdleMonitor 的内部状态，不应放在消息体中
    # 消费方（CLI 渲染层）不需要知道冷却时间
```

冷却时间由 `IdleMonitor` 内部维护：

```python
# agent/proactive.py IdleMonitor 内部状态
self._cooldown_until: dict[str, int] = {}  # session_key → cooldown_until_ms
```

### IdleMonitor 模块设计

```python
# agent/proactive.py
class IdleMonitor:
    """
    监控用户 idle 状态，触发后台主动生成消息。
    
    工作方式：
    1. 每次前台处理完用户消息后调用 record_user_activity()
    2. 常驻协程 run() 检测 idle 时长是否超过阈值
    3. 超过则调用 LLM 生成主动消息，写入 bus outbound
    """
    
    DEFAULT_IDLE_THRESHOLD_S: int = 300     # 5分钟
    DEFAULT_COOLDOWN_S: int = 600           # 10分钟冷却
    QUIET_HOURS: tuple = (23, 8)            # 23:00-08:00 静默
```

**关键接口：**

```python
def record_user_activity(self) -> None
    # 每次用户消息到来时调用，重置 idle 计时器

async def run(self) -> None
    # 常驻协程，在 cli/commands.py 的 run_interactive() 中以 asyncio.Task 启动

async def _generate_proactive_message(self, session_key: str) -> str | None
    # 调用 BackgroundAgent 同款 provider，生成主动消息
    # system prompt 读取 SOUL.md + MEMORY.md，赋予主动发起能力
```

**抑制策略（防止骚扰）：**

```python
def _should_trigger(self, session_key: str) -> bool:
    now = datetime.now()
    # 1. 静默时段检测
    if QUIET_HOURS[0] <= now.hour or now.hour < QUIET_HOURS[1]:
        return False
    # 2. 冷却期检测（per-session 内部状态，不暴露到消息体）
    if _now_ms() < self._cooldown_until.get(session_key, 0):
        return False
    # 3. 用户在线检测（最近 idle < threshold）
    if _now_ms() - self._last_activity_ms < self._threshold_ms:
        return False
    return True

def _set_cooldown(self, session_key: str) -> None:
    self._cooldown_until[session_key] = _now_ms() + self._cooldown_s * 1000
```

**主动触发的 LLM 调用成本控制：**

直接用 LLM 判断"是否要主动发话"会带来不必要的 API 消耗。实现时必须先经过规则过滤层，只有通过规则的 session 才升级到 LLM 判断：

```python
async def _decide_and_send(self, session_key: str) -> None:
    """两阶段决策：规则过滤 → LLM 判断。"""
    # 阶段 1：纯文本规则过滤（零 API 成本）
    # 只有满足以下任意条件才进入阶段 2：
    #   - MEMORY.md 的"待跟进"字段非空
    #   - 最近 session 消息中含有未竟话题信号词（"明天"、"下次"、"等我"等）
    if not self._has_contextual_hook(session_key):
        return  # 规则不通过，直接跳过，不花 token

    # 阶段 2：LLM 判断是否值得主动发起
    content = await self._generate_proactive_message(session_key)
    if content:
        await self._bus.publish_outbound(OutboundMessage(..., content=content))
        self._set_cooldown(session_key)

def _has_contextual_hook(self, session_key: str) -> bool:
    """规则判断：session 中是否有可供主动跟进的钩子。"""
    # 读取 MEMORY.md 待跟进字段
    mem = self._memory.read_long_term()
    if "## 待跟进" in mem and mem.split("## 待跟进")[-1].strip():
        return True
    # 检查最近 3 条消息中是否有延续性信号
    HOOKS = ["明天", "下次", "等我", "稍后", "回头", "待定", "later", "tomorrow"]
    recent = self._get_recent_messages(session_key, n=3)
    return any(h in m.get("content", "") for m in recent for h in HOOKS)
```

这样在绝大多数 idle 轮次中（无待跟进事项）规则层直接 `return`，LLM 调用只在真正有价值时才触发。

### 与现有架构的集成点

```
# cli/commands.py run_interactive() 中增加第三个 task：
idle_task = asyncio.create_task(idle_monitor.run())

# AgentLoop._process_message() 末尾增加：
if self.idle_monitor:
    self.idle_monitor.record_user_activity()

# IdleMonitor 产出的消息直接 publish_outbound，
# consume_outbound() 不需要修改
```

### 主动消息的 Prompt 设计

```
System:
  {SOUL.md 内容}
  {USER.md 内容}
  {MEMORY.md 内容}
  
  你可以主动发起话题。选择合适的方式：
  - 基于记忆里的待跟进事项问候
  - 分享与用户兴趣相关的想法
  - 轻松的问候
  
  回复要自然简短，不要生硬。如果没有合适的话说，返回空字符串。

User:
  用户已经 {N} 分钟没有说话了。
  最近的对话上下文：{最近3条消息摘要}
  你是否想主动发起话题？如果是，直接输出消息内容；如果否，输出空字符串。
```

---

## Feature 2: Memory 增强

### 目标

1. 将 `MemoryStore.consolidate()` 从"机械截断归档"升级为"LLM 驱动的摘要压缩"
2. 控制 `MEMORY.md` 大小（≤ 1000 token），自动归档旧内容到 `HISTORY.md`
3. 增强后台 Agent 对 `MEMORY.md` 的结构化写入质量

### 2.1 LLM 摘要压缩（替换现有 consolidate）

**现有实现问题：**
```python
# agent/memory.py 现有 consolidate()
# 只是机械截断：每条消息取前 200 字符追加到 HISTORY.md
# 没有 LLM 参与，信息损失严重
entry = f"[{timestamp}] {role}: {content[:200]}"
```

**新实现方案：**

```python
# agent/memory.py 新增方法
async def compress_session(
    self,
    session: Session,
    provider: LLMProvider,
    model: str,
    keep_count: int = 10,
) -> None:
    """
    LLM 驱动的会话压缩：
    1. 取 session 中待压缩的消息段
    2. 调用 LLM 生成结构化摘要
    3. 摘要追加到 HISTORY.md
    4. 原始消息从 session 中裁剪
    """
```

**压缩 Prompt 设计：**

```
请将以下对话压缩为结构化摘要，必须保留：
1. 用户透露的个人背景和偏好变化
2. 已经做出的决策和选择（不要省略）
3. 未完成的任务和待跟进事项
4. 重要的上下文约束（技术栈、边界条件等）

不需要保留：
- 具体代码细节（除非是核心架构决策）
- 过程中的错误尝试
- 礼貌性寒暄

输出格式为 Markdown，使用 ## 分节，控制在 300 token 以内。
```

### 2.2 MEMORY.md 大小控制

**触发策略：**

```python
MEMORY_MAX_TOKENS = 1000          # 超过则触发归档
COMPRESSION_THRESHOLD_TURNS = 20  # 每 20 轮检查一次（现有逻辑不变）
```

**归档流程：**

```
MEMORY.md 超过 1000 token
    │
    ▼
LLM：将 MEMORY.md 中超过 6 个月未更新的字段归档
    │
    ├──→ 归档内容追加到 HISTORY.md（不注入 prompt，保留可追溯）
    └──→ 更新后的精简版写回 MEMORY.md（≤ 1000 token）
```

**背景 Agent 调用时机：**
在 `BackgroundAgent._maybe_consolidate()` 中增加 `MEMORY.md` 大小检查：

```python
async def _maybe_consolidate(self, snapshot):
    # 现有：每 20 轮压缩 session 历史
    # 新增：检查 MEMORY.md token 数，超限则触发归档
    mem_content = self.memory.read_long_term()
    if _estimate_tokens(mem_content) > MEMORY_MAX_TOKENS:
        await self._archive_old_memory(mem_content)
```

### 2.3 后台写 MEMORY.md 的结构化约束

在 `BACKGROUND_SYSTEM_PROMPT` 中增加对 MEMORY.md 写入格式的明确约束：

```
当你写入 memory/MEMORY.md 时，必须遵循以下结构：

## 基本信息
（用户职业、技术背景、时区等稳定信息）

## 当前进行中的项目
（正在做的事，预计完成时间）

## 重要偏好
（沟通风格、技术偏好等）

## 最近关键决策
（带日期，最多保留最近 5 条）

## 待跟进
（未完成的事项，完成后删除）

每个字段更新时先 read_file 读取现有内容，只修改变化的部分，不要整个重写。
控制整个文件在 1000 token 以内。
```

### 2.4 修复 PromptHolder ephemeral 泄漏

**现有问题：**
```python
# agent/background.py
def read(self) -> PersonaPromptUpdate | None:
    return self._value   # 不清空，sticky until overwrite
```

**修复方案：消费后清空，且必须加锁**

`write()` 使用了 `async with self._lock`，`read_and_consume()` 也必须加锁，否则在极端情况下（background 同时写入）仍有竞态。同时由于加锁后变为 `async`，调用方必须同步修改为 `await`：

```python
# agent/background.py PromptHolder
async def read_and_consume(self) -> PersonaPromptUpdate | None:
    """读取并清空，实现真正的 once-per-turn 语义。加锁防止与 write() 竞态。"""
    async with self._lock:
        value = self._value
        self._value = None
        return value
```

调用方 `agent/loop.py` 必须对应修改为：

```python
# agent/loop.py _process_message() 中
if self.prompt_holder is not None:
    if update := await self.prompt_holder.read_and_consume():  # ← 必须 await
        ephemeral_hint = update.ephemeral_hint or ""
```

**注意**：如果仅在 `PromptHolder` 增加 `read_and_consume()` 但 `loop.py` 仍调用原来的同步 `read()`，竞态问题不会消失。两处必须同步修改。

---

## Feature 3: Eval Harness

### 目标

建立可量化的评估体系，能够回答"这次改动让 Agent 变好了还是变差了？"

### 评估维度

| 维度 | 指标 | 测量方式 |
|------|------|---------|
| 记忆召回 | 关键事实保留率 | 对话 N 轮后问用户之前提到的事实，LLM judge 评分 |
| 响应延迟 | P50 / P95 延迟 | 前台 `provider.chat()` 耗时打点 |
| 主动交互 | 触发精准率 | 人工标注：主动消息是否合适 |
| 人设一致性 | SOUL.md 遵从率 | LLM judge：回复是否符合人设描述 |
| Token 成本 | 每轮平均 token | provider 返回的 usage 数据 |

### 自建数据集结构

```
eval/
├── datasets/
│   ├── memory_recall.jsonl      # 记忆召回测试集（50条）
│   ├── persona_consistency.jsonl # 人设一致性测试集（30条）
│   └── proactive_trigger.jsonl  # 主动触发时机测试集（20条）
├── judge.py                     # LLM judge 评分逻辑
├── runner.py                    # 现有，扩展支持新数据集
└── results/                     # 每次运行结果落盘
```

**memory_recall.jsonl 样例格式：**

```jsonl
{
  "id": "mr_001",
  "type": "memory_recall",
  "setup_turns": [
    {"role": "user", "content": "我是做 Java 后端的，有 3 年经验"},
    {"role": "assistant", "content": "..."}
  ],
  "gap_turns": 15,
  "question": "我的技术背景是什么？",
  "expected_keywords": ["Java", "后端", "3年"],
  "judge_prompt": "回答中是否包含用户的技术背景信息？评分 0-1。"
}
```

### LLM Judge 实现

```python
# eval/judge.py
async def judge_response(
    response: str,
    expected_keywords: list[str],
    judge_prompt: str,
    provider: LLMProvider,
    model: str = "qwen-turbo",
) -> float:
    """
    返回 0-1 的评分。
    优先用关键词匹配（确定性）；
    复杂语义判断才调 LLM，减少 judge 成本。
    """
```

### 回归基线流程

```bash
# 每次重要改动前后各跑一次
agent eval run --dataset memory_recall
agent eval run --dataset persona_consistency

# 对比结果
agent eval results --compare latest two
```

---

## 横切关注点

### 可观测性

**结构化日志增强：**
每轮交互产出一条 JSON 日志，包含：

```python
{
  "trace_id": "uuid",           # 每轮对话唯一 ID
  "turn": 5,
  "persona_latency_ms": 1240,
  "background_latency_ms": 3800,
  "persona_tokens": {"input": 820, "output": 145},
  "background_tokens": {"input": 1100, "output": 210},
  "tools_used": ["read_file", "write_file"],
  "memory_updated": true,
  "ephemeral_hint": "用户看起来有点急，保持简洁"
}
```

**Provider 层增强（`providers/base.py`）：**
在 `LLMProvider.chat()` 返回的 `LLMResponse` 中补充 `latency_ms` 和 `usage` 字段（部分已有）。

### 稳定性

**Retry + 指数退避（`providers/qwen.py`）：**

```python
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 1.0

async def chat(self, ...):
    for attempt in range(MAX_RETRIES):
        try:
            return await self._call(...)
        except RateLimitError:
            await asyncio.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
    raise
```

**后台超时保护（`agent/background.py`）：**

```python
# _analyze() 中增加整体超时
final_content, tools_used = await asyncio.wait_for(
    run_agentic_loop(..., max_iterations=10),
    timeout=30.0,          # 后台最多 30 秒
)
```

**MEMORY.md 写操作防腐：**
`WriteFileTool` 对 `memory/MEMORY.md` 写入时额外检查 token 数，超过 1500 token 时拒绝写入并返回错误提示，让模型重新生成更精简的版本。

---

## 依赖变更

**新增（最小化）：**

```toml
# pyproject.toml
dependencies = [
    # 现有不变
    "typer>=0.9.0",
    "rich>=13.0.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    # 新增
    "tiktoken>=0.7.0",     # token 计数，用于 MEMORY.md 大小控制
]
```

**刻意不引入：**
- ❌ 向量数据库（chromadb / sqlite-vec）：memory.md 方案不需要
- ❌ LangChain / LlamaIndex：不引入框架依赖，保持架构自主性
- ❌ 新的 embedding 模型：同上

---

## 执行步骤清单

### Phase 1：修复现有设计缺陷（Week 1）

- [ ] **Step 1.1**：修复 `PromptHolder`：新增 `async def read_and_consume()`（加 `async with self._lock`），同步修改 `agent/loop.py` 中调用处为 `await prompt_holder.read_and_consume()`；两处必须同时改，否则竞态不消失
- [ ] **Step 1.2**：`providers/qwen.py` 增加 retry + 指数退避（3次，1/2/4s）
- [ ] **Step 1.3**：`agent/background.py` `_analyze()` 增加 `asyncio.wait_for(timeout=30)`
- [ ] **Step 1.4**：结构化日志：`agent/loop.py` 和 `agent/background.py` 输出带 `trace_id` 的 JSON 日志行

验收：`agent roleplay -u test --logs` 运行 5 轮，日志中出现结构化 JSON，无 sticky hint 跨轮泄漏。

---

### Phase 2：Memory 增强（Week 2-3）

- [ ] **Step 2.1**：`agent/memory.py` 新增 `compress_session()` 方法，接收 `provider` 和 `model` 参数
- [ ] **Step 2.2**：`agent/memory.py` 新增 `_estimate_tokens()` 工具函数（用 tiktoken）
- [ ] **Step 2.3**：`agent/memory.py` 新增 `archive_old_memory()` 方法，将超限内容归档到 `HISTORY.md`
- [ ] **Step 2.4**：`agent/background.py` `_maybe_consolidate()` 接入新的 `compress_session()`，替换现有机械截断
- [ ] **Step 2.5**：`agent/background.py` `_maybe_consolidate()` 中增加 `MEMORY.md` token 超限检查，触发 `archive_old_memory()`
- [ ] **Step 2.6**：`BACKGROUND_SYSTEM_PROMPT` 中增加 `MEMORY.md` 结构化写入规范
- [ ] **Step 2.7**：`agent/tools/filesystem.py` `WriteFileTool` 对 `memory/MEMORY.md` 增加 token 超限写保护

验收：对话 25 轮后，`MEMORY.md` token 数 ≤ 1000，`HISTORY.md` 包含 LLM 压缩的摘要文本（非截断原文）。

---

### Phase 3：Proactive Engagement（Week 4）

- [ ] **Step 3.1**：`bus/events.py` 新增 `ProactiveMessage` dataclass
- [ ] **Step 3.2**：创建 `agent/proactive.py`，实现 `IdleMonitor` 类
  - `record_user_activity()`
  - `_should_trigger()` 含静默时段 + 冷却逻辑
  - `_generate_proactive_message()` 调用 LLM 生成主动消息
  - `run()` 常驻协程
- [ ] **Step 3.3**：`agent/loop.py` `_process_message()` 末尾调用 `idle_monitor.record_user_activity()`
- [ ] **Step 3.4**：`cli/commands.py` `roleplay` 命令中实例化 `IdleMonitor` 并启动第三个 task
- [ ] **Step 3.5**：设计主动消息 Prompt（见 Feature 1 章节），写入 `context_file_template/SOUL.md`

验收：设置 `idle_threshold=30s`，等待 30s 后 Agent 主动发起话题；连续触发时冷却生效（第二次不早于 60s 后）。

---

### Phase 4：Eval Harness（Week 5）

- [ ] **Step 4.1**：创建 `eval/datasets/` 目录，编写 `memory_recall.jsonl`（50条）
- [ ] **Step 4.2**：创建 `eval/judge.py`，实现关键词匹配 + LLM judge 双模式评分
- [ ] **Step 4.3**：`eval/runner.py` 扩展支持 `memory_recall` 数据集和新的 judge 接口
- [ ] **Step 4.4**：在 `cli/commands.py` `eval run` 中增加 `--compare` 参数，支持两次结果对比
- [ ] **Step 4.5**：运行基线评估，记录 Phase 2 前后的记忆召回准确率对比数据

验收：`agent eval run --dataset memory_recall` 输出正确率数据；基线数据落盘至 `eval/results/`。

---

### Phase 5：收尾与交付物（Week 6）

- [ ] **Step 5.1**：`README.md` 重写：架构图（mermaid）+ 核心设计决策 + benchmark 数据
- [ ] **Step 5.2**：`Dockerfile` + `.dockerignore`
- [ ] **Step 5.3**：`tests/` 覆盖率补全：turn-offset 行为单测、idle 触发单测、memory 压缩单测
- [ ] **Step 5.4**：`pyproject.toml` 补全项目元信息（作者、homepage、classifiers）
- [ ] **Step 5.5**：录制 demo gif：展示主动交互 + 记忆跨会话保留

---

## 架构演进对比

**改动前：**
```
CLI → MessageBus → AgentLoop → BackgroundAgent → PromptHolder
                                    ↓
                               MEMORY.md（机械归档，无大小控制）
```

**改动后：**
```
CLI → MessageBus → AgentLoop ──────────────────────────────┐
                       ↓                                   │
                  IdleMonitor ── (idle触发) ──→ bus.outbound │
                       ↓                                   │
                  BackgroundAgent                          │
                       ↓                                   │
                  PromptHolder (真正 once-per-turn)         │
                       ↓                                   │
              MEMORY.md（LLM 压缩，≤1000 token）            │
              HISTORY.md（归档摘要，不注入 prompt）          │
                                                           ↓
                                                     消费 outbound → 终端渲染
```
