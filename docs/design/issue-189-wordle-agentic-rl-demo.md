# Issue #189: Build a Wordle agentic RL demo

## 系分文档

- **Issue**: [inclusionAI/AReno#189](https://github.com/inclusionAI/AReno/issues/189)
- **标题**: Build a Wordle agentic RL demo
- **认领人**: 夏烬 (xiajin.lcy)
- **日期**: 2026-07-28

---

## 1. Issue 概述

### 1.1 背景与动机

AReno 用户需要一个 **Wordle 猜词游戏的 Agentic RL demo**，作为一个聚焦的、可独立审查的能力。当前 agentic 示例中没有 Wordle 这种经典的"部分可观测 + 多轮反馈 + 有限尝试次数"游戏的示例。

### 1.2 目标

- 内嵌一个明确定义许可的小型词表
- 实现一个猜词工具，返回 exact（位置正确）、present（存在但位置错误）、absent（不存在）反馈
- 按照标准 Wordle 规则处理重复字母的计数
- 运行时无网络访问
- 提供确定性评估，报告 solve rate 和 guesses-to-solve（按词长度）

### 1.3 验收标准

- [ ] 覆盖无效单词、重复字母、不同支持长度、猜测耗尽；提供确定性评估，报告 solve rate 和 guesses-to-solve（按词长度）
- [ ] 实现使用现有 AReno 契约，不引入外部数据库或强制沙箱
- [ ] 默认行为保持向后兼容
- [ ] 聚焦的自动化测试覆盖成功路径、无效输入和一条边界/失败路径
- [ ] 用户文档包含最小可运行示例并解释可观测输出

---

## 2. 现状分析

### 2.1 现有 Agentic 示例对比

| 示例 | 类型 | 工具调用 | 轮数 | 与 Wordle 相似度 | 参考价值 |
|------|------|---------|------|-----------------|---------|
| tictactoe | 单步决策 | `choose_square` | 1 | 低 | 工具 schema 模式 |
| duelgrid | 单步决策 | `choose_action` | 1 | 低 | 复杂动作空间 |
| shopping | 固定 4 步流 | 4 个工具 | 4 | 中 | 多轮工具序列 |
| codebreaker | 多轮猜测 | `guess_code` | <=6 | **高** | **直接模板** |
| coding | 自由多轮 | 10 个工具 | <=8 | 中 | 多轮循环模式 |

**结论：以 `examples/agentic/codebreaker/` 为直接模板**，因为 codebreaker 也是一个"多轮猜测 + 工具结果反馈累积 + 有限尝试次数"的游戏，结构上与 Wordle 几乎完全对应。

### 2.2 Codebreaker 示例结构分析

codebreaker 目录包含以下文件，Wordle 将按相同结构创建：

```
examples/agentic/codebreaker/
  game.py              # 游戏规则、工具定义、评分函数
  dataset_generator.py # 可复现数据集生成
  dataset_loader.py    # 数据集加载器
  run_agent.py         # 多轮 agent 循环
  reward.py            # 奖励函数
  README.md            # 文档
  tui.py               # 终端 UI（可选）
```

### 2.3 Wordle 与 Codebreaker 的核心差异

| 维度 | Codebreaker | Wordle |
|------|------------|--------|
| 答案空间 | 4 位唯一数字（10*9*8*7=5040 种） | 5 字母英文单词（词表限定） |
| 反馈类型 | `exact`（位置对）+ `present`（存在但位置错） | `exact` + `present` + `absent`（不存在） |
| 重复元素 | 不允许（数字唯一） | **允许**（如 "eerie" 有 3 个 e） |
| 反馈复杂度 | 简单（集合交集减去精确匹配） | **需处理重复字母的配额计数** |
| 词表来源 | 数学组合（无外部依赖） | 需内嵌英文单词词表 |

### 2.4 AgentTrajectoryTurn 契约

每个示例的 `run_agent.py` 必须遵守的契约（来自 `areno/api/agentic.py:122`）：

```python
@dataclass(slots=True)
class AgentTrajectoryTurn:
    item: AgentItem           # 来自 batch.iter_samples()
    messages: list[dict]      # 发给模型的完整对话
    response: Any | None      # OpenAI 兼容端点返回的响应
    response_tokens: list[int]   # 自动从 response 提取
    response_logprobs: list[float]  # 自动从 response 提取
    parsed_tool_calls: list[dict]   # 自动从 response 提取
    model: str = "policy"
    tools: list[dict] = []
    tool_choice: Any = None
```

关键约束：
- `run_agent` 必须是 `async def run_agent(ctx, batch)`
- 必须使用 `AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, ...)`
- `model="policy"` 是固定值
- `stream=False`（agentic 端点不支持流式）
- 多轮时需手动把 `assistant_message` + `{"role":"tool",...}` 追加到 messages

---

## 3. 设计方案

### 3.1 架构概览

```
examples/agentic/wordle/
  game.py              ← 核心规则 + 工具定义 + 评分
  dataset_generator.py ← 数据集生成
  dataset_loader.py    ← 数据集加载器
  run_agent.py         ← 多轮 agent 循环
  reward.py            ← 奖励函数
  README.md            ← 用户文档
```

### 3.2 game.py 详细设计

#### 3.2.1 内嵌词表

使用 public domain / CC0 许可的常用 5 字母英文单词，约 50-100 个。

```python
# 词表许可：public domain（常用英文单词不具版权）
WORDLE_WORDS = [
    "about", "above", "abuse", "actor", "acute",
    "admit", "adopt", "adult", "after", "again",
    "agent", "agree", "ahead", "alarm", "album",
    # ... 约 80-100 个
]
WORDLE_LENGTH = 5
DEFAULT_MAX_GUESSES = 6
```

词表设计原则：
- 全部为 5 字母英文单词
- 常用词为主，避免过于生僻
- 确保至少有包含重复字母的词（如 "eerie", "speed", "llama"）
- 不含专有名词、缩写、俚语

#### 3.2.2 GUESS_TOOL 定义

```python
GUESS_TOOL = {
    "type": "function",
    "function": {
        "name": "guess_word",
        "description": "Guess a 5-letter word and receive Wordle feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]{5}$",
                    "description": "Exactly five English letters (case-insensitive).",
                }
            },
            "required": ["word"],
            "additionalProperties": False,
        },
    },
}
```

#### 3.2.3 猜测归一化

```python
def normalize_guess(value: object, *, word_length: int = WORDLE_LENGTH) -> str:
    """Return a validated lowercased guess."""
    guess = str(value).lower().strip()
    if len(guess) != word_length:
        raise ValueError(
            f"guess must be exactly {word_length} letters, got {len(guess)}"
        )
    if not guess.isalpha():
        raise ValueError(f"guess must contain only letters, got {guess!r}")
    return guess
```

#### 3.2.4 核心反馈逻辑（含重复字母处理）

这是 Wordle 与 Codebreaker 的核心区别。Wordle 的重复字母计数规则：

1. **先标记 exact**：逐位置比较，位置正确的标记为 `exact`，该位置的 secret 字母和 guess 字母都从候选中移除
2. **再标记 present**：对剩余非 exact 位置，如果 guess 字母在剩余 secret 字母中存在，标记为 `present` 并移除该 secret 字母
3. **其余为 absent**

```python
def score_guess(secret: str, guess: object) -> dict[str, Any]:
    """Score one guess with standard Wordle feedback including repeat-letter rules."""
    secret_lower = secret.lower()
    try:
        normalized_guess = normalize_guess(guess, word_length=len(secret_lower))
    except ValueError as exc:
        return {"valid": False, "error": str(exc), "guess": str(guess)}

    length = len(secret_lower)
    feedback: list[str] = ["absent"] * length
    secret_chars: list[str | None] = list(secret_lower)

    # Phase 1: mark exact matches
    for i in range(length):
        if normalized_guess[i] == secret_chars[i]:
            feedback[i] = "exact"
            secret_chars[i] = None  # consume this secret letter

    # Phase 2: mark present matches (handle repeat letters)
    for i in range(length):
        if feedback[i] == "exact":
            continue
        char = normalized_guess[i]
        if char in secret_chars:
            feedback[i] = "present"
            # consume the first occurrence of this letter
            secret_chars[secret_chars.index(char)] = None

    return {
        "valid": True,
        "guess": normalized_guess,
        "feedback": feedback,
        "solved": all(f == "exact" for f in feedback),
    }
```

**重复字母处理示例**（核心测试场景）：

```
secret = "eerie", guess = "eerie"
  → feedback = ["exact", "exact", "exact", "exact", "exact"]
  → solved = True

secret = "speed", guess = "erdey"
  Phase 1: s≠e, p≠r, e≠d, e≠e(exact!), d≠y → feedback=[_, _, _, "exact", _]
           secret_chars = ['s','p','e',None,'d']
  Phase 2:
    guess[0]='e' ∈ secret_chars? 'e' at index 2 → present, consume → secret_chars=['s','p',None,None,'d']
    guess[1]='r' ∈ secret_chars? no → absent
    guess[2]='d' ∈ secret_chars? 'd' at index 4 → present, consume → secret_chars=['s','p',None,None,None]
    guess[4]='y' ∈ secret_chars? no → absent
  → feedback = ["present", "absent", "present", "exact", "absent"]

secret = "llama", guess = "allay"
  Phase 1: l≠a, l≠l(exact!), a≠a(exact!), m≠l, a≠y → feedback=[_,"exact","exact",_,_]
           secret_chars = ['l',None,None,'m','a']
  Phase 2:
    guess[0]='a' ∈ secret_chars? 'a' at index 4 → present, consume → secret_chars=['l',None,None,'m',None]
    guess[3]='l' ∈ secret_chars? 'l' at index 0 → present, consume
    guess[4]='y' ∈ secret_chars? no → absent
  → feedback = ["present", "exact", "exact", "present", "absent"]
```

#### 3.2.5 Prompt 构建

```python
def make_prompt(record: dict[str, Any]) -> str:
    """Build one prompt without leaking the secret word."""
    max_guesses = int(record.get("max_guesses", DEFAULT_MAX_GUESSES))
    return (
        "Guess the hidden 5-letter English word. "
        f"You have at most {max_guesses} guesses. After each guess, you receive "
        "feedback for each position: 'exact' = correct letter in correct position, "
        "'present' = letter exists in the word but in a different position, "
        "'absent' = letter not in the word. "
        "Call guess_word once per turn and use the feedback to narrow down the word."
    )
```

#### 3.2.6 回合评分

```python
def score_episode(
    secret: str,
    guesses: list[object],
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
) -> float:
    """Reward solving efficiently, partial progress, and penalize invalid/repeated."""
    if not guesses:
        return -1.0
    valid_results = [score_guess(secret, guess) for guess in guesses[:max_guesses]]
    if any(not result["valid"] for result in valid_results):
        return -1.0
    for index, result in enumerate(valid_results, start=1):
        if result["solved"]:
            efficiency = (max_guesses - index) / max(max_guesses - 1, 1)
            return 0.8 + 0.2 * efficiency
    # Partial progress: best exact + present count
    best_information = max(
        result["feedback"].count("exact") + result["feedback"].count("present")
        for result in valid_results
    )
    return 0.1 * best_information / len(secret)
```

#### 3.2.7 确定性评估

```python
def evaluate_wordle(
    secret: str,
    guesses: list[object],
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
) -> dict[str, Any]:
    """Deterministic evaluation reporting solve status and guesses-to-solve."""
    valid_guesses = []
    for guess in guesses[:max_guesses]:
        result = score_guess(secret, guess)
        if not result["valid"]:
            break
        valid_guesses.append(result)
        if result["solved"]:
            return {
                "solved": True,
                "guesses_to_solve": len(valid_guesses),
                "word_length": len(secret),
            }
    return {
        "solved": False,
        "guesses_to_solve": None,
        "word_length": len(secret),
        "valid_guesses": len(valid_guesses),
    }
```

### 3.3 dataset_generator.py 设计

```python
"""Generate reproducible Wordle tasks."""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from game import DEFAULT_MAX_GUESSES, WORDLE_LENGTH, WORDLE_WORDS


def generate_records(count: int = 256, *, seed: int = 2026) -> list[dict]:
    """Return deterministic Wordle secrets drawn from the bundled word list."""
    rng = random.Random(seed)
    records = []
    pool = rng.sample(WORDLE_WORDS, min(count, len(WORDLE_WORDS)))
    # If count > pool size, allow repeats with different IDs
    while len(records) < count:
        idx = len(records) % len(pool)
        records.append({
            "id": f"wordle-{len(records) + 1:05d}",
            "secret": pool[idx],
            "word_length": WORDLE_LENGTH,
            "max_guesses": DEFAULT_MAX_GUESSES,
        })
    return records
```

### 3.4 dataset_loader.py 设计

```python
"""Dataset loader for Wordle agentic training."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DEFAULT_MAX_GUESSES, WORDLE_LENGTH, make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize records while remaining tokenizer independent."""
    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        record["word_length"] = int(record.get("word_length", WORDLE_LENGTH))
        record["max_guesses"] = min(
            max(int(record.get("max_guesses", DEFAULT_MAX_GUESSES)), 1), 6
        )
        record["secret"] = str(record["secret"]).lower()
        record["id"] = str(record.get("id", f"wordle-{index:05d}"))
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records
```

### 3.5 run_agent.py 设计

以 codebreaker 的 `_run_episode` 为模板，适配 Wordle 的 `guess_word` 工具和反馈格式：

```python
"""Bounded multi-turn agent loop for Wordle."""

from __future__ import annotations
import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import GUESS_TOOL, score_guess  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a Wordle solver. On every guessing turn call guess_word exactly once. "
    "Use the exact/present/absent feedback from prior guesses to deduce the hidden word. "
    "Never repeat a guess. After the game ends, summarize the outcome without a tool call."
)


async def run_agent(ctx, batch):
    """Run bounded concurrent Wordle episodes and preserve exact model outputs."""
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Wordle requires `openai` and `httpx`. Install: pip install openai"
        ) from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )
    try:
        grouped = await asyncio.gather(
            *(_run_episode(item, client) for item in items)
        )
        return AgentTrajectory(
            turns=[turn for episode in grouped for turn in episode]
        )
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    """One Wordle episode: up to max_guesses tool calls + a final summary."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.prompt},
    ]
    turns = []
    max_guesses = min(max(int(item.record["max_guesses"]), 1), 6)

    for guess_number in range(1, max_guesses + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Guess {guess_number} of {max_guesses}: call guess_word now."},
        ]
        tool_choice = {"type": "function", "function": {"name": "guess_word"}}
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=[GUESS_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=[GUESS_TOOL],
                tool_choice=tool_choice,
            )
        )

        assistant_message = _assistant_message(response)
        tool_result = _execute_guess(assistant_message, item.record)

        if tool_result is None:
            logger.warning("Wordle model returned no executable guess_word call")
            break

        messages.extend(_tool_messages(assistant_message, tool_result))

        game_over = (
            tool_result.get("solved")
            or not tool_result.get("valid")
            or guess_number == max_guesses
        )
        if game_over:
            finish_messages = [
                *messages,
                {"role": "user", "content": "The game is over. Briefly summarize the outcome without calling a tool."},
            ]
            finish_response = await client.chat.completions.create(
                model="policy",
                messages=finish_messages,
                stream=False,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=finish_messages,
                    response=finish_response,
                )
            )
            break

    return turns


def _assistant_message(response) -> dict:
    """Extract the assistant message from an OpenAI chat completion response."""
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (message.tool_calls or [])
        ],
    }


def _execute_guess(assistant_message: dict, record: dict) -> dict | None:
    """Execute the guess_word tool call and return the result dict."""
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "guess_word":
        return None
    try:
        arguments = json.loads(calls[0]["function"].get("arguments") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict) or "word" not in arguments:
        return None
    return score_guess(record["secret"], arguments["word"])


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    """Build the assistant + tool messages to append to the conversation."""
    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "guess_word",
            "content": json.dumps(tool_result),
        },
    ]
```

**与 codebreaker 的关键差异**：
- 工具名从 `guess_code` → `guess_word`
- 工具参数从 `code` → `word`
- `score_guess` 的返回结构包含 `feedback` 列表（exact/present/absent），而非 codebreaker 的 `exact`/`present` 整数
- system prompt 描述 Wordle 的三态反馈规则

### 3.6 reward.py 设计

```python
"""Outcome and process reward for Wordle trajectories."""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_episode  # noqa: E402


def reward_fn(record) -> float:
    """Reward valid, non-repeated deduction and efficient success."""
    source = dict(record.source_record)
    guesses = []
    for call in record.tool_calls:
        if call.get("name") != "guess_word":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return -1.0
        if not isinstance(arguments, dict) or "word" not in arguments:
            return -1.0
        guesses.append(arguments["word"])

    # Penalize repeated guesses
    if len(guesses) != len(set(map(str, guesses))):
        return -0.5

    return score_episode(
        source["secret"],
        guesses,
        max_guesses=int(source["max_guesses"]),
    )
```

---

## 4. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `examples/agentic/wordle/game.py` | 新增 | 核心规则、工具定义、评分函数 |
| `examples/agentic/wordle/dataset_generator.py` | 新增 | 可复现数据集生成 |
| `examples/agentic/wordle/dataset_loader.py` | 新增 | 数据集加载器 |
| `examples/agentic/wordle/run_agent.py` | 新增 | 多轮 agent 循环 |
| `examples/agentic/wordle/reward.py` | 新增 | 奖励函数 |
| `examples/agentic/wordle/README.md` | 新增 | 用户文档 |
| `tests/test_wordle_game_cpu.py` | 新增 | CPU 测试 |
| `docs/cookbook/wordle-agentic-rl.rst` | 新增 | Cookbook 文档 |

**不修改任何现有文件**——这是纯新增示例，完全向后兼容。

---

## 5. 测试计划

### 5.1 测试文件

`tests/test_wordle_game_cpu.py`

遵循项目测试约定：`unittest.TestCase` + 每方法写 docstring。

### 5.2 测试用例

| 编号 | 测试名 | 验证内容 |
|------|--------|---------|
| T1 | `test_correct_guess_all_exact` | secret="eerie", guess="eerie" → feedback 全 "exact", solved=True |
| T2 | `test_invalid_length` | guess="abcd"（4字母） → valid=False |
| T3 | `test_invalid_non_alpha` | guess="ab1cd" → valid=False |
| T4 | `test_repeat_letters_exact` | secret="eerie", guess="eerie" → 全 exact（重复字母正确处理） |
| T5 | `test_repeat_letters_present_quota` | secret="speed", guess="eerie" → e 只有两处 present（配额限制）|
| T6 | `test_repeat_letters_mixed` | secret="llama", guess="allay" → 正确的 exact/present 组合 |
| T7 | `test_all_absent` | secret="about", guess="frown" → 无匹配字母，全 absent |
| T8 | `test_guess_exhausted` | 6 次未猜中 → evaluate 返回 solved=False |
| T9 | `test_score_episode_solved` | 解出 → 返回 0.8 + 0.2 * efficiency |
| T10 | `test_score_episode_partial` | 未解出但有部分匹配 → 返回 0.1 * best_info / len |
| T11 | `test_score_episode_invalid` | 含无效猜测 → 返回 -1.0 |
| T12 | `test_score_episode_no_guesses` | 空猜测列表 → 返回 -1.0 |
| T13 | `test_evaluate_wordle_solved` | 解出 → guesses_to_solve 正确 |
| T14 | `test_evaluate_wordle_not_solved` | 未解出 → solved=False, guesses_to_solve=None |
| T15 | `test_word_list_all_length_5` | 词表中所有单词都是 5 字母 |
| T16 | `test_normalize_guess_case_insensitive` | "EERIE" → "eerie" |
| T17 | `test_normalize_guess_with_whitespace` | " eerie " → "eerie" |

### 5.3 重复字母测试的详细预期

T5 的详细推导：
```
secret = "speed", guess = "eerie"
Phase 1: s≠e, p≠e, e≠r, e≠i, d≠e → 无 exact
         secret_chars = ['s','p','e','e','d']
Phase 2:
  guess[0]='e' → 'e' 在 secret_chars 中 (index 2) → present, consume → ['s','p',None,'e','d']
  guess[1]='e' → 'e' 在 secret_chars 中 (index 3) → present, consume → ['s','p',None,None,'d']
  guess[2]='r' → 'r' 不在 secret_chars 中 → absent
  guess[3]='i' → 'i' 不在 secret_chars 中 → absent
  guess[4]='e' → 'e' 不在剩余 secret_chars 中 → absent
→ feedback = ["present", "present", "absent", "absent", "absent"]
```

关键验证点：guess 中有 3 个 `e`，但 secret "speed" 只有 2 个 `e`，所以只有 2 个 present，第 3 个 `e` 是 absent。这验证了配额计数正确性。

---

## 6. 文档计划

### 6.1 README.md

```markdown
# Agentic Wordle

Wordle is a deterministic multi-turn word-guessing game. A hidden 5-letter
English word must be guessed within 6 attempts. After each guess, the
`guess_word` tool returns per-position feedback:

- `exact`: correct letter in correct position
- `present`: letter exists in the word but at a different position
- `absent`: letter not in the word

The secret never appears in the prompt or tool result. Repeated letters are
handled according to standard Wordle counting rules (each secret letter can
only match once). The bundled word list is public domain.

## Generate data

    python examples/agentic/wordle/dataset_generator.py \
      --output /tmp/wordle.jsonl --count 256 --seed 2026

## Train

    areno train \
      --ckpt Qwen/Qwen3-0.6B \
      --dataset-path /tmp/wordle.jsonl \
      --dataset-loader-fn examples/agentic/wordle/dataset_loader.py \
      --reward-fn-path examples/agentic/wordle/reward.py \
      --agent-fn examples/agentic/wordle/run_agent.py \
      --algo gspo --tp-size 1 --world-size 1 \
      --batch-size 1 --n-samples 2 --max-new-tokens 64
```

### 6.2 Cookbook 文档

`docs/cookbook/wordle-agentic-rl.rst`：

```rst
Wordle agentic RL recipe
========================

This recipe runs a small Wordle word-guessing RL task with a bundled
word list, a ``guess_word`` tool with exact/present/absent feedback,
and a multi-turn agent loop.

.. code-block:: bash

   python examples/agentic/wordle/dataset_generator.py \
     --output /tmp/wordle.jsonl --count 256 --seed 2026

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/wordle.jsonl \
     --dataset-loader-fn examples/agentic/wordle/dataset_loader.py \
     --reward-fn-path examples/agentic/wordle/reward.py \
     --agent-fn examples/agentic/wordle/run_agent.py \
     --algo gspo --tp-size 1 --world-size 1

Key files:

* ``examples/agentic/wordle/game.py`` implements Wordle rules and scoring.
* ``examples/agentic/wordle/run_agent.py`` runs the multi-turn agent loop.
* ``examples/agentic/wordle/reward.py`` scores trajectories.
* :doc:`/cli/training` documents rollout, loss, and checkpoint flags.
```

---

## 7. 实施顺序

| 步骤 | 内容 | 依赖 |
|------|------|------|
| 1 | 创建 `examples/agentic/wordle/game.py`（核心规则+工具+评分） | 无 |
| 2 | 创建 `tests/test_wordle_game_cpu.py` 并运行 | 步骤 1 |
| 3 | 创建 `dataset_generator.py` | 步骤 1 |
| 4 | 创建 `dataset_loader.py` | 步骤 1 |
| 5 | 创建 `reward.py` | 步骤 1 |
| 6 | 创建 `run_agent.py` | 步骤 1 |
| 7 | 创建 `README.md` | 步骤 3-6 |
| 8 | 创建 `docs/cookbook/wordle-agentic-rl.rst` | 步骤 7 |
| 9 | 运行全部测试验证 | 步骤 2-8 |
