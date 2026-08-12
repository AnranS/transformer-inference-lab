# 学习单元 Day 16：ModelConfig

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 16**｜学习日 11

零件都齐了，但尺寸参数还散落在各个模块的构造函数里。今天把它们集中到一处，
并建立**尺寸约束的自动校验**。

一条铁律，从今天起生效：

> **所有模块只从 config 读取尺寸，禁止散落魔法数字。**

> 如果学习日 10 的 `docs/deliverables/03-modern-decoder-block.md` 没写完，今天先补——
> 今天只有 3 小时任务，有空间。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习 HF `LlamaConfig` 字段 | 阅读笔记 |
| ⬜ | 任务 2：实现 `config.py` | `src/mini_transformer/config.py` |
| ⬜ | 任务 3：尺寸约束校验 | 同上 |
| ⬜ | 任务 4：参数量公式与验证 | `docs/concepts/11-model-sizing.md` |
| ⬜ | 任务 5：单元测试 | `tests/test_config.py` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF LlamaConfig](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig) | [HF 配置基类](https://huggingface.co/docs/transformers/main_classes/configuration) | 明确所有尺寸参数及它们之间的约束 |

---

## 任务 1：学习 HF `LlamaConfig` 字段（30 分钟）

`config.py` 至少包含这 11 个字段。逐个确认自己知道它控制什么：

| 字段 | 控制什么 | 用在哪 |
|---|---|---|
| `vocab_size` | 词表大小 `V` | Embedding、LM Head |
| `hidden_size` | 隐藏维度 `D` | 到处 |
| `intermediate_size` | MLP 中间维度 `I` | SwiGLU |
| `num_hidden_layers` | Block 层数 `L` | 模型主循环 |
| `num_attention_heads` | Query head 数 `Hq` | Attention |
| `num_key_value_heads` | KV head 数 `Hkv` | Attention（GQA） |
| `head_dim` | 单头维度 `Dh` | Attention、RoPE |
| `max_position_embeddings` | 最大序列长度 | RoPE 缓存、Cache 预分配 |
| `rms_norm_eps` | RMSNorm 的 `eps` | RMSNorm（Llama 用 `1e-6`） |
| `rope_theta` | RoPE 的 base | RoPE（Llama-2 是 `10000`） |
| `tie_word_embeddings` | 是否共享 Embedding 与 LM Head | Day 3 已实现 |

**注意 `head_dim` 的历史变化**：早期 Llama 里它总是 `hidden_size // num_attention_heads`，
是个派生量；Llama-3 之后 HF 允许**显式指定**，可以不等于那个商。
我们也做成显式字段，但**默认值取那个商**——既跟得上 HF，也不给自己制造麻烦。

## 任务 2：实现 `config.py`（45 分钟）

用 `dataclass`，不要用裸 dict——需要类型提示和 `__post_init__` 做校验。

```python
@dataclass
class ModelConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 384
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int | None = None          # None → hidden_size // num_attention_heads
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
```

默认值就是 roadmap 给的**小配置**（`V=256, D=128, I=384, L=2, Hq=Hkv=4, max_S=128`），
这样 `ModelConfig()` 直接就是测试用的玩具模型。

另外提供两个便利属性：

```text
num_key_value_groups  = num_attention_heads // num_key_value_heads
```

`num_key_value_groups` 是 GQA 里「每个 KV head 服务几个 Query head」，
Day 31 的 `repeat_kv` 会用到。名字和 HF 保持一致。

### 改造已有模块

把 `embedding.py`、`norm.py`、`mlp.py`、`rope.py`、`attention.py`、`block.py`
的构造函数改成接收 `ModelConfig`。**Day 3 的 `tiny_lm.py` 暂时保留**，
Day 17 会被 `model.py` 取代。

这一步会让 `pytest` 大面积变红——**这是好事**，说明测试真的在测东西。
逐个修，别一次改完再跑。

## 任务 3：尺寸约束校验（35 分钟）

在 `__post_init__` 里检查。每条都要有**清晰的错误信息，带上实际值**（沿用 Day 01～04 的习惯）。

| 约束 | 为什么 | 违反的后果 |
|---|---|---|
| 所有尺寸 > 0 | 基本卫生 | 各种诡异形状 |
| `head_dim` 是偶数 | RoPE 要成对旋转 | Day 14 已检查过，这里前移 |
| `num_attention_heads % num_key_value_heads == 0` | GQA 要能均分 | `repeat_kv` 无法整除 |
| `num_key_value_heads <= num_attention_heads` | `Hkv > Hq` 无意义 | 逻辑错误 |
| `head_dim * num_attention_heads == hidden_size` | 拆分/合并要对得上 | **见下** |

最后一条要**留一个开关**。Llama-3 允许 `head_dim` 独立于 `hidden_size`，
此时 Attention 的输出投影是 `nn.Linear(Hq*Dh, D)` 而不是 `nn.Linear(D, D)`。
我们的实现如果只支持相等的情况，就**明确拒绝**不相等的配置，
而不是默默算错：

```text
head_dim × num_attention_heads (= X) 必须等于 hidden_size (= Y)。
当前实现不支持两者不等（Llama-3 允许，本项目 Day 31 后再考虑）。
```

**明确拒绝比悄悄支持一半更安全。** 这是今天最重要的工程判断。

## 任务 4：参数量公式与验证（50 分钟）

### 推导公式

自己从各模块的权重形状推一遍，写进 `docs/concepts/11-model-sizing.md`：

```text
每层 Attention   2 · D · Dh · (Hq + Hkv)
每层 MLP         3 · D · I
每层 2×RMSNorm   2 · D
─────────────────────────────────────
每层合计         上面三项之和
× L 层

Embedding        V · D
final RMSNorm    D
LM Head          tied → 0 ；untied → V · D
```

Attention 那一项的来源：`Wq [D, Hq·Dh]`、`Wk [D, Hkv·Dh]`、`Wv [D, Hkv·Dh]`、
`Wo [Hq·Dh, D]`，加起来 `D·Dh·(Hq + 2·Hkv + Hq) = 2·D·Dh·(Hq + Hkv)`。

### 用真实模型验证公式

**这是今天最有价值的练习。** 代入 Llama-3 8B 的配置：

```text
D=4096, I=14336, L=32, Hq=32, Hkv=8, Dh=128, V=128256, untied
```

我已经验证过结果，你自己算完再核对：

```text
单层 Attention = 2 × 4096 × 128 × (32+8) =    41.9M
单层 MLP       = 3 × 4096 × 14336        =   176.2M
单层 2×RMSNorm = 2 × 4096                =   0.008M
单层合计                                  =   218.1M
× 32 层                                   =    6.980B
Embedding      = 128256 × 4096           =   525.3M
LM Head（untied）                          =   525.3M
─────────────────────────────────────────────────────
总计                                      =    8.030B    ← 和官方 8.03B 一致
（若 tied）                                =    7.505B
```

**公式能算出官方参数量，说明你对每个权重的形状都理解正确。**
如果差得多，逐项排查——差 `V·D` 说明 tie 搞错了，差 `L·3·D·I` 说明 MLP 少算了一个投影。

### 再算自己的小配置

`ModelConfig()` 默认值下手算一遍，然后和实测对比：

```python
sum(p.numel() for p in model.parameters())
```

Day 17 组装出 `model.py` 后立刻做这个对比。

### 顺带记录 KV Cache 的量级

Day 32 会正式算，今天先把公式记下来：

```text
KV bytes = 2 × L × B × S × Hkv × Dh × bytes_per_element
```

代入 Llama-3 8B、`B=1`、`S=8192`、bf16：**恰好 1.00 GiB**。
（`2 × 32 × 1 × 8192 × 8 × 128 × 2` 字节）

注意这个量级：**KV Cache 会轻易超过权重本身**。
`B=16, S=8192` 时是 16 GiB，而模型权重 bf16 只有约 15 GiB。

## 任务 5：单元测试（20 分钟）

`tests/test_config.py`：

1. 默认构造成功，且 `head_dim` 被正确推导为 `hidden_size // num_attention_heads`
2. 显式给 `head_dim` 时以显式值为准
3. `num_attention_heads % num_key_value_heads != 0` 时报错
4. `head_dim` 为奇数时报错
5. `head_dim * num_attention_heads != hidden_size` 时报错，且错误信息包含两个实际值
6. 任意尺寸为 0 或负数时报错
7. `num_key_value_groups` 计算正确

---

## 过关标准

- [ ] 11 个字段的作用都能说清
- [ ] 所有模块改为从 `ModelConfig` 读尺寸，无魔法数字
- [ ] 五条尺寸约束都有校验，错误信息带实际值
- [ ] 参数量公式能算出 Llama-3 8B 的 8.03B
- [ ] 小配置的手算参数量与实测一致（Day 17 完成后核对）
- [ ] 记下了 KV Cache 公式，并知道它能超过权重体积
- [ ] `pytest -q` 全部通过

---

## 今日最重要的面试式问题

**给你一个模型配置，怎么估参数量？**

```text
L × [ 2·D·Dh·(Hq+Hkv)  +  3·D·I  +  2·D ]  +  V·D  +  D  +  (tied ? 0 : V·D)
```

关键是记住每个权重的形状来源，而不是背公式。

追问：**为什么 GQA 减少的是 `Wk`/`Wv` 的参数，却主要为了省显存？**

因为 `Wk`/`Wv` 从 `[D, Hq·Dh]` 缩到 `[D, Hkv·Dh]`，参数省下的量按上面公式是
`2·D·Dh·(Hq-Hkv)`——在 8B 模型里约 25M，占比很小。

真正的收益在 **KV Cache**：`2·L·B·S·Hkv·Dh·bytes` 里 `Hkv` 直接是一次项，
`Hq=32 → Hkv=8` 让 Cache **直接变成 1/4**。
8B 模型在 8K context 下从 4 GiB 降到 1 GiB。**GQA 是显存优化，不是参数优化。**
