# Embedding 笔记

## 形状约定

| 对象 | 形状 | 含义 |
|------|------|------|
| Embedding 权重 | `[V, D]` | 词表里每一行是一个 token 的向量 |
| 输入 `token_ids` | `[B, S]` | batch 里每条序列的 token ID |
| 输出 `hidden_states` | `[B, S, D]` | 每个位置查表得到的向量 |

- `V`：Vocabulary size，词表大小
- `D`：Hidden size，隐藏维度
- `B`：Batch size
- `S`：Sequence length

## 本质

Embedding **不是**对 token ID 做数学运算（乘系数、比大小），而是：

```text
token_id = 7
→ 取出 weight[7] 这一行，形状 [D]
```

所以：

1. token ID 是离散索引，`100` 并不比 `10`「语义更强」
2. 相同 token ID → 一定得到相同向量（权重固定时）
3. Token Embedding **不知道位置**：句首的 `the` 和句中的 `the` 查到同一行；位置信息要靠后面的 Positional Embedding 补

## 为什么是访存密集，不是计算密集

教科书说 embedding 是「one-hot 向量乘权重矩阵」，这在数学上完全成立——
`[B, S, V] @ [V, D]` 和查表的结果**逐比特相同**（误差 0，因为 `0 * x` 精确为 0，累加 0 也精确）。

但两条路的代价差几个数量级。以 GPT-2 尺度（`V=50257, D=768`，序列长 1024）为例：

| | one-hot 矩阵乘 | 查表 |
|---|---|---|
| 中间张量 | 206 MB | 无 |
| 计算量 | 79 GFLOP | 0 FLOP |
| 访存 | 206 MB + 权重 | 3.1 MB |

所以 embedding 层的开销全在「从大表里随机抽几千行」这个访存动作上。
做性能分析时它不会出现在 FLOP 统计里，只会出现在访存带宽和 cache 命中率的讨论中。

## 项目约定：强制 `[B, S]`

`nn.Embedding` 本身对输入形状毫无要求（文档写的是「输入 `(*)`，输出 `(*, H)`」），
`[B, S] → [B, S, D]` 是本项目自己立的规矩，由 `TokenEmbedding` 主动检查。

立这条规矩是因为**少一个 batch 维不会让程序崩溃，只会让它悄悄算错**：

```text
正确 [1, 5] -> (1, 5, 16)      (B, S, D)
漏了  [5]   -> (5, 16)         (S, D)   ← torch 一声不吭

hidden.mean(dim=1)：
  正确 -> (1, 16)   对 5 个 token 求平均
  错误 -> (5,)      对 16 个特征求平均，语义全错，但不报错
```

`dim=1` 在两种形状下分别指「序列维」和「特征维」。加位置编码时更隐蔽：
`[5,16] + [5,16]` 靠广播照样算得出来，连形状不匹配的报错都收不到。

一维输入还有歧义：`[5]` 是「1 条 5 token 的序列」还是「5 条 1 token 的序列」？
强制二维等于逼调用方把意图写清楚。

## padding_idx

`nn.Embedding(V, D, padding_idx=0)` 会把第 0 行初始化为全零，且训练时**不接收梯度**。
对应 tokenizer 笔记里的 PAD token——占位符不该学到任何东西。

两个容易误解的点：

1. 「全零」只是初始值，可以手动改成别的，它依然不吃梯度
2. 对纯推理必要性不高：PAD 位置会被 `attention_mask` 屏蔽，何况 GPT-2 原生没有 PAD token

## 权重初始化与真实体积

PyTorch 默认从 **N(0, 1)** 初始化，而 GPT-2 的 `initializer_range` 是 **0.02**，差 50 倍。
加载预训练权重时随机值会被整个覆盖，所以不影响；但自己初始化跑前向若数值炸掉，先查这里。

GPT-2 small 的 token embedding 是 `50257 × 768 = 38.6M` 参数（fp32 约 147 MB，fp16 约 74 MB），
占全模型 124M 参数的三成。

## 权重共享（weight tying）

GPT-2 的输出层 `lm_head` 和输入的 token embedding 是**同一个** `[V, D]` 矩阵。
同一份权重在推理时被用两次，性能特征却截然相反：

- 开头查表：纯访存，0 FLOP
- 结尾算 logits：`[B, S, D] @ [D, V]`，是全模型最大的 GEMM 之一

## 和流水线的关系

```text
文本
  --tokenizer.encode-->  input_ids [B, S]  (torch.long)
  --TokenEmbedding---->  hidden_states [B, S, D]
```

动手验证见 [`../../notebooks/day01_tokenizer_playground.ipynb`](../../notebooks/day01_tokenizer_playground.ipynb)
的「3. Embedding 查表实验」一节。
