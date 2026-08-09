# 自回归语言模型：从文本到下一个 Token

> 覆盖 Day 01～05：Tokenizer、Embedding、LM Head、贪心生成、广播与数值验证。
> 这里的目标不是介绍完整 Transformer，而是串起目前已经实现和验证过的最小推理链路。

## 1. 从文本到下一个 token

```text
文本
  → tokenizer.encode
  → input_ids [B, S]（torch.long，离散索引）
  → TokenEmbedding
  → hidden_states [B, S, D]（float，连续向量）
  → LM Head
  → logits [B, S, V]（float，未归一化词表分数）
  → logits[:, -1, :]
  → [B, V]
  → argmax(dim=-1, keepdim=True)
  → next_token_id [B, 1]
  → 拼回 input_ids
  → 下一轮
```

`B` 是 batch size，`S` 是当前序列长度，`D` 是 hidden size，`V` 是词表大小。
每生成一个 token，下一轮输入从 `[B, S]` 变为 `[B, S + 1]`。

## 2. 五个容易混淆的概念

| 概念 | 形状 | dtype | 含义 |
|---|---|---|---|
| token ID / `input_ids` | `[B, S]` | `torch.long` | 离散词表索引；数值大小没有连续语义 |
| embedding | `[B, S, D]` | float | 根据 token ID 从 `[V, D]` 表查出的向量 |
| hidden state | `[B, S, D]` | float | 模型某一层对 token 的连续表示 |
| logits | `[B, S, V]` | float | 每个位置对全词表的未归一化分数；可为负，和不必为 1 |
| probability | `[B, S, V]` | float | 对 logits 做 softmax 后的概率；每行落在 `[0,1]` 且和为 1 |

Embedding 是第 0 层的 hidden state。完整 Decoder 模型中的每个 Transformer Block 都会继续更新
hidden states，但形状保持 `[B, S, D]`。

## 3. 当前项目的贪心自回归伪代码

```python
output_ids = input_ids

with torch.inference_mode():
    for _ in range(max_new_tokens):
        logits = model(output_ids)                         # [B, S, V]
        next_token = logits[:, -1, :].argmax(             # [B, 1]
            dim=-1,
            keepdim=True,
        )
        output_ids = torch.cat([output_ids, next_token], dim=1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
```

这版实现是无 KV Cache baseline：每轮都会把全部历史 token 重新送入模型。
它故意保留低效路径，供 Day 26～30 实现 KV Cache 后比较。

## 4. 为什么第 t 个 token 依赖之前所有 token

语言模型按链式法则建模联合概率：

```text
P(x₁, x₂, ..., xₜ) = Π P(xᵢ | x₁, ..., xᵢ₋₁)
```

因此要选择第 `t` 个 token，模型必须以之前的 token 作为条件。生成出的 token 又会成为下一轮的
上下文，所以单条序列的生成依赖链是串行的：不能在不知道第 `t` 个 token 时先生成第 `t+1` 个。

完整 Transformer 的计算层面由 causal attention 保证：位置 `t` 只能读取 `0..t` 的位置，
因此该位置的 hidden state 可以编码可见历史。

### 当前最小模型的缺口

当前 `TinyLM` 没有位置编码，也没有 Attention。它的 hidden state 只来自 token embedding，
实际上只依赖当前位置的 token，而不依赖前文。因此生成的 token 没有语言意义。

目前验证的是输入输出形状、LM Head、生成循环与推理接口的正确性；Day 06～09 加入 Attention，
Day 14 加入 RoPE 后，才会补齐“读取历史上下文”的真正机制。

## 5. 两类优化不要混淆

- `forward_last_position`：避免为前 `S - 1` 个位置计算无用的 LM Head logits；
- KV Cache：避免在每个 decode 步重复计算旧 token 的 Attention K/V 和相关 hidden states。

无 Cache 时，prompt 长度为 `P`、生成 `N` 个 token 的累计 token-位置处理量为：

```text
P + (P + 1) + ... + (P + N - 1)
```

理想 Cache 路径为：

```text
P + (N - 1)
```

例如 `P=1024`、`N=100` 时，前者为 107,350，后者为 1,123，约相差 95.6 倍。
详见 [`concepts/03-autoregressive-loop.md`](./concepts/03-autoregressive-loop.md)。
