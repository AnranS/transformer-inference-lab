# 自回归生成循环

> 对应 [`day03-04.md`](../day03-04.md) 的 Day 4：先实现一个无 KV Cache 的正确 baseline，
> 再量化它为什么慢。

## 一轮生成做什么

给定当前 token 序列 `input_ids [B, S]`：

```text
input_ids [B, S]
  → model
  → logits [B, S, V]
  → logits[:, -1, :]                 [B, V]
  → argmax(dim=-1, keepdim=True)     [B, 1]
  → cat 到序列末尾                  [B, S + 1]
  → 下一轮
```

最后一位 logits 预测的是还不存在的下一个 token。生成出的 token 必须拼回输入，
因为下一轮预测要以它作为上下文。

## 为什么必须逐个生成

第 `t + 1` 个 token 的分布依赖前 `t` 个 token，其中第 `t` 个 token 是上一轮才生成的。
因此单条序列的解码依赖链是串行的，不能一次并行得到整段回答。

训练不同：teacher forcing 下，整句真实 token 已知；模型能在一次前向中为所有位置同时算 logits
并计算 loss。

## 贪心选择与停止条件

贪心生成每轮选择词表分数最大的 token：

```python
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
```

停止条件：

1. 已生成 `max_new_tokens` 个新 token；
2. 生成 EOS。当前 Day 4 简化实现只在 **batch 内所有序列** 都生成 EOS 时停止。
   若部分序列先结束，它们仍会继续生成；Day 21 用每条序列自己的 `finished` mask 解决。

`eos_token_id=None` 表示不启用 EOS 停止。

## 无 KV Cache 为什么浪费

设 prompt 长度为 `P`，要生成 `N` 个新 token。

无 Cache 时，每一轮把完整历史重新送进模型，累计处理的 token-位置数为：

```text
P + (P + 1) + ... + (P + N - 1)
```

理想的 Cache 路径会先 prefill 一次 prompt，然后每个后续 decode 步只处理最新 token：

```text
P + (N - 1)
```

| `P` | `N` | 无 Cache | 有 Cache | 倍数 |
|---:|---:|---:|---:|---:|
| 10 | 20 | 390 | 29 | 13.4× |
| 1024 | 100 | 107,350 | 1,123 | 95.6× |

Day 4 notebook 验证了：把 token 追加到末尾后，旧前缀的 `hidden_states` 逐元素不变；
重算它们不会产生新信息。

本项目当前 `TinyLM` 尚未实现 Attention，因此此处的重复计算只包含 embedding 查表和 LM Head。
真实 Transformer 的 Attention 还会处理随序列长度增长的中间结果，KV Cache 的收益会更关键。

## 与 `forward_last_position` 的区别

`forward_last_position` 只避免为前 `S - 1` 个位置计算无用的 **LM Head logits**。
它不避免重新计算前缀 token 的 hidden states。

KV Cache 解决的是后者：缓存前缀的 K/V，使解码阶段不必重复处理已经见过的上下文。
