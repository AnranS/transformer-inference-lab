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

## 和流水线的关系

```text
文本
  --tokenizer.encode-->  input_ids [B, S]  (torch.long)
  --TokenEmbedding---->  hidden_states [B, S, D]
```
