# 张量约定（Day 01）

## 符号

| 符号 | 含义 |
|------|------|
| `B` | Batch size，一次喂给模型的样本数 |
| `S` | Sequence length，序列长度（含特殊 token / padding 后） |
| `V` | Vocabulary size，词表大小 |
| `D` | Hidden size，隐藏维度 |

## 形状与类型

| 张量 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[B, S]` | `dtype=torch.long`，离散 token 索引 |
| `attention_mask` | `[B, S]` | `1`=有效 token，`0`=PAD |
| embedding weight | `[V, D]` | 每一行是一个 token 的向量 |
| `hidden_states` | `[B, S, D]` | Embedding 查表后的连续表示 |

## 数据流

```text
文本
  → tokenizer → input_ids [B, S]  (long)
  → TokenEmbedding(weight [V, D]) → hidden_states [B, S, D]
```

Embedding 细节见 [`01-embedding.md`](./01-embedding.md)。

## 今日实验结论

1. **中文 / 英文 / 代码的 token 数因 tokenizer 而异**  
   同一句「你好，世界！」，Qwen（BPE）约 4 个 token，mBERT（WordPiece）约 6 个；英文与代码也会因词表合并方式不同而长短不一。比较「上下文能塞多少字」时，必须指定具体 tokenizer。

2. **left / right padding 只改位置与 mask，不改有效内容**  
   `padding_side="right"` 时 PAD 与 `mask=0` 在右侧；`"left"` 时在左侧。用 `attention_mask == 1` 取出的有效 `input_ids` 在两种设置下应相同。

3. **token ID 不能当作连续数值直接输入模型**  
   ID 只是查表下标：`100` 并不比 `10`「大十倍语义」。必须先经过 Embedding，把离散索引映射成 `[D]` 维向量，模型才能做加减乘等运算。

## 面试题

**如果 token ID 100 比 token ID 10 大十倍，是否代表语义更强？**

不代表。Token ID 只是查找 Embedding 行的离散索引。
