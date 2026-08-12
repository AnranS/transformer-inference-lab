# 概念笔记索引

概念文件名前缀用于保持稳定引用，不等同于学习单元编号。请按下面的学习顺序阅读：

| 学习单元 | 主题 | 笔记 | 配套任务 |
|---|---|---|---|
| Day 1 | Tokenizer | [00-tokenizer.md](./00-tokenizer.md) | [day01-02.md](../day01-02.md) |
| Day 2 | Token Embedding | [01-embedding.md](./01-embedding.md) | [day01-02.md](../day01-02.md) |
| Day 3 | LM Head 与 logits | [02-lm-head.md](./02-lm-head.md) | [day03-04.md](../day03-04.md) |
| Day 4 | 自回归生成循环 | [03-autoregressive-loop.md](./03-autoregressive-loop.md) | [day03-04.md](../day03-04.md) |
| Day 5 | 广播与数值容差 | [05-numerical-tolerance.md](./05-numerical-tolerance.md) | [day05.md](../day05.md) |
| Day 6～7 | Q/K/V 与 Scaled Dot-Product Attention | [04-attention-qkv.md](./04-attention-qkv.md) | [day06.md](../day06.md)、[day07-08.md](../day07-08.md) |
| Day 8 | Causal 与 Padding Mask | [06-masks.md](./06-masks.md) | [day07-08.md](../day07-08.md) |

Day 9 的多头实现与验证见：

- [day09.md](../day09.md)
- [`day09_multi_head_attention.ipynb`](../../notebooks/day09_multi_head_attention.ipynb)
- [`attention.py`](../../src/mini_transformer/attention.py)
- [`test_attention.py`](../../tests/test_attention.py)

后续概念笔记会随学习进度补充，不提前创建空文件。
