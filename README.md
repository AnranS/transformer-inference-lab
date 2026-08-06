# Transformer Inference Lab

从零手写 Transformer 推理引擎，边写边测，最终跑通 GPT-2。

## 主线

```text
文本 → input_ids → Embedding 查表 → hidden_states
```

## 文档怎么分

| 类型 | 目录 | 作用 |
|------|------|------|
| **每日任务** | `docs/dayXX.md` | 当天要做什么、过关标准、进度 |
| **概念笔记** | `docs/concepts/` | 可复用的知识点，不和某一天任务绑死 |

当前任务：[docs/day01.md](docs/day01.md)

## 目录

```text
docs/
  day01.md                       # Day01 任务记录
  concepts/
    00-tokenizer.md              # Tokenizer 概念
    00-tensor-conventions.md     # 张量约定
    01-embedding.md              # Embedding 概念
experiments/
  tokenizer_playground.ipynb
src/mini_transformer/
  embedding.py
tests/
  test_embedding.py
```

## 环境

```bash
source .venv/bin/activate
pip install -e .
pytest -q
```
