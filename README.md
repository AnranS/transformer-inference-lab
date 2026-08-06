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
uv sync                # 创建 .venv 并装好全部依赖（dev 组的 pytest 默认包含）
source .venv/bin/activate
pytest -q
```

`pytest` 放在 `[dependency-groups]` 的 `dev` 组里，`uv sync` 默认会装，不需要额外加参数。
不用 uv 的话：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest
```

已验证组合：Python 3.12 + torch 2.13 + transformers 5.14。

## 跑 notebook 前先缓存模型

`experiments/tokenizer_playground.ipynb` 里设了 `HF_HUB_OFFLINE=1` 和
`local_files_only=True`（避免 kernel 被代理卡死），所以**模型必须先下载到本地缓存**，
否则会直接报错。第一次跑之前，在联网环境下执行：

```bash
python -c "
from transformers import AutoTokenizer
for name in ['bert-base-uncased', 'bert-base-multilingual-cased', 'Qwen/Qwen2.5-0.5B-Instruct']:
    AutoTokenizer.from_pretrained(name)
"
```

只会拉 tokenizer 文件，不下载模型权重。

notebook 的输出不进版本库，提交前请先清空（Kernel → Restart & Clear Output）。
