# Transformer Inference Lab

从零手写一个现代 Decoder-only Transformer 推理引擎，边写边测。

目标不是训练大模型，而是把它改造成支持生成、Prefill、Decode、GQA 和 KV Cache 的单卡推理器。
**毕业标准**：与 Hugging Face 参考实现 logits 对齐；有无 KV Cache 生成结果一致；完成 Prefill/Decode benchmark。

完整计划见 **[docs/roadmap.md](docs/roadmap.md)**——40 个学习单元的资料、任务、落点和门禁都在那里。
28 个学习日的详细任务书已全部写完（`docs/dayXX.md`），**这个仓库是自给自足的，不需要再查外部文档**。

## 当前进度

**2 / 40 学习单元完成。** 下一步：[docs/day03-04.md](docs/day03-04.md)（LM Head 与自回归循环）。

```text
文本 → input_ids → Embedding → hidden_states → LM Head → logits → next token ─┐
       └── Day 1 ──┘└─ Day 2 ─┘                └──── Day 3 ────┘              │
                                                拼回 input_ids，再来一轮 ←──────┘
                                                      └─ Day 4 ─┘
       ────── 已完成 ──────┘└────────────── 进行中 ──────────────┘
```

已完成的部分**还没有位置概念、也没有上下文交互**（Attention 在 Day 6～9，RoPE 在 Day 14），
所以现阶段生成的 token 序列没有实际意义。当前验证的是管道通不通，不是输出好不好。

## 文档怎么找

| 类型 | 位置 | 作用 |
|------|------|------|
| **总纲** | [`docs/roadmap.md`](docs/roadmap.md) | 40 单元的计划、必读资料、进度总览。**迷路了回这里** |
| **每日任务** | `docs/dayXX-YY.md` | 当天做什么、落点、过关标准。**每天从这里开始** |
| **细粒度笔记** | `docs/concepts/` | 边学边记，一个知识点一篇 |
| **专题交付物** | `docs/NN-*.md` | 每周门禁产出，综合当周笔记 |

文件名里的 `dayXX-YY` 是**学习单元**编号，不是自然日——一天完成 1～2 个单元。
编号约定见 [roadmap 的「编号约定」一节](docs/roadmap.md#编号约定先看这个容易搞混)。

每份日文档的结构固定：进度与落点表 → 必读/选读资料 → 按分钟切分的任务 →
过关标准 → 一道面试式问题。**打开就能直接开工，不需要先读 roadmap。**

四周的学习日文档：

| 周 | 学习日文档 | 门禁 |
|---|---|---|
| 1 | [01-02](docs/day01-02.md) [03-04](docs/day03-04.md) [05](docs/day05.md) [06](docs/day06.md) [07-08](docs/day07-08.md) [09](docs/day09.md) [10](docs/day10.md) | Attention 与 SDPA 数值对齐 |
| 2 | [11-12](docs/day11-12.md) [13](docs/day13.md) [14-15](docs/day14-15.md) [16](docs/day16.md) [17-18](docs/day17-18.md) [19](docs/day19.md) [20](docs/day20.md) | 最终 logits 与 HF 对齐 |
| 3 | [21-22](docs/day21-22.md) [23](docs/day23.md) [24-25](docs/day24-25.md) [26](docs/day26.md) [27-28](docs/day27-28.md) [29](docs/day29.md) [30](docs/day30.md) | 有无 KV Cache 结果一致 |
| 4 | [31-32](docs/day31-32.md) [33](docs/day33.md) [34-35](docs/day34-35.md) [36](docs/day36.md) [37-38](docs/day37-38.md) [39](docs/day39.md) [40](docs/day40.md) | 性能报告可复现 |

## 目录

```text
docs/
  roadmap.md                     # 总纲：40 单元计划与资料
  00-tensor-conventions.md       # 张量约定（全项目共用）
  day01-02.md … day40.md         # 28 份学习日任务书（已全部写完）
  concepts/
    00-tokenizer.md              # 已写
    01-embedding.md              # 已写
    02-lm-head.md                # 已写
    03-*.md … 23-*.md            # 随学习进度补
notebooks/
  tokenizer_playground.ipynb     # Day 1～4 的动手实验
src/mini_transformer/
  embedding.py                   # Day 2
tests/
  test_embedding.py              # Day 2
```

后续会出现的 `attention.py`、`rope.py`、`cache.py`、`benchmarks/` 等文件，
以及它们各自属于哪个学习单元，见 [roadmap 的目录结构一节](docs/roadmap.md#四目录结构)。

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

已验证组合：Python 3.13 + torch 2.13 + transformers 5.14 + pytest 9.1。前 4 周 CPU 即可。

## 跑 notebook 前先缓存模型

`notebooks/tokenizer_playground.ipynb` 里设了 `HF_HUB_OFFLINE=1` 和
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
