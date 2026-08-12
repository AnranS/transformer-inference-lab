# Transformer Inference From Scratch: Zero to One

[![CI](https://github.com/AnranS/transformer-inference-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/AnranS/transformer-inference-from-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

从 Tokenizer 到 KV Cache，边写边测，手写一个 Llama 风格的现代 Decoder-only Transformer 推理引擎。

目标不是训练大模型，而是从数学公式出发，逐步实现生成、Prefill、Decode、GQA 和 KV Cache，
最后完成 Hugging Face 数值对齐与性能分析。
**毕业标准**：与 Hugging Face 参考实现 logits 对齐；有无 KV Cache 生成结果一致；完成 Prefill/Decode benchmark。

从 **[docs/README.md](docs/README.md)** 开始学习；完整计划见
**[docs/roadmap.md](docs/roadmap.md)**——40 个学习单元的资料、任务、落点和门禁都在那里。
28 个学习日的详细任务书已全部写完（`docs/dayXX.md`），**这个仓库是自给自足的，不需要再查外部文档**。

## 适合谁

- 已会 Python，希望真正理解 Transformer 推理数据流的人
- 想从调用 `transformers.generate()` 进阶到理解 Attention、RoPE、GQA 和 KV Cache 的工程师
- 希望通过 shape、数值对齐和 benchmark 建立 ML Systems 基础的人

项目保持教学规模：先用 CPU 验证正确性，后续性能与显存实验再使用 CUDA GPU。

## 当前进度

**9 / 40 学习单元完成。** 下一步：[docs/day10.md](docs/day10.md)（Attention 模块封装与首周门禁）。
当前测试基线：**70 个 pytest case**，覆盖 Embedding、LM Head、TinyLM、生成循环、Mask 和 Attention。

```text
hidden_states [B,S,D]
  → Q/K/V 投影 → split heads [B,H,S,Dh]
  → scaled dot-product attention + mask
  → merge heads [B,S,D] → Wo
  └──────────────── Day 6～9 已完成 ────────────────┘
```

当前已经具备多头 Attention 与 causal/padding mask，但还没有完整 Decoder Block 和位置编码；
RoPE 将在 Day 14 接入。

## 文档怎么找

| 类型 | 位置 | 作用 |
|------|------|------|
| **学习入口** | [`docs/README.md`](docs/README.md) | 当前进度、每天固定学习顺序和文档分工 |
| **总纲** | [`docs/roadmap.md`](docs/roadmap.md) | 40 单元的计划、必读资料、进度总览。**迷路了回这里** |
| **每日任务** | `docs/dayXX-YY.md` | 当天做什么、落点、过关标准。**每天从这里开始** |
| **细粒度笔记** | [`docs/concepts/README.md`](docs/concepts/README.md) | 按实际学习顺序组织概念笔记 |
| **专题交付物** | [`docs/deliverables/README.md`](docs/deliverables/README.md) | 每周门禁产出和计划中的阶段总结 |

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
  README.md                     # 学习入口与固定使用顺序
  roadmap.md                     # 总纲：40 单元计划与资料
  00-tensor-conventions.md       # 张量约定（全项目共用）
  day01-02.md … day40.md         # 28 份学习日任务书（已全部写完）
  concepts/
    README.md                    # 按学习顺序排列的概念索引
    00-tokenizer.md … 06-masks.md # Day 1～8 已完成
  deliverables/
    README.md                    # 阶段交付物索引
    01-autoregressive-language-model.md # Day 5 已完成
notebooks/
  day01_tokenizer_playground.ipynb  # 已写，Day 1 的动手实验
  day03_logits_and_softmax.ipynb    # 已写，含只算最后一位的耗时对比
  day04_autoregressive_waste.ipynb  # 已写，无 Cache 重算实验
  day08_attention_masks.ipynb       # 已写，Mask 可视化
  day09_multi_head_attention.ipynb  # 已写，多头实现与数值验证
  day10_*.ipynb … day26_*.ipynb     # Planned，按学习单元逐步完成
src/mini_transformer/
  embedding.py                   # Day 2，TokenEmbedding
  lm_head.py                     # Day 3，LMHead
  tiny_lm.py                     # Day 3，TinyLM 与 weight tying
  generate.py                    # Day 4，无 Cache 贪心生成
  attention.py                   # Day 7～9，Attention、Mask 与 MHA
tests/
  test_embedding.py              # Day 2，13 例
  test_lm_head.py                # Day 3，20 例
  test_tiny_lm.py                # Day 3，6 例
  test_generation.py             # Day 4，4 例
  test_mask.py                   # Day 8，8 例
  test_attention.py              # Day 7～9，19 例
```

后续会出现的 `rope.py`、`cache.py`、`benchmarks/` 等文件，
以及它们各自属于哪个学习单元，见 [roadmap 的目录结构一节](docs/roadmap.md#四目录结构)。

Notebook 一律**清空输出后再提交**（输出里有绝对路径，而且 diff 噪声大），
并按 nbformat 自己的写法存盘（`indent=1` + 键名排序）。
不按这个格式存，编辑器下次保存时会重排整个文件，几百行的假 diff 会盖掉真改动。

## 环境

```bash
git clone https://github.com/AnranS/transformer-inference-from-scratch.git
cd transformer-inference-from-scratch
uv sync                # 创建 .venv 并装好全部依赖（dev 组的 pytest 默认包含）
source .venv/bin/activate
pytest -q
```

`pytest` 放在 `[dependency-groups]` 的 `dev` 组里，`uv sync` 默认会装，不需要额外加参数。
发行包名是 `transformer-inference-from-scratch`，Python import 保持简洁：

```python
from mini_transformer import MultiHeadAttention, TokenEmbedding
```

不用 uv 的话：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest
```

当前验证组合：Python 3.12 + torch 2.13 + transformers 5.14 + pytest 9.1。前 4 周 CPU 即可。

## 跑 notebook 前先缓存模型

`notebooks/day01_tokenizer_playground.ipynb` 里设了 `HF_HUB_OFFLINE=1` 和
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

## License

[MIT](LICENSE) © 2026 AnranS
