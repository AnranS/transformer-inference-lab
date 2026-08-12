# 学习路线总纲

本项目的唯一权威计划：40 个学习单元的资料、任务、落点和门禁都在这里。

**定位**：不训练大模型，而是从零实现一个现代 Decoder-only Transformer，
并把它改造成支持生成、Prefill、Decode、GQA 和 KV Cache 的单卡推理器。

**毕业标准**：自研模型与 Hugging Face 参考实现 logits 对齐；有无 KV Cache 的生成结果一致；
完成 Prefill / Decode benchmark。

**硬件**：前 4 周 CPU 即可；性能与显存实验建议使用支持 BF16 的 CUDA GPU。

## 学完之后应该能做到

这九条是最终检验标准。任何时候觉得迷失了方向，回来看这一节。

- [ ] 从 token ID 开始，讲清下一 token 的**完整计算路径**
- [ ] 为每个主要算子**写出输入输出 shape**
- [ ] 手写 causal self-attention、RMSNorm、SwiGLU、RoPE、GQA
- [ ] 组装一个 Llama 风格 Decoder-only Transformer
- [ ] 实现 greedy、temperature、top-k、top-p generation
- [ ] 实现动态 KV Cache 和预分配 KV Cache
- [ ] 区分 Prefill 和 Decode，并测量 **TTFT、TPOT 和显存**
- [ ] 阅读 Hugging Face `LlamaAttention` 与 `LlamaDecoderLayer` 源码
- [ ] 解释常见错误：mask、position、dtype、padding、cache offset 和 shape

其中「测量 TTFT、TPOT」的两个指标：

| 指标 | 全称 | 含义 | 主要由什么决定 |
|---|---|---|---|
| **TTFT** | Time To First Token | 从收到请求到吐出第一个 token 的时间 | Prefill（大 GEMM，计算密集） |
| **TPOT** | Time Per Output Token | 之后每个 token 的平均间隔 | Decode（小 GEMM + 读 Cache，访存密集） |

这两个指标性能特征相反，是推理优化里一切权衡的起点——第 4 周的 benchmark 全部围绕它们。

## 编号约定（先看这个，容易搞混）

路线里有**两套编号**，不要混用：

| 名称 | 范围 | 含义 |
|------|------|------|
| **学习单元** | Day 1 ～ Day 40 | 知识单元，**不等于自然日** |
| **学习日** | 学习日 1 ～ 28 | 实际坐下来学的一天，完成 1～2 个学习单元 |

教材式的「内容第 1～8 周」章节划分被压缩进 4 个日历周执行，
本文只用「学习单元」和「学习日」两套编号。

每日任务文档按**学习日**命名，文件名标出它覆盖的学习单元：

```text
docs/day01-02.md    学习日 1 = 学习单元 Day 1 + Day 2
docs/day03-04.md    学习日 2 = 学习单元 Day 3 + Day 4
docs/day05.md       学习日 3 = 学习单元 Day 5
```

## 进度总览

| 状态 | 说明 |
|------|------|
| ✅ | 已完成 |
| 🔄 | 进行中 |
| ⬜ | 未开始 |

| 学习单元 | 主题 | 状态 | 每日任务文档 |
|---|---|---|---|
| Day 1 | Tokenizer | ✅ | [day01-02.md](./day01-02.md) |
| Day 2 | Embedding | ✅ | [day01-02.md](./day01-02.md) |
| Day 3 | LM Head 与 logits | ✅ | [day03-04.md](./day03-04.md) |
| Day 4 | 自回归循环 | ✅ | [day03-04.md](./day03-04.md) |
| Day 5 | 广播、数值测试、第一篇专题文档 | ✅ | [day05.md](./day05.md) |
| Day 6 | Q、K、V | ✅ | [day06.md](./day06.md) |
| Day 7 | Attention score 与 Softmax | ✅ | [day07-08.md](./day07-08.md) |
| Day 8 | Causal Mask 与 Padding Mask | ✅ | [day07-08.md](./day07-08.md) |
| Day 9 | Multi-Head Attention | ✅ | [day09.md](./day09.md) |
| Day 10 | HF LlamaAttention 源码、数值对齐 | ⬜ | [day10.md](./day10.md) |
| Day 11 | RMSNorm | ⬜ | [day11-12.md](./day11-12.md) |
| Day 12 | SwiGLU MLP | ⬜ | [day11-12.md](./day11-12.md) |
| Day 13 | RoPE 论文与公式 | ⬜ | [day13.md](./day13.md) |
| Day 14 | RoPE 实现 | ⬜ | [day14-15.md](./day14-15.md) |
| Day 15 | Decoder Block | ⬜ | [day14-15.md](./day14-15.md) |
| Day 16 | ModelConfig | ⬜ | [day16.md](./day16.md) |
| Day 17 | 完整模型 | ⬜ | [day17-18.md](./day17-18.md) |
| Day 18 | 数值容差设计 | ⬜ | [day17-18.md](./day17-18.md) |
| Day 19 | 逐模块复制 HF 权重 | ⬜ | [day19.md](./day19.md) |
| Day 20 | 最终 logits parity | ⬜ | [day20.md](./day20.md) |
| Day 21 | Greedy 与停止条件 | ⬜ | [day21-22.md](./day21-22.md) |
| Day 22 | Temperature、Top-k、Top-p | ⬜ | [day21-22.md](./day21-22.md) |
| Day 23 | Batch 与 Padding | ⬜ | [day23.md](./day23.md) |
| Day 24 | 对照 HF generate | ⬜ | [day24-25.md](./day24-25.md) |
| Day 25 | Sampler 边界与文档 | ⬜ | [day24-25.md](./day24-25.md) |
| Day 26 | 无 Cache baseline、Prefill/Decode 分解 | ⬜ | [day26.md](./day26.md) |
| Day 27 | 动态 KV Cache | ⬜ | [day27-28.md](./day27-28.md) |
| Day 28 | Prefill / Decode API | ⬜ | [day27-28.md](./day27-28.md) |
| Day 29 | 预分配 KV Cache | ⬜ | [day29.md](./day29.md) |
| Day 30 | 有无 Cache 等价性测试 | ⬜ | [day30.md](./day30.md) |
| Day 31 | GQA / MQA | ⬜ | [day31-32.md](./day31-32.md) |
| Day 32 | 手算 KV Cache 显存 | ⬜ | [day31-32.md](./day31-32.md) |
| Day 33 | Prefill Benchmark | ⬜ | [day33.md](./day33.md) |
| Day 34 | Decode Benchmark、CUDA 异步 | ⬜ | [day34-35.md](./day34-35.md) |
| Day 35 | Profiler | ⬜ | [day34-35.md](./day34-35.md) |
| Day 36 | 显存快照、Cache 增长分析 | ⬜ | [day36.md](./day36.md) |
| Day 37 | inference_mode | ⬜ | [day37-38.md](./day37-38.md) |
| Day 38 | SDPA 与 torch.compile 对照 | ⬜ | [day37-38.md](./day37-38.md) |
| Day 39 | 完整实验矩阵与复现检查 | ⬜ | [day39.md](./day39.md) |
| Day 40 | 毕业验收、性能报告、README | ⬜ | [day40.md](./day40.md) |

---

# 一、28 个学习日的执行映射

## 第 1 周（学习单元 Day 1～10）

| 学习日 | 单元 | 主要任务 |
|---|---|---|
| 1 | [Day 1～2](./day01-02.md) | Tokenizer、Embedding、shape 测试 |
| 2 | [Day 3～4](./day03-04.md) | LM Head、自回归 baseline |
| 3 | [Day 5](./day05.md) | 广播、数值测试、第一篇文档 |
| 4 | [Day 6](./day06.md) | Attention 论文、QKV 手算 |
| 5 | [Day 7～8](./day07-08.md) | Naive Attention、SDPA、Mask |
| 6 | [Day 9](./day09.md) | Multi-Head Attention 完整实现 |
| 7 | [Day 10](./day10.md) | HF 源码阅读、数值对齐、测试与文档 |

**周门禁**：Naive Attention 与 PyTorch SDPA 数值对齐，causal / padding mask 测试全部通过。
**未通过不得进入 Decoder Block。**

## 第 2 周（学习单元 Day 11～20）

| 学习日 | 单元 | 主要任务 |
|---|---|---|
| 8 | [Day 11～12](./day11-12.md) | RMSNorm、SwiGLU、单测 |
| 9 | [Day 13](./day13.md) | RoPE 论文、公式和手算 |
| 10 | [Day 14～15](./day14-15.md) | RoPE 实现、Decoder Block |
| 11 | [Day 16](./day16.md) | ModelConfig、尺寸约束、参数量 |
| 12 | [Day 17～18](./day17-18.md) | 完整模型、数值容差设计 |
| 13 | [Day 19](./day19.md) | 逐模块复制 HF 权重、逐层对齐 |
| 14 | [Day 20](./day20.md) | 最终 logits parity、文档 |

**周门禁**：RMSNorm、RoPE、Attention、MLP、Block 和最终 logits 均与 HF 参考实现对齐。

## 第 3 周（学习单元 Day 21～30）

| 学习日 | 单元 | 主要任务 |
|---|---|---|
| 15 | [Day 21～22](./day21-22.md) | Greedy、EOS、Temperature、Top-k、Top-p |
| 16 | [Day 23](./day23.md) | Batch、Padding、Mask、Position |
| 17 | [Day 24～25](./day24-25.md) | HF generate 对照、Sampler 文档 |
| 18 | [Day 26](./day26.md) | 无 Cache baseline、Prefill/Decode 分解 |
| 19 | [Day 27～28](./day27-28.md) | 动态 KV Cache、Cache position |
| 20 | [Day 29](./day29.md) | 预分配 KV Cache、原地写入 |
| 21 | [Day 30](./day30.md) | 有/无 Cache 逐步 logits 与 token 等价性测试 |

**周门禁**：Greedy 生成路径结果一致；Prefill 后每轮 Decode 只输入新 token；
动态与预分配 Cache 测试通过。

## 第 4 周（学习单元 Day 31～40）

| 学习日 | 单元 | 主要任务 |
|---|---|---|
| 22 | [Day 31～32](./day31-32.md) | GQA/MQA、真实模型 KV Cache 手算 |
| 23 | [Day 33](./day33.md) | 可靠 Benchmark 工具与方法 |
| 24 | [Day 34～35](./day34-35.md) | CUDA 异步、Prefill/Decode Profiler |
| 25 | [Day 36](./day36.md) | 显存快照、Cache 增长分析 |
| 26 | [Day 37～38](./day37-38.md) | inference_mode、SDPA、torch.compile 对照 |
| 27 | [Day 39](./day39.md) | 完整实验矩阵、数据整理和复现检查 |
| 28 | [Day 40](./day40.md) | 毕业项目验收、性能报告、README、下一阶段清单 |

**最终门禁**：所有正确性测试通过；README 可让别人复现；
报告包含 Prefill、Decode、GQA、Cache、SDPA、BF16 的数据和结论。

## 高强度执行规则

- [ ] 每天开始前明确当天必须通过的测试，**不以「读完资料」为完成**。
- [ ] 必读和选读资料全部保留；选读放在当天编码完成后的晚间时段。
- [ ] 每个学习日至少提交一次可运行增量。
- [ ] 工作日采用两个 90 分钟深度工作块；双单元日增加一个 45～60 分钟阅读块。
- [ ] 周末先完成正确性门禁，再做文档和重构。
- [ ] 如果某日延期，**只能挪用下一日的选读/复盘时段，不能删除测试或项目能力**。
- [ ] 不提前进入 vLLM、FlashAttention 或 Triton。

---

# 二、40 个学习单元详情

每个单元：先读**必读**，带着**阅读目标**进入编码；**选读**只在当天任务完成后再看。

## 学习单元 Day 1～5：Token、Embedding 与自回归生成

**本周问题**：文本怎样变成 token ID？Embedding 为什么是查表？
hidden state 如何变成词表 logits？为什么 LLM 必须逐 token 生成？

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 1 | [HF Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer) | [HF Tokenizers 文档](https://huggingface.co/docs/tokenizers/) | 理解 encode/decode、special tokens、attention mask、padding side |
| Day 2 | [PyTorch nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) | 确认 Embedding 是索引查表，并掌握输入输出 shape |
| Day 3 | [PyTorch nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) | [HF Model Outputs](https://huggingface.co/docs/transformers/main_classes/output) | 理解 LM Head、logits 和词表维度，不把 logits 当概率 |
| Day 4 | [HF Generation Strategies](https://huggingface.co/docs/transformers/main/generation_strategies) | [HF Generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) | 看懂自回归循环、greedy generation 和停止条件 |
| Day 5 | [PyTorch Broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) | [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/generated/torch.testing.assert_close.html) | 复盘 shape 与广播规则，并学会写数值测试 |

**任务与落点**

| 单元 | 任务 | 落点 |
|---|---|---|
| Day 1 | 编解码 10 条中英文；打印 input_ids / attention_mask / special tokens；对比 left/right padding；记录同句在不同 tokenizer 下的 token 数 | `notebooks/day01_tokenizer_playground.ipynb`、[`concepts/00-tokenizer.md`](./concepts/00-tokenizer.md) |
| Day 2 | 实现 `[B,S] → [B,S,D]`；验证相同 token ID 得到相同向量；验证梯度关闭时仍能 forward | `src/mini_transformer/embedding.py`、`tests/test_embedding.py`、[`concepts/01-embedding.md`](./concepts/01-embedding.md) |
| Day 3 | 最小 embedding → linear → logits 模型；取 `logits[:, -1, :]` 预测下一个 token；对比 tied / untied 参数量 | `src/mini_transformer/lm_head.py`、`tiny_lm.py`、`tests/test_lm_head.py`、[`concepts/02-lm-head.md`](./concepts/02-lm-head.md) |
| Day 4 | 实现无 KV Cache 的自回归 baseline；理解每轮为什么重复计算全部历史 token | `src/mini_transformer/generate.py`、`tests/test_generation.py` |
| Day 5 | 文本到 token 流程图；token ID / embedding / hidden state / logits / probability 的区别；自回归伪代码；为什么第 t 个 token 依赖之前所有 token | **`docs/deliverables/01-autoregressive-language-model.md`** |

**本段验收**

- [ ] 能从文本一路讲到 next token
- [ ] 能写出 embedding 和 LM Head 的 shape
- [ ] 能解释 logits 为什么不需要先 Softmax 再 argmax
- [ ] 完成 `tests/test_embedding.py`

## 学习单元 Day 6～10：Scaled Dot-Product Attention

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 6 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [NeurIPS 论文 PDF](https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | 只读 3.1、3.2.1、3.4、3.5，写出 Q/K/V 和 Attention 公式 |
| Day 7 | [PyTorch SDPA API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) | [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) | 逐项理解 scale、mask、dropout、is_causal 和 GQA 参数 |
| Day 8 | [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) | [PyTorch MaskedTensor Softmax](https://docs.pytorch.org/maskedtensor/main/notebooks/safe_softmax.html) | 区分 causal mask、padding mask、Boolean mask 与 additive mask |
| Day 9 | [PyTorch Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html) | [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) | 理解 head reshape、transpose、合并与 SDPA 调用方式 |
| Day 10 | [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [HF Llama 文档](https://huggingface.co/docs/transformers/model_doc/llama) | 只找 `LlamaAttention`，标注 Q/K/V 投影、RoPE、Attention 和输出投影 |

**任务与落点**

| 单元 | 任务 | 落点 |
|---|---|---|
| Day 6 | 完成一个 4 token、4 维向量的手算例子。理解 Q 表示当前 token 想找什么、K 表示每个 token 可被怎样匹配、V 是匹配后聚合的信息，且三者都来自当前 hidden states 的线性投影 | 手写笔记 |
| Day 7 | 实现 `naive_attention`：`Q @ Kᵀ` → 除 `sqrt(Dh)` → 加 mask → 末维 Softmax → `probability @ V`。测试每行概率和约等于 1、与 SDPA 对齐、检查 FP32/BF16 容差 | `src/mini_transformer/attention.py` |
| Day 8 | 画出 S=4 时的 mask。区分 causal / padding、Boolean / additive；注意 Decode 阶段 `Sq ≠ Skv` | `tests/test_mask.py` |
| Day 9 | `[B,S,D]` 投影成 Q/K/V → reshape+transpose 到 `[B,H,S,Dh]` → 每 head 独立 Attention → 合并回 `[B,S,D]` → 输出投影 `Wo` | `src/mini_transformer/attention.py` |
| Day 10 | 测试与文档 | `tests/test_attention.py`、`notebooks/day10_attention_shapes.ipynb`、**`docs/deliverables/02-attention-and-masks.md`** |

**Day 9 常见 bug 清单**（写之前先看一遍）

- Softmax 维度写错
- transpose 后直接 `view`
- mask 广播维度错误
- 缩放使用 `D` 而不是 `Dh`
- 合并 head 后顺序错误

**本段验收**

- [ ] 不看资料写出 Attention 公式
- [ ] 从 `[B,S,D]` 推导到 `[B,H,S,Dh]` 再还原
- [ ] 解释为什么 attention score 是 S×S
- [ ] naive 实现与 SDPA 数值对齐
- [ ] 能分别处理 causal mask 和 padding mask

## 学习单元 Day 11～15：现代 Decoder Block

**目标结构**（Llama 风格 Pre-Norm）：

```text
x → RMSNorm → Attention → Residual Add → RMSNorm → SwiGLU MLP → Residual Add
```

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 11 | [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) | [RMSNorm 论文](https://arxiv.org/abs/1910.07467) | 理解归一化维度、epsilon，以及它和 LayerNorm 的差异 |
| Day 12 | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | [PyTorch SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) | 写出 SwiGLU 的 gate/up/down 数据流和三个权重 shape |
| Day 13 | [RoFormer / RoPE 论文](https://arxiv.org/abs/2104.09864) | [RoFormer PDF](https://arxiv.org/pdf/2104.09864) | 理解二维成对旋转、position 与相对位置性质 |
| Day 14 | [HF Llama RoPE 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [HF Llama RoPE 参数](https://huggingface.co/docs/transformers/model_doc/llama) | 定位 `LlamaRotaryEmbedding` 和 `apply_rotary_pos_emb`，核对布局 |
| Day 15 | [HF Llama DecoderLayer 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [原始 Transformer 论文](https://arxiv.org/abs/1706.03762) | 画出 Pre-Norm、Residual、Attention、MLP 的完整执行顺序 |

**任务与落点**

| 单元 | 任务 | 落点 |
|---|---|---|
| Day 11 | 实现并与 `torch.nn.RMSNorm` 对齐。注意中间统计通常转 FP32 | `src/mini_transformer/norm.py` |
| Day 12 | `gate = SiLU(Wgate x)`；`up = Wup x`；`output = Wdown(gate × up)`。记录三次矩阵乘的 shape、参数量和理论 FLOPs | `src/mini_transformer/mlp.py` |
| Day 13 | 手推 RoPE 公式与相对位置性质 | 手写笔记 |
| Day 14 | 生成 cos/sin cache；对 Q/K 应用旋转；支持 position offset；与 HF Llama RoPE 对齐 | `src/mini_transformer/rope.py`、`tests/test_rope.py`、`notebooks/day14_rope_visualization.ipynb` |
| Day 15 | Attention Pre-Norm → 第一次 Residual → MLP Pre-Norm → 第二次 Residual；dropout 在推理模式关闭 | `src/mini_transformer/block.py`、`tests/test_block.py`、**`docs/deliverables/03-modern-decoder-block.md`** |

**RoPE 要点**（Day 13～14 必须搞清）

- RoPE 作用在 **Q/K，不作用在 V**
- 偶数 / 奇数维成对旋转
- `position_ids` 和频率的关系
- 为什么 Decode 必须传**正确的 cache position**
- `rotate_half` 两种布局**不能混用**

**本段文档要求**：`docs/deliverables/03-modern-decoder-block.md` 必须包含 Decoder Block 数据流、
Pre-Norm 与 Post-Norm 区别、RMSNorm 公式、SwiGLU 公式及参数 shape、
RoPE 的输入输出和 position 规则、一个 Block 的参数量估算。

## 学习单元 Day 16～20：组装完整模型并与 HF 对齐

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 16 | [HF LlamaConfig](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig) | [HF 配置基类](https://huggingface.co/docs/transformers/main_classes/configuration) | 明确所有尺寸参数及它们之间的约束 |
| Day 17 | [HF LlamaModel 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [HF Model 基类](https://huggingface.co/docs/transformers/main_classes/model) | 定位 Embedding、层循环、Final Norm、LM Head 和 forward 返回值 |
| Day 18 | [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/generated/torch.testing.assert_close.html) | [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) | 确定 FP32/BF16 对齐所需的 rtol、atol 和测试粒度 |
| Day 19 | [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [PyTorch Module API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) | 完成逐模块权重复制与中间输出对照 |
| Day 20 | [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | [nanoGPT](https://github.com/karpathy/nanoGPT) | 固定随机性；如做可选训练，只参考最小训练闭环 |

**Day 16：`config.py` 至少包含**

`vocab_size`、`hidden_size`、`intermediate_size`、`num_hidden_layers`、
`num_attention_heads`、`num_key_value_heads`、`head_dim`、
`max_position_embeddings`、`rms_norm_eps`、`rope_theta`、`tie_word_embeddings`

> 所有模块**只从 config 读取尺寸**，禁止散落魔法数字。

**Day 17：`model.py` 数据流**

```text
token IDs → embedding → N × DecoderBlock → final RMSNorm → LM Head → logits
```

先用小配置：`vocab_size=256`、`hidden_size=128`、`intermediate_size=384`、
`layers=2`、`query heads=4`、`KV heads=4`、`max sequence=128`。

**Day 19～20：HF Parity 步骤**

1. 创建一个极小的 HF `LlamaConfig`
2. 实例化随机权重的 `LlamaForCausalLM`
3. 将相同权重复制到自研模型
4. 传入完全相同的 `input_ids` 和 `position_ids`
5. 比较每层输出和最终 logits

测试等级（逐级打通，落点 `tests/test_hf_parity.py`）：

- [ ] RMSNorm 对齐
- [ ] RoPE 对齐
- [ ] Attention 对齐
- [ ] MLP 对齐
- [ ] 单个 Decoder Block 对齐
- [ ] 最终 logits 对齐

> 这比「看起来能生成文本」更能证明实现正确。

**Day 20 可选训练的限制**：参数量小于 20M；训练不超过数小时；
不把时间花在数据清洗、分布式训练或调参。

**本段验收**

- [ ] 完整 forward 输出 `[B,S,V]`
- [ ] 参数量统计与手算基本一致
- [ ] `tests/test_hf_parity.py` 通过
- [ ] 每层误差都在设定容差内
- [ ] 能从 HF `LlamaDecoderLayer` 反向定位到自己的实现

## 学习单元 Day 21～25：Generation 与 Batch

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 21 | [HF Generation Strategies](https://huggingface.co/docs/transformers/main/generation_strategies) | [HF GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) | 实现 greedy、EOS、max_new_tokens 和 finished 状态 |
| Day 22 | [HF Sampling Strategies](https://huggingface.co/docs/transformers/main/generation_strategies#sampling) | [PyTorch multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) | 理解 temperature、top-k、top-p 与随机采样 |
| Day 23 | [HF Padding and Truncation](https://huggingface.co/docs/transformers/pad_truncation) | [HF Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer) | 理清 left/right padding、attention_mask 和 position_ids |
| Day 24 | [HF Generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) | [HF Generation Utilities](https://huggingface.co/docs/transformers/internal/generation_utils) | 将自研生成参数逐项映射到 HF generate，做结果对照 |
| Day 25 | [HF Logits Processors](https://huggingface.co/docs/transformers/internal/generation_utils#logitsprocessor) | [HF Streamers](https://huggingface.co/docs/transformers/internal/generation_utils#streamers) | 区分 logits processor、sampler、stopping criteria 和 streamer |

**任务与落点**

| 单元 | 任务 | 落点 |
|---|---|---|
| Day 21 | greedy、`max_new_tokens`、EOS、每条序列**独立** finished 状态、finished 后不再产生有效 token | `src/mini_transformer/generate.py` |
| Day 22 | Temperature 改变分布尖锐程度；Top-k 保留最高 k 个；Top-p 保留累计概率达阈值的最小集合；固定种子后结果可复现 | `src/mini_transformer/sampling.py` |
| Day 23 | left/right padding 对 generation 的影响、attention_mask、position_ids、不同长度请求同 batch 的问题；**最后有效 token 位置不一定等于数组最后一列** | `tests/test_generation.py` |
| Day 24 | 同种子同参数下对照 HF：greedy、temperature+top-k、temperature+top-p、EOS 停止、batch 中不同长度请求 | `tests/test_generation.py` |
| Day 25 | 一轮生成的流程图；logits processor 与 sampler 的边界；greedy/top-k/top-p 适用场景；padding、mask、position 的关系 | **`docs/deliverables/04-generation.md`** |

## 学习单元 Day 26～30：Prefill、Decode 与 KV Cache

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 26 | [HF Caching 原理](https://huggingface.co/docs/transformers/main/cache_explanation) | [HF Generation Utilities](https://huggingface.co/docs/transformers/internal/generation_utils) | 明确缓存的是每一层 K/V，而不是 token 或 Attention score |
| Day 27 | [HF Cache Strategies](https://huggingface.co/docs/transformers/kv_cache) | [HF cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py) | 比较 DynamicCache、StaticCache、QuantizedCache 的数据结构 |
| Day 28 | [HF Llama Attention Cache 路径](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [HF Caching 原理](https://huggingface.co/docs/transformers/main/cache_explanation) | 追踪 `past_key_values`、`cache_position` 与本轮 K/V 更新 |
| Day 29 | [HF Static Cache](https://huggingface.co/docs/transformers/kv_cache#fixed-size-cache) | [PyTorch Tensor Indexing](https://docs.pytorch.org/docs/stable/tensor_view.html) | 设计预分配缓存和按 position 原地写入，避免逐步 concat |
| Day 30 | [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) | [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/generated/torch.testing.assert_close.html) | 完成有/无 Cache 的逐步 logits 与 token 等价性测试 |

**任务与落点**

| 单元 | 任务 | 落点 |
|---|---|---|
| Day 26 | 记录每轮：输入长度、forward 时间、计算的 token 数、峰值显存。**证明**无 Cache 时历史 token 被重复计算 | `benchmarks/`、`notebooks/day26_cache_growth.ipynb` |
| Day 27 | 每层返回本轮新 K/V：`past [B,Hkv,S,Dh]` + `new [B,Hkv,1,Dh]` → `combined [B,Hkv,S+1,Dh]`。第一版允许 concat，**优先保证正确** | `src/mini_transformer/cache.py` |
| Day 28 | 设计 `prefill(input_ids, attention_mask)` 和 `decode(next_token_ids, cache, cache_position)`；每层独立 Cache；Prefill 可多 token，Decode 通常单 token | `src/mini_transformer/cache.py` |
| Day 29 | 初始化最大 batch / 最大 sequence；按 `cache_position` 原地写入；维护当前有效长度；返回有效 KV view；**禁止每轮重新分配整个缓存** | `src/mini_transformer/cache.py` |
| Day 30 | 固定权重与输入 → 路径 A 每轮输入完整序列不用 Cache，路径 B Prefill 后每轮只输入新 token → 比较每轮最后位置 logits 和最终 greedy token 序列；覆盖 batch=1、batch>1、不同 prompt 长度 | `tests/test_cache_equivalence.py`、**`docs/deliverables/05-prefill-decode-kv-cache.md`** |

**必须排查的错误**（这一段最容易出错，写之前贴在显示器上）

- RoPE position 从 0 重新开始
- causal mask 在 `Sq=1`、`Skv>1` 时错误
- K/V 在错误的 sequence 维拼接
- Cache 写入位置 off-by-one
- Padding token 被写入有效 Cache
- GQA 的 KV head 扩展方式错误

## 学习单元 Day 31～35：MHA、GQA、MQA 与性能基础

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 31 | [GQA 论文](https://arxiv.org/abs/2305.13245) | [GQA 论文 PDF](https://aclanthology.org/2023.emnlp-main.298.pdf) | 区分 MHA、GQA、MQA，并理解 KV heads 的共享关系 |
| Day 32 | [Llama 2 论文](https://arxiv.org/abs/2307.09288) | [HF LlamaConfig](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig) | 从真实配置计算 head_dim、KV heads、参数量和 KV Cache |
| Day 33 | [PyTorch Benchmark 教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html) | 学会 warmup、同步、重复测量和结果比较 |
| Day 34 | [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) | [PyTorch CUDA API](https://docs.pytorch.org/docs/stable/cuda.html) | 理解异步执行、synchronize、Stream 和显存统计 |
| Day 35 | [PyTorch Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | [PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html) | 分别定位 Prefill 和 Decode 的热点算子 |

**任务与落点**

| 单元 | 任务 | 落点 |
|---|---|---|
| Day 31 | 配置改为 `Hq` 可大于 `Hkv`：MHA（`Hq=Hkv`）、GQA（`Hq>Hkv>1`）、MQA（`Hkv=1`）。实现 KV head 到 Query head 的共享映射 | `src/mini_transformer/attention.py` |
| Day 32 | 手算 `KV bytes = 2 × layers × batch × tokens × Hkv × Dh × bytes_per_element`。分别算 2 层玩具模型、7B/8B 级 GQA 模型、batch=1 与 16、context=2K/8K/32K | `benchmarks/bench_cache_memory.py` |
| Day 33 | 控制变量：batch 1/2/4/8；prompt 128/512/1024/2048；dtype FP32/FP16/BF16；naive vs SDPA。记录 latency、tokens/s、峰值显存 | `benchmarks/bench_prefill.py` |
| Day 34 | 固定 batch 改 context 长度；固定 context 改 batch；比较 MHA/GQA/MQA。记录单 token latency 与 tokens/s | `benchmarks/bench_decode.py` |
| Day 35 | 观察 Prefill 大 GEMM、Decode 小 GEMM、Attention 与 MLP 时间占比、Cache concat 复制开销、预分配是否减少 allocation/copy、CPU launch 与 GPU kernel 时间 | `benchmarks/bench_attention.py`、`notebooks/day26_cache_growth.ipynb` |

## 学习单元 Day 36～40：毕业项目与报告

| 单元 | 必读 | 选读 | 阅读目标 |
|---|---|---|---|
| Day 36 | [PyTorch CUDA Memory](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | [torch.cuda.memory_snapshot](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html) | 区分 allocated、reserved、峰值显存，并生成内存快照 |
| Day 37 | [torch.inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html) | [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | 确认推理路径关闭梯度及版本计数开销 |
| Day 38 | [torch.compile 教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | [PyTorch Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html) | 比较 eager、SDPA、compile；记录编译冷启动与稳态性能 |
| Day 39 | [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | 整理 benchmark 方法，保证结果可复现且控制变量明确 |
| Day 40 | [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | 完成代码审查、性能报告、README 和下一阶段瓶颈清单 |

**毕业项目：Mini Transformer Inference Engine**

```text
Text → Tokenizer → Embedding → Decoder Blocks → LM Head → Sampling → KV Cache → Streaming Tokens
```

必须支持：

- [ ] Llama 风格 Decoder-only Block
- [ ] RMSNorm、RoPE、SwiGLU
- [ ] MHA 与 GQA
- [ ] Causal mask 和 padding mask
- [ ] Greedy、temperature、top-k、top-p
- [ ] Prefill 与单 token Decode
- [ ] 动态 Cache 与预分配 Cache
- [ ] Batch generation
- [ ] Hugging Face logits parity
- [ ] 有无 Cache 结果一致
- [ ] CPU 与 CUDA GPU
- [ ] BF16 inference
- [ ] Prefill / Decode benchmark

**性能报告**（`docs/deliverables/07-inference-performance.md`）至少回答：

1. Prefill 延迟如何随 prompt 长度变化？
2. Decode 延迟如何随 context 长度变化？
3. Batch 增大如何影响单请求延迟和总吞吐？
4. GQA 将 KV Cache 减少了多少？
5. 动态 concat Cache 与预分配 Cache 的差距是多少？
6. naive Attention 与 PyTorch SDPA 的差距是多少？
7. 峰值显存由权重、激活和 Cache 哪些部分组成？
8. 当前实现最慢的三个算子是什么？
9. 下一步为什么应该学习 FlashAttention、Triton 或 Paged KV Cache？

---

# 三、统一符号与张量约定

| 符号 | 含义 | 示例 |
|---|---|---|
| `B` | Batch size | 2 |
| `S` | Sequence length | 128 |
| `D` | Hidden size | 512 |
| `Hq` | Query head 数 | 8 |
| `Hkv` | KV head 数 | 2 |
| `Dh` | Head dimension | 64 |
| `V` | Vocabulary size | 32000 |

始终明确：

```text
token_ids          [B, S]
hidden_states      [B, S, D]
Q                  [B, Hq,  S,  Dh]
K / V              [B, Hkv, S,  Dh]
attention scores   [B, Hq,  Sq, Skv]
logits             [B, S, V]
```

约定 `D = Hq × Dh`；GQA 情况下 `Hkv < Hq`。

当前已落地的部分见 [`00-tensor-conventions.md`](./00-tensor-conventions.md)，随学习推进扩充。

---

# 四、目录结构

目标结构（`*` 表示已存在）：

```text
transformer-inference-from-scratch/
├── README.md                                  *
├── pyproject.toml                             *
├── src/mini_transformer/
│   ├── config.py            Day 16
│   ├── embedding.py         Day 2               *
│   ├── lm_head.py           Day 3               *
│   ├── tiny_lm.py           Day 3（过渡产物，Day 17 后由 model.py 取代）*
│   ├── norm.py              Day 11
│   ├── rope.py              Day 14
│   ├── attention.py         Day 7/9/31           *
│   ├── mlp.py               Day 12
│   ├── block.py             Day 15
│   ├── model.py             Day 17
│   ├── cache.py             Day 27～29
│   ├── sampling.py          Day 22
│   └── generate.py          Day 4/21             *
├── tests/
│   ├── test_embedding.py            Day 2       *
│   ├── test_lm_head.py              Day 3       *
│   ├── test_tiny_lm.py              Day 3       *
│   ├── test_generation.py           Day 4/23～24*
│   ├── test_attention.py            Day 10      *
│   ├── test_mask.py                 Day 8       *
│   ├── test_rope.py                 Day 14
│   ├── test_block.py                Day 15
│   ├── test_hf_parity.py            Day 19～20
│   └── test_cache_equivalence.py    Day 30
├── benchmarks/
│   ├── bench_attention.py           Day 35
│   ├── bench_prefill.py             Day 33
│   ├── bench_decode.py              Day 34
│   └── bench_cache_memory.py        Day 32
├── notebooks/                       文件名前缀 = 创建它的学习单元
│   ├── day01_tokenizer_playground.ipynb   Day 1      *
│   ├── day03_logits_and_softmax.ipynb     Day 3      *
│   ├── day04_autoregressive_waste.ipynb   Day 4      *
│   ├── day08_attention_masks.ipynb        Day 8      *
│   ├── day09_multi_head_attention.ipynb   Day 9      *
│   ├── day10_attention_shapes.ipynb       Day 10     *
│   ├── day14_rope_visualization.ipynb     Day 14     *
│   ├── day20_tiny_training.ipynb          Day 20（选做）*
│   └── day26_cache_growth.ipynb           Day 26/29/35*
└── docs/
    ├── README.md                               *  学习导航
    ├── roadmap.md                              *  本文
    ├── dayXX-YY.md                             *  每日任务
    ├── concepts/                               *  细粒度工作笔记
    ├── 00-tensor-conventions.md    Day 1～2     *
    └── deliverables/
        ├── 01-autoregressive-language-model.md  Day 5
        ├── 02-attention-and-masks.md            Day 10
        ├── 03-modern-decoder-block.md           Day 15
        ├── 04-generation.md                     Day 25
        ├── 05-prefill-decode-kv-cache.md        Day 30
        ├── 06-gqa-and-kv-cache-memory.md        Day 40
        ├── 07-inference-performance.md          Day 40
        └── 08-next-steps.md                     Day 40
```

**三类文档的分工**

| 类型 | 位置 | 作用 |
|---|---|---|
| 总纲 | `docs/roadmap.md` | 本文。全局计划、资料、进度 |
| 每日任务 | `docs/dayXX-YY.md` | 当天做什么、过关标准 |
| 细粒度笔记 | `docs/concepts/` | 边学边记，一个知识点一篇 |
| 专题交付物 | `docs/deliverables/` | 每周门禁产出，综合当周 concepts |

> **不要同时创建多个玩具仓库。** 所有阶段都在同一个项目上递增，
> 方便看到「数学公式 → 正确实现 → 推理优化」的演进过程。

---

# 五、过关问题清单

随时抽查自己，答不上来就回去补对应单元。

## Token 与生成（Day 1～5）

- [ ] Token ID、Embedding、Hidden State、Logits、Probability 有什么区别？
- [ ] 为什么生成只取最后有效位置 logits？
- [ ] 为什么 Softmax 前输入叫 logits？

## Attention（Day 6～10）

- [ ] 为什么要除以 `sqrt(Dh)`？
- [ ] Softmax 为什么沿 key 维？
- [ ] causal mask 与 padding mask 有什么区别？
- [ ] 为什么 Attention 中间矩阵随 S² 增长？
- [ ] 多头的价值是什么？

## Decoder Block（Day 11～15）

- [ ] Pre-Norm 的数据流是什么？
- [ ] RMSNorm 与 LayerNorm 少了什么？
- [ ] SwiGLU 为什么有 gate/up/down 三个投影？
- [ ] RoPE 为什么应用到 Q/K？

## KV Cache（Day 26～32）

- [ ] 为什么只缓存 K/V，不缓存 Q？
- [ ] Prefill 与 Decode 的 Q/K/V shape 分别是什么？
- [ ] Cache 为什么省计算却增加显存？
- [ ] 有 Cache 后 Decode Attention 为什么仍随 context 线性增长？
- [ ] GQA 为什么显著减少 KV Cache？

---

# 六、方法

## 每个学习日的时间模板

| 动作 | 时间 | 产出 |
|---|---|---|
| 读官方资料 | 2h | 带问题阅读笔记 |
| 公式与 shape 手推 | 1h | 一页 shape 表 |
| 编码 | 4～5h | 最小正确实现 |
| 测试与对照 | 2h | pytest 与误差报告 |
| 文档和复盘 | 1～2h | 专题文档 |

> 这是**单个模块周**的模板，每个日历周执行两轮。

## 每天如何使用资料

1. **预读 20～30 分钟**：只回答当天阅读目标
2. **手推 15 分钟**：写出公式、shape 和关键状态
3. **编码 60～120 分钟**：先做最小正确实现
4. **对照 30 分钟**：与 PyTorch / HF reference 比较
5. **记录 15 分钟**：写下结果、错误、修复和仍未理解的问题

> 如果当天编码没有完成，**不继续读选读资料**；先确保主项目产生一个可测试增量。

## 读资料的三个问题

每篇资料只带着这三个问题读：

1. 它定义了什么计算？
2. 输入输出 shape 是什么？
3. 推理时它需要保存、读取多少数据？

论文不要求从第一页读到最后一页：

- **Attention Is All You Need**：先读 3.1、3.2.1、3.4、3.5
- **RoFormer**：先读 RoPE 定义和相对位置推导
- **HF Llama 源码**：先找 `RMSNorm`、`RotaryEmbedding`、`Attention`、`MLP`、`DecoderLayer`、`Model.forward`
- **KV Cache 文档**：重点看 Prefill 后保存什么、Decode 如何更新和读取

---

# 七、当前不要学的内容

完成 Day 40 之前，**先不深入**：

FlashAttention kernel 实现、PagedAttention 和 Block Manager、vLLM Scheduler、
CUDA Graph、TensorRT-LLM、INT4/FP8 kernel、Tensor/Pipeline/Expert Parallel、
MoE、Speculative Decoding。

> 这些不是不重要，而是**都依赖本文中的 Attention、shape、Prefill/Decode 和 KV Cache 心智模型**。

---

# 八、第一优先级资料索引

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
- [Hugging Face Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Hugging Face KV Cache](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/main/generation_strategies)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

查 API 时用：
[SDPA API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)、
[HF Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)、
[HF Generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)
