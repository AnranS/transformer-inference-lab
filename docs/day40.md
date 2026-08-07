# 学习单元 Day 40：毕业验收、性能报告与下一阶段清单

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 40**｜学习日 28

最后一天。

> **最终门禁**：所有正确性测试通过；README 可让别人复现；
> 报告包含 Prefill、Decode、GQA、Cache、SDPA、BF16 的数据和结论。

**毕业标准**：自研模型与 HF 参考实现 logits 对齐；
有无 KV Cache 生成结果一致；完成 Prefill / Decode benchmark。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：全量测试与代码审查 | 全仓库 |
| ⬜ | 任务 2：写第六篇专题文档 | **`docs/06-gqa-and-kv-cache-memory.md`** |
| ⬜ | 任务 3：写第七篇专题文档（性能报告） | **`docs/07-inference-performance.md`** |
| ⬜ | 任务 4：重写 README 为可复现入口 | `README.md` |
| ⬜ | 任务 5：毕业验收自查 | 本文末 |
| ⬜ | 任务 6：下一阶段瓶颈清单 | `docs/08-next-steps.md` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | 完成代码审查、性能报告、README 和下一阶段瓶颈清单 |

---

## 任务 1：全量测试与代码审查（80 分钟）

### 跑全量测试

```bash
pytest -q
```

**必须全绿。** 如果有 skip，检查每一条 skip 的理由是否合理
（CPU 环境跳过 CUDA 测试是合理的；「暂时跑不过」不是）。

### 代码审查清单

自己过一遍，逐项确认：

**架构层面**

- [ ] 每个模块只从 `ModelConfig` 读尺寸，**无散落的魔法数字**（Day 16 的铁律）
- [ ] 没有并行的重复实现（`tiny_lm.py` 已退役）
- [ ] `__init__.py` 的导出和实际模块一致
- [ ] `benchmarks/` 里没有和 `src/` 重复的逻辑

**接口层面**

- [ ] `position_ids` 在 `rope.py` 层强制外部传入（不会静默用 `arange`）
- [ ] `prefill` / `decode` 各有严格的形状断言
- [ ] 所有错误信息**带实际数值**（Day 01～04 建立的习惯）
- [ ] Cache 的 `get_kv` 返回有效 view，不是整个 buffer

**HF 对应表的缺口全部关闭**

打开 Day 10 建的 `concepts/07-hf-llama-attention.md`，
确认「缺口」一列已经空了：`apply_rotary_pos_emb`（Day 14）、
`repeat_kv`（Day 31）、`past_key_value`（Day 27）都实现了。

**六个必查错误的防线都在**

Day 30 那张表，逐条确认「我的防线」一列指向的代码还在、测试还在跑。

### 补一条集成测试

如果还没有，加一条**端到端冒烟测试**：

```python
def test_end_to_end_smoke():
    """tokenizer → 模型 → 生成 → 解码，全链路跑通。"""
```

用真实 tokenizer（Day 1 缓存的那个）+ 你的模型 + `generate`，
生成几个 token 再解码回文本。结果会是乱码（模型没训练过），
但**链路必须通**。这是唯一覆盖 tokenizer 集成的测试。

## 任务 2：写第六篇专题文档（60 分钟）

**`docs/06-gqa-and-kv-cache-memory.md`**。综合 Day 31～32 和 Day 36：

1. **MHA / GQA / MQA 的关系**和 `num_key_value_groups`
2. **GQA 省显存不省参数**，含量化对比
3. **KV Cache 公式**，逐项解释每个因子的来源
4. **显存矩阵**：玩具模型 / 8B 模型 × batch 1、16 × context 2K/8K/32K
5. **每 token KV 成本**，以及「给定显存能缓存多少 token」的估算方法
6. **KV Cache 超过权重的转折点**（理论 + 实测）
7. **实现 GQA 的两个坑**：扩展顺序、`repeat_kv` 必须在写 Cache 之后
8. **预分配的代价与 PagedAttention 方向**

第 5 和第 7 项是最有实用价值的两节。

## 任务 3：写第七篇专题文档（120 分钟）

**`docs/07-inference-performance.md`**——最终性能报告，本项目的门面。

### 结构建议

```text
1. 摘要         三到五条最重要的结论，每条带数字
2. 实验设置     硬件、软件版本、模型配置、测量方法
3. 正确性验证   HF parity 各级误差表 + Cache 等价性
4. Prefill      数据 + 图 + 结论
5. Decode       数据 + 图 + 结论
6. Prefill vs Decode  两者的性能特征对比（本报告的核心洞察）
7. GQA          Cache 大小与 decode 延迟
8. KV Cache     dynamic vs static
9. SDPA 后端    显存阶数对比
10. 精度        fp32 / fp16 / bf16 三方面对照
11. 优化收益汇总  含代价/风险列
12. 已知限制与异常数据
```

### 三条写作要求

**1. 摘要先写。** 三到五条结论，每条一句话 + 一个数字。
读者只看摘要就该拿到主要发现。

**2. 每张图都要有一句结论文字。** 图本身不说明问题，
要写「这张图说明 X」。

**3. 「已知限制」不能省。** 至少包含：

- 在什么硬件上测的（CPU 上的数据要标注，且说明哪些结论待 GPU 验证）
- 模型规模只到 GPT-2 small 量级，大模型行为可能不同
- 没测多卡、没测长上下文（>8K）
- 没验证梯度路径（Day 20 任务 6 如果跳过了）
- Day 39 记录的异常数据

**如实写出限制会让报告更可信，不是更弱。**

### 第 6 节是报告的核心

「Prefill vs Decode」这一节要把整个第 4 周的发现串起来：

| | Prefill | Decode |
|---|---|---|
| 瓶颈 | 计算密集 | 访存密集 |
| batch 增大 | 时间线性增加 | 时间几乎不变 |
| 热点算子 | Attention（`S²` 项） | MLP（参数量大） |
| 主要优化 | FlashAttention、分块 | 大 batch、`torch.compile`、GQA |
| 对应指标 | TTFT | TPOT |

**这张表是四周学习的最终结晶。** 能独立填出它，就说明这个项目达到了目的。

## 任务 4：重写 README 为可复现入口（60 分钟）

最终门禁的一条是「README 可让别人复现」。README 要包含：

- [ ] **项目是什么**、不是什么（不训练大模型，是推理引擎）
- [ ] **毕业标准**和当前达成情况
- [ ] **环境安装**：`uv sync`，Python / PyTorch 版本要求
- [ ] **模型缓存**：`HF_HUB_OFFLINE` 的说明和预下载步骤
- [ ] **跑测试**：`pytest -q`，预期结果
- [ ] **跑 benchmark**：完整命令 + `--quick` 版本 + 预计耗时 + 显存需求
- [ ] **CPU / GPU 差异**：哪些实验在 CPU 上会跳过
- [ ] **目录结构**：每个目录放什么
- [ ] **文档导航**：`roadmap.md` + 七篇专题文档 + `concepts/` 的关系
- [ ] **主要结论**：从性能报告里摘三条

### 真的验证一遍

**在一个干净的环境里从头走一遍**（新建一个目录，重新 clone，重新 `uv sync`）。
任何一步卡住，就是 README 的 bug。

如果做不到完全干净的环境，至少：
删掉 `.venv` 重新 `uv sync`，然后 `pytest -q` + `run_all.py --quick`。

## 任务 5：毕业验收自查（40 分钟）

### 毕业标准（三条硬指标）

- [ ] 自研模型与 HF 参考实现 **logits 对齐**
- [ ] **有无 KV Cache 生成结果一致**
- [ ] 完成 **Prefill / Decode benchmark**

### 学完之后应该能做到（roadmap 那九条）

闭卷自答，一条都不要跳：

- [ ] 从 token ID 开始，讲清下一 token 的完整计算路径
- [ ] 为每个主要算子写出输入输出 shape
- [ ] 手写 causal self-attention、RMSNorm、SwiGLU、RoPE、GQA
- [ ] 组装一个 Llama 风格 Decoder-only Transformer
- [ ] 实现 greedy、temperature、top-k、top-p generation
- [ ] 实现动态 KV Cache 和预分配 KV Cache
- [ ] 区分 Prefill 和 Decode，并测量 TTFT、TPOT 和显存
- [ ] 阅读 HF `LlamaAttention` 与 `LlamaDecoderLayer` 源码
- [ ] 解释常见错误：mask、position、dtype、padding、cache offset 和 shape

**第 3 条是真的动手写**：找张白纸或空文件，不看已有代码，
把这五个算子写一遍。写完和自己的实现对比。

这一步不能省——**能读懂和能写出来是两种不同的能力**，
面试和实际工作要的是后者。

### 过关问题清单

roadmap 里那份清单，逐条口述作答。
每天的日文档末尾都有「今日最重要的面试式问题」，
它们加起来就是这份清单的展开版——答不上来的回去看对应那天。

## 任务 6：下一阶段瓶颈清单（60 分钟）

**`docs/08-next-steps.md`**。这是本项目的最后一份产出，也是承接下一阶段的桥梁。

### 从数据出发，不是从流行词出发

不要写「接下来学 vLLM 和 FlashAttention」。要写：

> **根据 Day 34～35 的 profile，Decode 阶段 CPU 时间占 X%，
> 说明存在 kernel launch 瓶颈。可能的解法是 CUDA Graph 或算子融合。**

每条都要**指向你自己测出的数据**。

### 建议的清单结构

| 瓶颈 | 证据（来自哪个实验） | 可能的解法 | 优先级 |
|---|---|---|---|

候选项（用你的数据判断优先级，不要照抄）：

| 瓶颈 | 典型证据 | 方向 |
|---|---|---|
| Decode 的 kernel launch 开销 | profile 里 CPU 时间占比高 | CUDA Graph、算子融合 |
| Attention 的 `S²` 显存 | Day 33 的显存阶数图 | FlashAttention（已通过 SDPA 用上，但可以自己实现学原理） |
| 预分配 Cache 的显存浪费 | Day 36 的显存曲线 | PagedAttention |
| 单请求吞吐低 | Day 34 的 batch 扫描（大 batch 收益明显） | Continuous batching |
| 权重显存 | Day 36 的显存拆解 | 量化（int8 / int4 / AWQ / GPTQ） |
| KV Cache 显存 | Day 32 的显存矩阵 | KV Cache 量化 |
| 串行解码 | 每个 token 一次完整前向 | 投机解码（speculative decoding） |

### 明确「不该马上做什么」

roadmap 有一节「当前不要学的内容」。四周结束后，
其中一些可以解锁了，但要按顺序。建议在文档里写清：

**先做**：把上表里**你自己测出证据**的那一两项做深。

**后做**：直接上 vLLM / TensorRT-LLM。用之前先自己实现一遍对应的核心机制
（比如先自己写个简化的 PagedAttention），否则只是学会调 API，
学不到你这四周想要的那种理解。

### 顺手做一件事：回顾整个项目

在文档最后写一段**回顾**：

- 哪个单元最难？为什么？
- 哪个 bug 花的时间最多？根因是什么？
- 哪个测试最有价值（抓住了最多问题）？
- 如果重来一遍，会改变什么顺序？

**这段回顾对下一个项目的价值可能超过技术清单本身。**

---

## 过关标准

- [ ] `pytest -q` 全绿，所有 skip 都有正当理由
- [ ] 代码审查清单全部确认
- [ ] HF 对应表的缺口已全部关闭
- [ ] 端到端冒烟测试存在且通过
- [ ] `docs/06-gqa-and-kv-cache-memory.md` 完成
- [ ] `docs/07-inference-performance.md` 完成，含摘要、六项数据、已知限制
- [ ] **第 6 节「Prefill vs Decode」对比表能独立填出**
- [ ] README 在删掉 `.venv` 重装后能走通全流程
- [ ] 三条毕业标准全部达成
- [ ] 九条能力目标闭卷自答通过，**含手写五个算子**
- [ ] `docs/08-next-steps.md` 完成，每条瓶颈都指向自己的实验数据
- [ ] 写了项目回顾

---

## 最后一个问题

**用五分钟，从 token ID 讲到下一个 token，包含所有 shape。**

不看任何东西，讲一遍。讲不顺的地方就是还没真正掌握的地方。

```text
input_ids [B,S] (long)
  → Embedding 查表                        → [B,S,D]
  ┌─ × L 层 ────────────────────────────────────────────┐
  │  RMSNorm                              → [B,S,D]     │
  │  Wq/Wk/Wv 投影         → Q [B,S,Hq·Dh] K/V [B,S,Hkv·Dh]
  │  view+transpose        → Q [B,Hq,S,Dh] K/V [B,Hkv,S,Dh]
  │  RoPE（只 Q/K，按绝对位置）                            │
  │  写 KV Cache（Hkv 份）  → K/V [B,Hkv,Skv,Dh]          │
  │  repeat_kv             → K/V [B,Hq,Skv,Dh]           │
  │  QKᵀ/√Dh + mask, softmax(-1), @V → [B,Hq,Sq,Dh]      │
  │  transpose+reshape, Wo                → [B,S,D]      │
  │  + 残差                                               │
  │  RMSNorm → SwiGLU(gate⊙up) → + 残差    → [B,S,D]     │
  └─────────────────────────────────────────────────────┘
  → final RMSNorm                         → [B,S,D]
  → LM Head（可与 Embedding 共享权重）       → [B,S,V]
  → 取 [:, -1, :]                          → [B,V]
  → temperature / top-k / top-p            → [B,V]
  → argmax 或 multinomial                  → [B]
```

能把这张图连着「为什么」一起讲出来——为什么 RoPE 在 Cache 之前、
为什么 softmax 沿最后一维、为什么只取最后一位、为什么 Cache 存 `Hkv` 份——
那这四周就没白过。
