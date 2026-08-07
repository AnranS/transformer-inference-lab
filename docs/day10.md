# 学习单元 Day 10：HF 源码、数值对齐与第 1 周门禁

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 10**｜学习日 7

第 1 周最后一天。今天做三件事：读一遍工业级实现、补齐测试和文档、**过门禁**。

> **第 1 周门禁**：Naive Attention 与 PyTorch SDPA 数值对齐，
> causal / padding mask 测试全部通过。**未通过不得进入 Decoder Block。**

这条门禁不是形式。Attention 的 shape 或 mask 有问题，
后面 Decoder Block、完整模型、KV Cache 每一层都会被污染，而且越往后越难定位。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：读 HF `LlamaAttention` 源码 | 阅读笔记 |
| ⬜ | 任务 2：建立源码到自研实现的对应表 | `docs/concepts/07-hf-llama-attention.md` |
| ⬜ | 任务 3：补齐测试 | `tests/test_attention.py`、`tests/test_mask.py` |
| ⬜ | 任务 4：Attention shape notebook | `notebooks/day10_attention_shapes.ipynb` |
| ⬜ | 任务 5：写第二篇专题文档 | **`docs/02-attention-and-masks.md`** |
| ⬜ | 任务 6：门禁自查 | 本文末 |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [HF Llama 文档](https://huggingface.co/docs/transformers/model_doc/llama) | **只找 `LlamaAttention`**，标注 Q/K/V 投影、RoPE、Attention 和输出投影 |

---

## 任务 1：读 HF `LlamaAttention` 源码（60 分钟）

**只读 `LlamaAttention` 这一个类**，不要顺着往下读整个文件——那是 Day 15 和 Day 17 的事。

本地就有源码，比在 GitHub 上翻更快：

```bash
python -c "import transformers.models.llama.modeling_llama as m; print(m.__file__)"
```

按这个顺序找，每找到一处就在自己的笔记上记下行号：

| 找什么 | 关键词 | 对应你的哪部分 |
|---|---|---|
| 四个投影的定义 | `q_proj` `k_proj` `v_proj` `o_proj` | Day 9 任务 2 |
| head 数与维度 | `num_attention_heads` `num_key_value_heads` `head_dim` | `Hq` / `Hkv` / `Dh` |
| 拆分与 transpose | `.view(...).transpose(1, 2)` | Day 9 任务 1 |
| 位置编码 | `apply_rotary_pos_emb` | **还没实现**（Day 14） |
| KV head 复制 | `repeat_kv` | **还没实现**（Day 31） |
| 缩放系数 | `self.scaling` | 确认它是 `head_dim ** -0.5` 而非 `hidden_size ** -0.5` |
| attention 计算 | `eager_attention_forward` 或 `ALL_ATTENTION_FUNCTIONS` | Day 7 的 `naive_attention` |
| 合并与输出投影 | `.reshape(...)` 然后 `o_proj` | Day 9 任务 1 |

**注意两件事**：

1. HF 的实现里 `apply_rotary_pos_emb` 在**投影之后、attention 之前**，
   而且**只作用于 Q 和 K，不作用于 V**。这是 Day 13～14 的重点，今天先记住位置。
2. `repeat_kv` 的存在说明 HF 的 GQA 实现是**把 KV head 复制**到 `Hq` 份。
   Day 31 你会发现这不是唯一做法（也可以用 SDPA 的 `enable_gqa`），代价不同。

HF 版本迭代较快，函数名可能和上面略有差异。**以你本地装的版本为准**，
对不上就在文件里搜关键词。

## 任务 2：建立源码到自研实现的对应表（45 分钟）

落到 `docs/concepts/07-hf-llama-attention.md`。做一张两列表：

| HF `LlamaAttention` | 我的实现 | 差异与原因 |
|---|---|---|
| `q_proj` / `k_proj` / `v_proj` | `MultiHeadAttention` 的三个 `nn.Linear` | — |
| `apply_rotary_pos_emb` | **缺** | Day 14 补 |
| `repeat_kv` | **缺**（当前 `Hkv = Hq`） | Day 31 补 |
| ... | ... | ... |

**「差异与原因」这一列是重点。** 它是你后面几周的待办清单——
每补上一项就划掉一行，到 Day 19 做 HF parity 时这张表就是你的对照清单。

这个单元的要求是「能从 HF `LlamaDecoderLayer` 反向定位到自己的实现」。
这张表就是实现那个能力的载体。

## 任务 3：补齐测试（75 分钟）

把 Day 07～09 三天的测试整理成能长期维护的样子：

- `tests/test_attention.py`：Day 9 列的 10 条
- `tests/test_mask.py`：Day 8 列的 7 条

整理时做三件事：

1. **去重**。三天里可能写了重复的 shape 检查，合并。
2. **参数化**。用 `@pytest.mark.parametrize` 覆盖多组 `(B, Hq, S, Dh)`，
   尤其要包含 `B=1`、`S=1`、`Hq=1` 这些边界。
3. **补上 `Sq != Skv` 的用例**。这是今天必须新增的：
   `Sq=1, Skv=8` 的解码形态，现在就要测，不能等到 Day 26～30。

再加一条**门禁级测试**，用显眼的名字标出来：

```python
def test_week1_gate_naive_attention_matches_sdpa():
    """第 1 周门禁：naive attention 必须与 PyTorch SDPA 数值对齐（fp32）。"""
```

以后每次改 attention 都会跑到它。

## 任务 4：Attention shape notebook（45 分钟）

新建 `notebooks/day10_attention_shapes.ipynb`。这是目录规范里的第一个专用 notebook，
目的是**把形状变换可视化**，以后忘了随时回来看。

四节：

1. **形状流水线**：从 `[B,S,D]` 到 `[B,S,D]`，每一步打印形状，做成一张表
2. **mask 长什么样**：`S=4` 的因果掩码、带 PAD 的 padding 掩码、两者合成的结果
3. **`Sq != Skv`**：`Sq=1, Skv=8` 时各中间张量的形状，和 Prefill 形态对比
4. **注意力权重热图**：随机权重下画一张 `[S, S]` 的注意力图，
   确认下三角结构肉眼可见

第 4 节用 matplotlib 画。**如果环境里没有 matplotlib，用 `print` 打印字符矩阵也完全够用**
（`■` 表示可见、`·` 表示屏蔽），别为了画图卡在装包上。

## 任务 5：写第二篇专题文档（75 分钟）

**`docs/02-attention-and-masks.md`**，第 1 周的正式交付物。必须包含：

1. **Attention 公式**与每一步的形状
2. **Q/K/V 的语义**（来自 Day 6）
3. **为什么除 `sqrt(Dh)`** 的方差推导
4. **`[B,S,D]` ↔ `[B,Hq,S,Dh]` 的双向变换**，含为什么要 transpose、为什么用 reshape
5. **四种 mask 的组合表**（causal/padding × boolean/additive）
6. **`Sq != Skv` 时因果掩码的正确构造**（右下角对齐）
7. **五个常见 bug** 及各自的检测方法
8. **与 HF `LlamaAttention` 的对应关系**和当前缺口
9. **数值对齐结果**：fp32 下与 SDPA 的最大误差、bf16 下的最大误差

第 9 条要写具体数字，不要写「基本一致」。这些数字在第 4 周设计 HF parity 容差时要用。

## 任务 6：门禁自查（15 分钟）

逐条确认，不能跳：

- [ ] `naive_attention` 与 `F.scaled_dot_product_attention` 在 **fp32** 下 `assert_close` 通过
- [ ] `MultiHeadAttention`（`Hq > 1`）与 SDPA 对齐
- [ ] causal mask 测试全部通过，含 `Sq == Skv` 和 `Sq=1, Skv>1` 两种形态
- [ ] padding mask 测试全部通过
- [ ] causal + padding 合成后能广播到 `[B, Hq, Sq, Skv]`
- [ ] 因果 attention 第 0 行概率为 `[1, 0, 0, ...]`
- [ ] `pytest -q` 全部通过

### 第 1 周段落验收

- [ ] 不看资料写出 Attention 公式
- [ ] 从 `[B,S,D]` 推导到 `[B,Hq,S,Dh]` 再还原
- [ ] 解释为什么 attention score 是 `S × S`
- [ ] naive 实现与 SDPA 数值对齐
- [ ] 能分别处理 causal mask 和 padding mask

> **任何一条没过，不要开始第 2 周。** 宁可多花一天。

---

## 过关标准

- [ ] 在 HF 源码里定位到了表格中的全部条目并记下行号
- [ ] 完成 HF ↔ 自研实现对应表，缺口列清晰
- [ ] `tests/test_attention.py` 与 `tests/test_mask.py` 整理完成，含 `Sq != Skv` 用例
- [ ] `notebooks/day10_attention_shapes.ipynb` 四节完成
- [ ] `docs/02-attention-and-masks.md` 九个部分完成，含具体误差数字
- [ ] 门禁七条全部通过

---

## 今日最重要的面试式问题

**为什么 attention score 矩阵是 `S × S`，这带来什么后果？**

因为每个 query 都要和每个 key 算一次相似度，`Sq` 个 query × `Skv` 个 key。
Prefill 时 `Sq = Skv = S`，所以是 `S × S`。

后果是**显存和计算都随 `S²` 增长**：`S` 翻倍，attention 中间矩阵变 4 倍。
这是长上下文的核心瓶颈，也是 FlashAttention（不实体化这个矩阵）和
稀疏注意力存在的理由。

追问：**Decode 阶段还是 `S × S` 吗？**

不是。有 KV Cache 时 `Sq = 1`，中间矩阵是 `1 × Skv`，**随 context 线性增长而非平方**。
所以 Prefill 是计算密集（大 GEMM）、Decode 是访存密集（读整个 Cache）——
这两种截然不同的性能特征，是第 4 周所有 benchmark 的主题。
