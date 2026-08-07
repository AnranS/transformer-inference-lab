# 学习单元 Day 09：Multi-Head Attention

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 9**｜学习日 6｜预算 5 小时

把 Day 7～08 的单头 attention 扩成多头，并加上输入输出投影，
得到一个完整的、可以放进 Decoder Block 的 Attention 模块。

```text
[B, S, D]
  → Wq/Wk/Wv        → 三个 [B, S, D]
  → view + transpose → 三个 [B, Hq, S, Dh]
  → attention（Day 7～8 的实现）→ [B, Hq, S, Dh]
  → transpose + reshape → [B, S, D]
  → Wo              → [B, S, D]
```

今天是长任务日（5 小时），**一半时间在调 shape**。这是正常的。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习 head 的拆分与合并 | 阅读笔记 |
| ⬜ | 任务 2：实现 `MultiHeadAttention` | `src/mini_transformer/attention.py` |
| ⬜ | 任务 3：五个常见 bug 逐个自查 | 本文清单 |
| ⬜ | 任务 4：单元测试 | `tests/test_attention.py` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [PyTorch Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html) | [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) | 理解 head reshape、transpose、合并与 SDPA 调用方式 |

---

## 任务 1：学习 head 的拆分与合并（30 分钟）

### 多头到底在做什么

一个头只能学一种「匹配模式」。多头把 `D` 维切成 `Hq` 份，每份 `Dh = D / Hq` 维，
**每份独立做一次完整的 attention**，最后拼回来再过一个输出投影 `Wo`。

关键：多头**不增加**参数量和计算量（相对于单头 `D` 维），只是把同样的预算拆成并行的几路。
`Wq` 依然是 `[D, D]`，只是在计算时把输出看成 `Hq` 个 `Dh` 维的块。

### 拆分：`view` + `transpose` 两步

```text
[B, S, D]
  .view(B, S, Hq, Dh)      拆最后一维        → [B, S,  Hq, Dh]
  .transpose(1, 2)         把 head 提到前面  → [B, Hq, S,  Dh]
```

**为什么要 transpose？** 因为 attention 要在 `(S, Dh)` 这两维上做矩阵乘，
必须让 `B` 和 `Hq` 都变成「批」维度。`[B, S, Hq, Dh]` 的话矩阵乘会在错误的维度上进行。

### 合并：反向两步

```text
[B, Hq, S, Dh]
  .transpose(1, 2)         → [B, S, Hq, Dh]
  .reshape(B, S, D)        → [B, S, D]
```

**必须先 transpose 回来再合并。** 顺序错了不会报错，只会静默算错——见任务 3 的 bug 2。

### 为什么这里是 `reshape` 而不是 `view`

`transpose` 只改变 stride，不搬动内存，结果是**非连续**张量。
`view` 要求连续，会抛 `RuntimeError`；`reshape` 会在需要时自动复制。
两种写法都可以：

```python
x.transpose(1, 2).reshape(B, S, D)              # 推荐
x.transpose(1, 2).contiguous().view(B, S, D)    # 等价，更啰嗦
```

## 任务 2：实现 `MultiHeadAttention`（110 分钟）

契约：

- `__init__(hidden_size, num_heads, *, bias=False)`
- `hidden_size % num_heads != 0` → `ValueError`（`Dh` 必须是整数）
- `forward(hidden_states, attn_mask=None)`：`[B, S, D]` → `[B, S, D]`
- 输入不是三维 → `ValueError`；最后一维不等于 `hidden_size` → `ValueError`
- 内部复用 Day 7～8 写的 `naive_attention` 和 mask 构造函数
- `bias=False` 对齐 Llama（Llama 的 q/k/v/o 投影都不带 bias）

四个投影都是 `nn.Linear(D, D, bias=False)`。先不做 GQA——
`Hkv = Hq`，Day 31 再拆开。

**建议的实现顺序**（不要一次写完再跑）：

1. 先只写投影 + 拆分，`forward` 直接返回 `Q` 的形状，打印确认是 `[B, Hq, S, Dh]`
2. 接上 attention，返回 `[B, Hq, S, Dh]`，确认形状
3. 接上合并 + `Wo`，确认回到 `[B, S, D]`
4. 最后才加 mask

每一步都跑一次、打印一次形状。5 小时的任务如果一口气写完再调，会调更久。

## 任务 3：五个常见 bug 逐个自查（70 分钟）

Notion 明确列了五个高频 bug。**逐条对着自己的代码看一遍**，并且为每一条设计一个能抓住它的断言。

### Bug 1：Softmax 维度写错

写成 `dim=-2` 或 `dim=1`。代码正常跑，结果全错。

**怎么抓**：`probs.sum(dim=-1)` 必须全为 1。写错维度后这个和不是 1。

### Bug 2：transpose 后直接合并（最阴险的一个）

```python
# 错：跳过了 transpose 回来的步骤
out.reshape(B, S, D)          # out 是 [B, Hq, S, Dh]
```

元素总数 `B × Hq × S × Dh` 恰好等于 `B × S × D`，
所以 **`reshape` 会成功**，但数据被彻底打乱——每个位置拿到的是别的位置、别的 head 的数值。

**怎么抓**：`Hq = 1` 时，MHA 的输出必须和单头 attention 完全一致。
`Hq=1` 时 transpose 是恒等操作，这个 bug 隐身；但配合下面这条就能抓住：
构造 `Wo = I`、`Wq = Wk = Wv = I`，用 Day 6 那个 4×4 例子，
输出必须等于手算结果。

### Bug 3：mask 广播维度错误

`[B, Skv]` 直接加到 `[B, Hq, Sq, Skv]` 上。

**怎么抓**：Day 07～08 的 `test_mask.py` 第 2 条——因果 attention 第 0 行必为 `[1,0,0,...]`。

### Bug 4：缩放用了 `D` 而不是 `Dh`

**怎么抓**：`Hq=1` 时 `D == Dh`，测不出来。必须用 `Hq>1` 的配置和 SDPA 对齐
（SDPA 默认 `scale=1/sqrt(query.size(-1))`，而它的 query 是 `[..., Dh]`）。

### Bug 5：合并 head 后顺序错误

和 Bug 2 同源，但表现为 `view(B, S, Hq, Dh)` 里 `Hq` 和 `Dh` 写反：

```python
.view(B, S, Dh, Hq)    # 错：拆分顺序和 D = Hq × Dh 的内存布局不符
```

**怎么抓**：同 Bug 2 的恒等权重测试。

> **一个统一的抓法**：把所有投影设成单位矩阵、`Hq=2`、用固定输入，
> 手算或用 SDPA 算出期望值，写成一条 `assert_close`。
> 这一条测试能同时覆盖上面五个 bug 中的四个。

## 任务 4：单元测试（90 分钟）

`tests/test_attention.py` 至少包含：

1. `[B,S,D]` 进，`[B,S,D]` 出
2. `hidden_size % num_heads != 0` 时构造报错
3. 输入不是三维时报错
4. `probs.sum(-1)` 全为 1（需要暴露中间概率，或单独测 `naive_attention`）
5. `Hq=1` 时与单头 attention 结果一致
6. `Hq>1` 时与 `F.scaled_dot_product_attention` 对齐（**本周门禁**）
7. 恒等权重 + 固定输入下，输出等于 Day 6 手算结果
8. 默认 `bias=False`（四个投影的 `bias` 都是 `None`）
9. `torch.inference_mode()` 下能跑，输出 `requires_grad` 为 `False`
10. `attn_mask` 传入后，被屏蔽位置对输出无影响
    （改动被屏蔽位置的 `V`，输出不变——这条比检查概率更直接）

第 10 条的思路值得记住：**验证「屏蔽生效」最可靠的方法不是看概率，
而是改动被屏蔽的输入、确认输出不变。** 这个套路在 Day 30 验证 KV Cache 时还会用到。

Day 01～04 的教训继续适用：写完测试后**临时删掉被测逻辑，确认测试会变红**。

---

## 过关标准

- [ ] 能不看资料写出 `[B,S,D]` → `[B,Hq,S,Dh]` → `[B,S,D]` 的完整变换
- [ ] 能解释为什么拆分后要 `transpose`，以及为什么合并要用 `reshape`
- [ ] `Hq>1` 时与 SDPA 数值对齐（fp32）
- [ ] 五个常见 bug 每个都有一条能抓住它的测试
- [ ] 恒等权重测试通过，输出等于 Day 6 手算结果
- [ ] `pytest -q` 全部通过

---

## 今日最重要的面试式问题

**多头注意力的价值是什么？它增加了多少参数量？**

价值是让模型在**同一层内并行学习多种匹配模式**——一个头可能关注语法邻接，
另一个关注长距离指代。单头只能学一种加权方式。

**参数量不变。** `Wq/Wk/Wv/Wo` 依然是 `[D, D]`，多头只是把 `D` 切成 `Hq` 份分别做 attention。
增加的是 attention 中间矩阵的数量：从 1 个 `[S, S]` 变成 `Hq` 个 `[S, S]`，
这部分是**显存**开销而非参数开销。

追问：**为什么合并 head 之后还需要一个 `Wo`？**

因为拼接只是把 `Hq` 个独立的 `Dh` 维结果并排放着，各头之间还没有交互。
`Wo` 负责把多个头的结论**混合**成一个 `D` 维表示。没有 `Wo`，
后面的 MLP 看到的就是几段互不相干的向量拼接。
