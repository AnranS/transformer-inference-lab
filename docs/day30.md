# 学习单元 Day 30：有 / 无 Cache 等价性测试与第 3 周门禁

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 30**｜学习日 21｜预算 5～6 小时

第 3 周最后一天。KV Cache 是**纯优化**——它不该改变任何输出。
今天用最严格的方式证明这一点。

> **第 3 周门禁**：Greedy 生成路径结果一致；Prefill 后每轮 Decode 只输入新 token；
> 动态与预分配 Cache 测试通过。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：逐步 logits 等价性 | `tests/test_cache_equivalence.py` |
| ⬜ | 任务 2：token 序列等价性 + 多配置扫描 | 同上 |
| ⬜ | 任务 3：六个必查错误逐个排查 | 本文清单 |
| ⬜ | 任务 4：写第五篇专题文档 | **`docs/05-prefill-decode-kv-cache.md`** |
| ⬜ | 任务 5：第 3 周门禁与段落验收 | 本文末 |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) | [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/generated/torch.testing.assert_close.html) | 完成有/无 Cache 的逐步 logits 与 token 等价性测试 |

---

## 任务 1：逐步 logits 等价性（90 分钟）

### 两条路径

固定权重、固定输入，跑两条路径：

```text
路径 A（无 Cache）  每轮输入完整序列，取 logits[:, -1, :]
路径 B（有 Cache）  Prefill 一次，之后每轮只输入 1 个新 token
```

**逐轮比较最后位置的 logits。**

```python
for step in range(n):
    logits_a = model(input_ids_full)[:, -1, :]          # 路径 A
    logits_b, cache = decode(next_token, cache, pos)    # 路径 B
    torch.testing.assert_close(logits_b, logits_a, rtol=..., atol=...)
```

### 为什么要逐轮比，不能只比最终结果

**因为 greedy 会掩盖误差。** 如果只比最终 token 序列，
一个第 5 轮就开始偏离的实现，可能因为 argmax 恰好选中同一个 token 而「通过」。

逐轮比 logits 能在**第一次偏离**时就报错，直接定位到出问题的步数。
这和 Day 19 分层测试的思路完全一致：**在最细的粒度上比，才有定位能力。**

### 容差怎么定

**不能用 `rtol=0, atol=0`。** 两条路径的浮点运算顺序不同：

- 路径 A：`[S, D] @ [D, D]` 一次算完
- 路径 B：Prefill 算 `[S, D]`，Decode 算 `[1, D]`

矩阵乘的分块策略不同，浮点加法的结合顺序就不同，结果会有**最后几位**的差异。
这是**正常且不可避免的**，不是 bug。

用 fp32 默认容差起步。如果失败，**先确认误差量级**：

```text
误差 ~1e-6  → 正常的浮点重排，放宽容差
误差 ~1e-2  → 有 bug，去查任务 3 的清单
误差 ~1.0   → 严重 bug，大概是 position 或 mask
```

**看误差量级来区分「浮点噪声」和「真 bug」**，这是今天最重要的判断技巧。
不要一失败就放宽容差——先看看差多少。

### 覆盖两种 Cache

路径 B 要分别用 `DynamicCache` 和 `StaticCache` 各跑一遍。
Day 29 已经测过两者互相等价，但和路径 A 的对比要各自做一次
（万一 `StaticCache` 有 Day 29 第 9 条那种未写区问题）。

## 任务 2：token 序列等价性 + 多配置扫描（70 分钟）

### token 序列必须完全相同

```python
torch.testing.assert_close(tokens_a, tokens_b, rtol=0, atol=0)
```

token id 是整数，**必须逐比特相同**。虽然 logits 有浮点噪声，
但噪声量级（`1e-6`）远小于 logits 之间的典型差距，argmax 不该被影响。

**如果 token 序列不同但 logits 只差 `1e-6`**，说明某一步出现了**接近平局**的两个候选
（差值小于浮点噪声）。这种情况在随机初始化的模型上偶尔会碰到。
处理方式：换个随机种子，或者检查一下是不是 logits 真的几乎相等
（如果是，这不算 bug，但要在测试里说明）。

### 多配置扫描（Notion 要求的三个维度）

| 维度 | 取值 | 想暴露什么 |
|---|---|---|
| `batch` | 1, 2, 4 | batch 维处理 |
| prompt 长度 | 1, 3, 8, 16 | 边界；**`S=1` 的 prompt 尤其重要** |
| 长度是否齐 | 齐 / 不齐（left pad） | Day 23 的内容与 Cache 的交互 |

再加两个我们自己关心的：

| 维度 | 取值 | 想暴露什么 |
|---|---|---|
| `max_new_tokens` | 1, 2, 10 | **`=1` 时只有 Prefill 没有 Decode** |
| 采样 | greedy / `top_k=1` | 确定性路径 |

**`max_new_tokens=1`** 值得单独测：这时候完全走不到 `decode()`，
能验证 `prefill` 独立正确。**`prompt_len=1`** 也值得：
此时 Prefill 的 `Sq=1`，形状和 Decode 相同但语义不同。

### 长度不齐 + Cache 是今天最容易错的组合

Day 23 处理了长度不齐，Day 27～28 处理了 Cache，两者叠加会暴露新问题：

- PAD 的 K/V 写进了 Cache（**允许**），但 mask 必须屏蔽它
- `cache_position`（Cache 下标，全 batch 相同）和 RoPE position
  （按 `cumsum` 算，每条不同）**是两个不同的量**

这一条测试通过，才算真正把 batch 推理做对了。

## 任务 3：六个必查错误逐个排查（80 分钟）

Notion 列的六条。**即使测试全绿，也逐条主动验证一遍**——
测试可能恰好没覆盖到。方法是**故意引入错误，确认测试变红**。

### 1. RoPE position 从 0 重新开始

```python
# 故意把 decode 的 position 改成 0
logits, cache = decode(tok, cache, cache_position=torch.zeros_like(pos))
```

**必须**导致等价性测试失败。如果没失败，说明你的测试没覆盖 RoPE
（可能 `max_new_tokens` 太小，或者模型太小看不出差异）。

### 2. causal mask 在 `Sq=1`、`Skv>1` 时错误

```python
# 故意用 tril(Sq, Skv) 构造 decode 的 mask
```

必须失败。这条在 Day 08 就预防过，现在验证防线有效。

### 3. K/V 在错误的 sequence 维拼接

```python
# 把 dim=-2 改成 dim=1
```

Day 27 加的断言应该**直接抛异常**，而不是让测试静默失败。
如果它静默通过了，说明断言写得不够严。

### 4. Cache 写入位置 off-by-one

```python
# 把 cache_position 改成 cache_position - 1
```

必须失败。这条最阴险，因为 `-1` 会**覆盖上一个 token 的 K/V**，
生成的文本会退化但不会崩。

### 5. Padding token 被写入有效 Cache

这条的正解是「允许写入，但 mask 必须屏蔽」（Day 28 的结论）。
验证方式：**把 PAD 位置的 token id 改成别的值**，
结果必须不变。变了说明 mask 没屏蔽住。

### 6. GQA 的 KV head 扩展方式错误

**Day 31 才实现 GQA**，今天 `Hq = Hkv`。
在笔记里记一句「本项目 Day 30 时点尚未实现 GQA，此项待 Day 31 验证」，
明天补上。

### 记录到文档

把这六条整理成一张表进 `docs/05-prefill-decode-kv-cache.md`：

| 错误 | 症状 | 检测方法 | 我的防线 |
|---|---|---|---|

「我的防线」一列写你代码里哪一处在防它（断言 / 接口设计 / 测试）。
**这张表是整个第 3 周最有价值的产出**——以后接手任何推理代码，
这六条就是排查清单。

## 任务 4：写第五篇专题文档（80 分钟）

**`docs/05-prefill-decode-kv-cache.md`**。建议结构：

1. **Prefill / Decode 的六项对比表**（Day 26）
2. **KV Cache 缓存什么、为什么能缓存、为什么 Q 不缓存**（Day 26）
3. **无 Cache 的 `O(n²)` 浪费**，含量化数据和那张对比曲线（Day 26）
4. **DynamicCache vs StaticCache**：数据结构、更新方式、开销对比（Day 27、29）
5. **`cache_position` 的语义**，以及它和 RoPE position 的关系（Day 28）
6. **预分配的代价与 PagedAttention 方向**（Day 29）
7. **六个必查错误的排查表**（任务 3）
8. **等价性验证方法与实测容差**（任务 1～2）

第 7 项是重点。第 3 项和第 4 项要放实测数据，不要只写理论。

## 任务 5：第 3 周门禁与段落验收（40 分钟）

### 门禁

- [ ] 逐步 logits 等价性通过（两种 Cache 各一遍）
- [ ] token 序列逐比特相同
- [ ] 多配置扫描通过，含 `prompt_len=1`、`max_new_tokens=1`、长度不齐
- [ ] 六个必查错误：五个已验证测试能捕捉（GQA 待 Day 31）
- [ ] **Prefill 后每轮 Decode 只输入新 token**（读一遍 `generate` 确认循环里没有拼接完整序列）
- [ ] `pytest -q` 全部通过

### 第 3 周段落验收（Notion 原文）

- [ ] 解释 KV Cache 缓存了什么
- [ ] 说明 Prefill 与 Decode 的区别
- [ ] 实现动态与预分配两种缓存
- [ ] 有/无 Cache 输出一致
- [ ] 解释 Cache position 与 RoPE position 的关系

最后一条容易答得含糊。清晰的答案：

> Decode 时它们是**同一个数**——新 token 的绝对位置，既决定 RoPE 的旋转角度，
> 又决定写入 Cache 的下标。
>
> 但在**长度不齐的 batch** 里两者会分离：
> Cache 下标全 batch 相同（left padding 保证有效数据右对齐），
> 而 RoPE position 要按 `attention_mask.cumsum(-1) - 1` 逐条算，各不相同。

### 顺手做一次性能对比

不是门禁，但值得现在测（Day 33～34 会正式做）：
同一个生成任务，无 Cache vs 有 Cache 的墙钟时间比。

在小配置上可能只快几倍（模型太小，开销占比高），
在 `L=12, D=768` 这种量级上会明显得多。记下这个数字，
第 4 周做完正式 benchmark 后回来对比。

---

## 过关标准

- [ ] 能解释为什么等价性测试要逐轮比 logits 而不只比最终 token
- [ ] 能解释为什么容差不能设 `0`（浮点运算顺序不同）
- [ ] 能通过误差量级区分「浮点噪声」和「真 bug」
- [ ] 六个必查错误的排查表完成
- [ ] `docs/05-prefill-decode-kv-cache.md` 八节完成，含实测数据
- [ ] 第 3 周段落验收五条全部能答，尤其最后一条
- [ ] 门禁全部通过

---

## 今日最重要的面试式问题

**怎么证明 KV Cache 的实现是正确的？**

**跑两条路径做等价性对比**：路径 A 每轮输入完整序列不用 Cache，
路径 B Prefill 后每轮只输入新 token。

关键是**逐轮比较 logits**，而不是只比最终 token 序列——
greedy 的 argmax 会掩盖误差，一个第 5 轮就开始偏离的实现可能碰巧选中同样的 token。
逐轮比才能在第一次偏离时定位。

再覆盖多个配置：`batch` 1/2/4、prompt 长度含 1、`max_new_tokens` 含 1、长度不齐的 batch。

追问：**两条路径的 logits 能要求逐比特相同吗？**

**不能。** 路径 A 算 `[S,D] @ [D,D]`，路径 B 算 `[1,D] @ [D,D]`，
矩阵乘的分块策略不同，浮点加法结合顺序就不同，最后几位必然有差异。

正确做法是用 fp32 默认容差，并**看误差量级判断性质**：
`~1e-6` 是正常的浮点重排；`~1e-2` 说明有 bug；`~1.0` 一般是 position 或 mask 错了。

**token 序列则必须逐比特相同**（`rtol=0, atol=0`），
因为浮点噪声远小于 logits 之间的典型差距，不该影响 argmax。
