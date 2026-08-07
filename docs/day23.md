# 学习单元 Day 23：Batch、Padding、Mask 与 Position

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 23**｜学习日 16

昨天的 `generate` 假设 batch 里所有序列**一样长**。真实场景不是这样。
今天处理长度不齐——这是**推理服务里最容易出错、也最少被讲清**的一块。

核心难点：

> **最后有效 token 位置不一定等于数组最后一列。**

这一句话就是今天的全部内容。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习 left / right padding | 阅读笔记 |
| ⬜ | 任务 2：推演三者的关系（mask / position / 取值位置） | `docs/concepts/14-padding-and-position.md` |
| ⬜ | 任务 3：改造 `generate` 支持不齐长度 | `src/mini_transformer/generate.py` |
| ⬜ | 任务 4：测试 | `tests/test_generation.py` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Padding and Truncation](https://huggingface.co/docs/transformers/pad_truncation) | [HF Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer) | 理清 left / right padding、`attention_mask` 和 `position_ids` |

---

## 任务 1：学习 left / right padding（30 分钟）

Day 1 已经对比过两种 padding 的 `input_ids` 长什么样
（见 [`concepts/00-tokenizer.md`](./concepts/00-tokenizer.md)）。今天的问题不同：
**它们对生成的影响是什么？**

设 batch 里两条序列，长度 3 和 5，`max_len=5`：

```text
Right padding                    Left padding
[a, b, c, P, P]                  [P, P, a, b, c]
[d, e, f, g, h]                  [d, e, f, g, h]
     ↑ 最后有效在第 2 列              ↑ 最后有效在第 4 列
       (与第二条不同)                  (与第二条相同)
```

### 为什么生成必须用 left padding

生成时要取 `logits[:, -1, :]`——**数组的最后一列**。

- **Left padding**：所有序列的最后有效 token 都在最后一列。`[:, -1, :]` 正确。
- **Right padding**：第 0 条序列的最后一列是 PAD。取 `[:, -1, :]`
  拿到的是「PAD 位置的预测」，**完全是垃圾**。

而且新 token 要往右追加。Right padding 下第 0 条的新 token
会被拼到 PAD 之后，序列变成 `[a,b,c,P,P,新]`——中间夹着 PAD，位置全乱。

**结论：生成一律用 left padding。** 训练用 right padding
（因为 loss 是按 mask 逐位置算的，位置不敏感）。
这就是为什么 HF 的 tokenizer 在生成时要设 `tokenizer.padding_side = "left"`，
不设会给一个明确的警告。

### 如果非要用 right padding

那就不能取 `[:, -1, :]`，必须按每条序列的**实际长度**索引：

```python
last_idx = attention_mask.sum(dim=1) - 1        # [B]
last_logits = logits[torch.arange(B), last_idx] # [B, V]
```

这是可行的，但后续追加新 token 依然麻烦。**用 left padding 更简单**，
不过上面这个 `gather` 技巧值得会——它是「最后有效位置不等于最后一列」的通用解法。

## 任务 2：推演三者的关系（50 分钟）

这是今天的核心。Left padding 下有**三样东西**必须同时改对，
漏一个就错。写进 `docs/concepts/14-padding-and-position.md`。

### 1. `attention_mask`

PAD 位置必须屏蔽，否则模型会去 attend 无意义的 PAD 向量。

```text
[P, P, a, b, c]  →  attention_mask = [0, 0, 1, 1, 1]
```

Day 8 实现的 padding mask 直接用。

### 2. `position_ids`（最容易漏的一个）

**PAD 不应该占用位置编号。** Left padding 下，第一个真实 token 的位置应该是 0：

```text
input_ids     = [P, P, a, b, c]
错的 position = [0, 1, 2, 3, 4]     ← 直接 arange，a 被当成第 2 个 token
对的 position = [0, 0, 0, 1, 2]     ← a 是第 0 个，PAD 位随便填
```

为什么这很重要：RoPE 按位置旋转（Day 13）。位置错了，
相对距离全部错位——**同一个 prompt 在不同 padding 长度下会得到不同的输出**。
这是个非常隐蔽的 bug，因为单条序列（无 padding）时完全正常。

HF 的算法是：

```python
position_ids = attention_mask.cumsum(dim=-1) - 1
position_ids.masked_fill_(attention_mask == 0, 0)      # PAD 位填 0（值无关，会被 mask 掉）
```

`cumsum - 1` 的效果：`[0,0,1,1,1]` → `[0,0,0,1,2]`。正好。

PAD 位置填什么其实**不影响结果**（它们的输出会被 attention mask 屏蔽），
填 0 只是为了避免负数索引越界。

### 3. 取 logits 的位置

Left padding 下就是 `[:, -1, :]`。这是选 left padding 的全部理由。

### 三者的关系总结

在笔记里画一张表，把「单条无 padding」和「left padding 的 batch」并排对比：

| | `input_ids` | `attention_mask` | `position_ids` | 取 logits |
|---|---|---|---|---|
| 单条 | `[a,b,c]` | `[1,1,1]` | `[0,1,2]` | `[:, -1, :]` |
| batch（left pad 到 5） | `[P,P,a,b,c]` | `[0,0,1,1,1]` | `[0,0,0,1,2]` | `[:, -1, :]` |

**两行的 `position_ids` 在真实 token 上必须完全一致**——这就是正确性的判据，
也是任务 4 第 1 条测试的内容。

### 生成过程中 position 怎么递增

Prefill 之后，新 token 的位置是 `真实长度`，而不是 `input_ids.shape[1]`：

```text
第 0 条：真实长度 3 → 新 token 的 position 是 3
第 1 条：真实长度 5 → 新 token 的 position 是 5
```

所以每条序列的 position 递增起点不同。维护一个 `[B]` 的
`current_position` 张量，每轮 `+1`。

**这一点会直接延续到 Day 27～28 的 `cache_position`。** 今天想清楚，Day 27 会轻松很多。

## 任务 3：改造 `generate` 支持不齐长度（60 分钟）

在 Day 21～22 的 `generate` 上加：

- 接收 `attention_mask` 参数（`None` 时视为全 1）
- 内部按 `cumsum` 规则算 `position_ids`
- 维护 `[B]` 的 `current_position`，每步递增
- 每步把新 token 的 mask 位置**追加为 1**
  （新生成的 token 都是有效的）

```text
每轮：
  input_ids      = cat([input_ids, next_token[:, None]], dim=1)
  attention_mask = cat([attention_mask, ones(B, 1)], dim=1)
  current_position += 1
```

注意 `attention_mask` 也要跟着长——忘了这一步的话，
新生成的 token 会被自己屏蔽掉。

## 任务 4：测试（40 分钟）

`tests/test_generation.py` 新增：

1. **padding 不变性（今天最重要的测试）**：
   同一个 prompt，单独跑一次 vs 放进 batch 用 left padding 跑一次，
   greedy 结果**必须完全相同**
2. `position_ids` 的 `cumsum` 计算正确（对着任务 2 的表逐个断言）
3. 长度不齐的 batch，每条序列的生成结果都和它单独跑的一致
4. PAD 位置对输出无影响：改变 PAD 的 token id，结果不变
5. right padding 时用 `gather` 取最后有效位置，结果与 left padding 一致
6. 生成过程中 `attention_mask` 正确增长

**第 1 条是本周门禁的一部分**，也是整个 batch 推理正确性的核心判据。
它一条就能覆盖 mask、position、取值位置三处错误。

第 4 条的思路是 Day 9 用过的那个套路：**验证「屏蔽生效」最可靠的方法是
改动被屏蔽的输入、确认输出不变。**

> 如果第 1 条不过，按这个顺序查：
> ① `position_ids` 是不是直接 `arange`（最常见）
> ② `attention_mask` 有没有传到 attention 里
> ③ 取 logits 的位置对不对

---

## 过关标准

- [ ] 能解释为什么生成必须用 left padding
- [ ] 能写出 `position_ids = cumsum(mask) - 1` 并解释为什么
- [ ] 知道 PAD 位置的 `position_ids` 填什么都行，以及为什么
- [ ] `generate` 支持 `attention_mask` 和不齐长度
- [ ] **padding 不变性测试通过**（单条 vs batch 结果一致）
- [ ] 会用 `gather` 取最后有效位置（right padding 的解法）
- [ ] `pytest -q` 全部通过

---

## 今日最重要的面试式问题

**Batch 里序列长度不同时，取 `logits[:, -1, :]` 会有什么问题？**

**Right padding 下会取到 PAD 位置的预测，是垃圾。** 因为短序列的最后一列是 PAD，
不是最后一个真实 token。

两个解法：用 **left padding**（所有序列的最后有效 token 都在最后一列），
或者按 `attention_mask.sum(1) - 1` 用 `gather` 取实际位置。
生成场景一般选 left padding，因为新 token 要往右追加，right padding 下会夹在 PAD 后面。

追问：**left padding 时 `position_ids` 该怎么算，直接 `arange` 有什么问题？**

要用 `attention_mask.cumsum(-1) - 1`，让第一个真实 token 的位置是 0。

直接 `arange` 的话，PAD 会占用位置编号，真实 token 的位置被整体后移。
RoPE 按位置旋转，位置错了相对距离就全错——后果是**同一个 prompt
在不同 padding 长度下输出不同**。单条序列时完全正常，所以这个 bug 极难发现。
