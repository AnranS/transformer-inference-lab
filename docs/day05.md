# 学习单元 Day 05：广播、数值测试与第一篇专题文档

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 5**｜学习日 3

今天不加新算子，只做两件事：把前四个单元的 shape 知识**固化成规则**，
并学会**怎么判断两个张量算得一样**——后者是接下来所有 HF 对齐工作的地基。

如果学习日 2 有溢出（Day 3～4 的测试或笔记没写完），今天先补完再开始。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ✅ | 任务 1：学习广播规则 | [`00-tensor-conventions.md`](./00-tensor-conventions.md) |
| ✅ | 任务 2：广播陷阱实验 | [`../notebooks/day03_logits_and_softmax.ipynb`](../notebooks/day03_logits_and_softmax.ipynb)「7. 广播陷阱」 |
| ✅ | 任务 3：数值容差与 `assert_close` | [`concepts/05-numerical-tolerance.md`](./concepts/05-numerical-tolerance.md) |
| ✅ | 任务 4：写第一篇专题文档 | [`deliverables/01-autoregressive-language-model.md`](./deliverables/01-autoregressive-language-model.md) |
| ✅ | 任务 5：第一段验收自查 | 本文末；已完成 review |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [PyTorch Broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) | [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/generated/torch.testing.assert_close.html) | 复盘 shape 与广播规则，并学会写数值测试 |

---

## 任务 1：学习广播规则（25 分钟）

规则只有两条，但要背下来：

1. **从尾部对齐**两个张量的维度（右对齐，不是左对齐）
2. 每一对维度必须满足：**相等**、或**其中一个是 1**、或**其中一个不存在**

```text
[B, S, D]  和  [D]      → 尾部对齐，D 相等，缺的维当 1 → [B, S, D]  ✓
[B, S, D]  和  [S, 1]   → 尾部对齐：D vs 1 ✓，S vs S ✓ → [B, S, D]  ✓
[B, S, D]  和  [B, S]   → 尾部对齐：D vs S ✗                        ✗
```

第三行是最容易犯的错：`attention_mask` 是 `[B, S]`，
你想让它作用在 `[B, S, D]` 上，**不能直接加**——必须先 `unsqueeze(-1)` 变成 `[B, S, 1]`。

## 任务 2：广播陷阱实验（30 分钟）

在 notebook 新开一节「6. 广播陷阱」。这三个陷阱**都不报错**，但结果全错，
所以必须亲手看一遍它们长什么样。

**陷阱一：漏 batch 维**（Day 1～2 已经踩过，这次量化它）

```python
a = torch.randn(1, 5, 16)   # 正确 [B, S, D]
b = torch.randn(5, 16)      # 漏了 batch
print(a.mean(dim=1).shape)  # [1, 16]  对 5 个 token 求平均
print(b.mean(dim=1).shape)  # [5]      对 16 个特征求平均 ← 语义全错，不报错
```

**陷阱二：mask 维度没对齐**（这是第 2 周 Attention 最高频的 bug，提前见一面）

```python
scores = torch.randn(2, 4, 8, 8)   # [B, H, Sq, Skv]
mask    = torch.randn(2, 8)        # [B, Skv]
# scores + mask 会报错吗？自己先猜，再运行
# 正确做法：mask.view(2, 1, 1, 8) 再相加
```

**陷阱三：意外的外积膨胀**

```python
x = torch.randn(2, 5, 1)
y = torch.randn(2, 1, 5)
print((x + y).shape)   # [2, 5, 5] ← 你可能只想要 [2, 5]
```

**关键结论：** 广播的危险不在于它会报错，而在于它**太经常不报错**。
凡是两个张量维数不同就要相加的地方，都手写一次 `unsqueeze` / `view` 把意图写明，
不要指望广播猜对。

## 任务 3：数值容差与 `assert_close`（25 分钟）

必读：[torch.testing.assert_close](https://docs.pytorch.org/docs/stable/generated/torch.testing.assert_close.html)

从今天起，**不要再用 `==` 或 `torch.equal` 比较浮点结果**。判据是：

```text
|actual - expected| <= atol + rtol * |expected|
```

`atol` 管绝对误差（保护接近 0 的值），`rtol` 管相对误差（保护数值很大的值），两者相加。

PyTorch 按 dtype 给的默认容差（这几个数记住，第 4 周对齐 HF 时天天用）：

| dtype | rtol | atol |
|---|---|---|
| `float64` | `1e-7` | `1e-7` |
| `float32` | `1.3e-6` | `1e-5` |
| `float16` | `1e-3` | `1e-5` |
| `bfloat16` | `1.6e-2` | `1e-5` |

**bf16 的 rtol 比 fp32 松约 12000 倍。** 原因是 bf16 只有 8 位有效尾数
（fp32 有 24 位），相对精度约 0.4%。它拿指数范围换了精度——
所以 bf16 适合推理，而**用 bf16 做数值对齐测试几乎测不出实现错误**。

由此得到一条今后必须遵守的规则：

> **正确性对齐一律在 fp32 下做。** bf16 只用来测性能和溢出行为。

把这三点写进 `docs/concepts/05-numerical-tolerance.md`：默认容差表、判据公式、上面这条规则。

## 任务 4：写第一篇专题文档（55 分钟）

这是路线里第一个正式交付物：**`docs/deliverables/01-autoregressive-language-model.md`**。

和 `concepts/` 下的笔记不同——那些是边学边记的碎片，这篇要**综合成一条完整叙事**，
写给「一个月后忘光了的自己」看。四个必需部分：

**1. 文本到 token 的流程图**

```text
文本 → tokenizer.encode → input_ids [B,S] → Embedding → hidden_states [B,S,D]
     → LM Head → logits [B,S,V] → 取 [:, -1, :] → argmax → next_token_id [B]
     → 拼回 input_ids → 回到第一步
```

**2. 五个概念的区别**（这是最重要的一节，也是过关问题的第一题）

| 概念 | 形状 | 类型 | 含义 |
|---|---|---|---|
| token ID | `[B, S]` | `long` | 离散索引，只是查表下标 |
| embedding / hidden state | `[B, S, D]` | `float` | 连续向量表示 |
| logits | `[B, S, V]` | `float` | 未归一化分数，可负、和不为 1 |
| probability | `[B, S, V]` | `float` | softmax 之后，`[0,1]` 且和为 1 |

hidden state 和 embedding 的关系也要说清：embedding 是**第 0 层**的 hidden state，
后面每个 Decoder Block 都会产出新的 hidden state，形状始终 `[B,S,D]`。

**3. 自回归生成伪代码**（写你自己 Day 4 实现的那版，不要抄现成的）

**4. 为什么第 t 个 token 依赖之前所有 token**

答案要包含两层：

- **建模上**：语言模型定义就是 `P(x_t | x_1..x_{t-1})`，链式法则展开联合概率
- **计算上**：Causal Attention 让位置 `t` 能看到 `0..t`，所以 hidden state 里编码了全部历史

第二层现在还是欠着的——今天的模型没有 Attention，所以位置 `t` 的 hidden state
**实际上只依赖 token t 本身**。在文档里**明确标注这个缺口**，第 2 周补上后回来改。

## 任务 5：第一段验收自查（15 分钟）

这一段（学习单元 Day 1～5）的验收标准，逐条自问，不看资料：

- [x] 能从文本一路讲到 next token
- [x] 能写出 embedding 和 LM Head 的 shape
- [x] 能解释 logits 为什么不需要先 Softmax 再 argmax
- [x] 完成 `tests/test_embedding.py`

答不上来的条目，回对应单元补。**别带着漏洞进第 2 周**——Attention 会立刻放大所有 shape 上的糊涂。

---

## 过关标准

- [x] 能不看资料说出广播的两条规则
- [x] 亲手复现了三个广播陷阱，并能解释为什么不报错
- [x] 记住 fp32 和 bf16 的默认容差量级差异，以及「对齐只在 fp32 做」这条规则
- [x] 完成 `docs/deliverables/01-autoregressive-language-model.md` 四个部分
- [x] `pytest -q` 全部通过
- [x] 第一段四条验收全部能答

---

## 今日最重要的面试式问题

**两个张量数值上「相等」该怎么判断？**

不能用 `==`。要用 `|a-b| <= atol + rtol*|b|`，且**容差必须跟着 dtype 走**。

追问：**为什么 bf16 的 rtol 比 fp32 松一万倍，这对测试策略意味着什么？**

因为 bf16 只有 8 位尾数，相对精度约 0.4%。意味着**bf16 下的对齐测试没有鉴别力**——
一个真有 bug 的实现也可能在 bf16 容差内通过。所以正确性测试一律用 fp32，
bf16 只用来验证性能和是否溢出。
