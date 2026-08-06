# Day 02：LM Head 与 logits

今天把 Day 01 的输出接上出口，让流水线第一次能吐出「下一个 token」：

```text
hidden_states [B, S, D] → LM Head → logits [B, S, V] → 取最后一位 → next token
```

## 进度与落点


| 进度  | 任务                          | 落点                                                                             |
| --- | --------------------------- | -------------------------------------------------------------------------------- |
| ⬜   | 任务 0：准备                     | 新建 `lm_head.py` / `tiny_lm.py` / `test_lm_head.py` 骨架                        |
| ⬜   | 任务 1：学习 LM Head 与 logits    | `[concepts/02-lm-head.md](./concepts/02-lm-head.md)`                             |
| ⬜   | 任务 2：logits 与 softmax 实验    | notebook「4. logits 与 softmax」                                                  |
| ⬜   | 任务 3：实现 `LMHead`            | `[../src/mini_transformer/lm_head.py](../src/mini_transformer/lm_head.py)`       |
| ⬜   | 任务 4：串成最小模型，预测下一个 token     | `[../src/mini_transformer/tiny_lm.py](../src/mini_transformer/tiny_lm.py)`       |
| ⬜   | 任务 5：Weight tying 与参数量对比    | `tie_weights()` + notebook 对比                                                   |
| ⬜   | 任务 6：单元测试                   | `[../tests/test_lm_head.py](../tests/test_lm_head.py)`                           |
| ⬜   | 任务 7：概念笔记与张量约定收尾            | `[concepts/00-tensor-conventions.md](./concepts/00-tensor-conventions.md)`       |


验收：`pytest -q` 全部通过，并能口述上面那条数据流。

预算约 3.8 小时。

---



## 关于顺序的说明

今天**跳过** Positional Embedding 和 Attention，直接从 embedding 接到出口。

这是刻意的：先用最少的零件打通一条端到端能跑的链路，后面每加一层都能立刻验证「输出还对不对」。
如果先堆 attention，要等很久才能看到第一个 token 被预测出来。

代价是这个模型现在**没有位置概念**（Day 01 已经验证过 Token Embedding 查不出位置），
预测结果毫无意义。Positional Embedding 是欠着的债，记在这里别忘：

- [ ] Positional Embedding（Day 03 补）

---



## 任务 0：准备（10 分钟）

建三个空骨架，只写签名和 docstring，方法体留 `raise NotImplementedError`：

```text
src/mini_transformer/lm_head.py     LMHead
src/mini_transformer/tiny_lm.py     TinyLM
tests/test_lm_head.py               （先空着）
```

`src/mini_transformer/__init__.py` 里记得导出新类。

---



## 任务 1：学习 LM Head 与 logits（25 分钟）

必读：PyTorch `nn.Linear`、`torch.softmax`

只需要搞明白：

1. LM Head 就是一个 `nn.Linear(D, V)`：把每个位置的 D 维向量，映射成词表上 V 个分数
2. 形状怎么变：`[B, S, D]` → `[B, S, V]`
3. logits 是**未归一化的分数**：可以是负数、可以大于 1、加起来不等于 1
4. softmax 做了什么，以及**什么时候才真的需要它**
5. `argmax(logits)` 和 `argmax(softmax(logits))` 是什么关系
6. 为什么 GPT-2 的 lm_head 是 `bias=False`
7. `nn.Linear(D, V)` 的 `weight` 形状为什么是 `[V, D]` 而不是 `[D, V]`

第 7 条是明天 weight tying 的伏笔，别跳过。

**不要深入**采样策略（top-k / top-p / beam search）和训练损失函数。

笔记见：`[concepts/02-lm-head.md](./concepts/02-lm-head.md)`

---



## 任务 2：logits 与 softmax 实验（35 分钟）

在 `experiments/tokenizer_playground.ipynb` 新开一节「4. logits 与 softmax」，
沿用第 3 节的风格：每个实验一个 markdown + 一个 code cell。

四个实验：

1. **logits 不是概率**：打印 min / max / sum，确认有负数、和不等于 1
2. **softmax 之后才是**：确认全部落在 `[0, 1]` 且每行和为 1
3. **argmax 不变**：验证 `logits.argmax(-1)` 和 `softmax(logits).argmax(-1)` 完全相同
4. **温度**：对 `logits / T` 做 softmax，比较 `T=0.5 / 1.0 / 2.0` 的分布尖锐程度

**关键结论：** softmax 是单调变换，不改变排序。
所以贪心解码（只要最大的那个）根本不需要算 softmax；
只有要**概率值**（采样、算困惑度、看置信度）时才需要。

在 GPT-2 尺度上，每步对 50257 个数做一次 softmax 是纯浪费。

---



## 任务 3：实现 `LMHead`（40 分钟）

在 `src/mini_transformer/lm_head.py` 中实现。

契约：

- 输入 `[B, S, D]`，输出 `[B, S, V]`
- `bias` 默认 `False`，对齐 GPT-2
- 输入不是三维 → `ValueError`
- 最后一维不等于 `hidden_size` → `ValueError`（错误信息要带上实际形状）
- 提供一个**只算最后一个位置**的路径（见任务 4）

沿用 Day 01 的习惯：错误信息一律带上实际值，不要只说「形状不对」。

理解每一行，不要直接把它当最终答案复制过去。

---



## 任务 4：串成最小模型，预测下一个 token（40 分钟）

在 `src/mini_transformer/tiny_lm.py` 里把两块拼起来：

```text
TinyLM = TokenEmbedding + LMHead

forward(input_ids [B, S]) -> logits [B, S, V]
predict_next_token(input_ids [B, S]) -> next_ids [B]
```

`predict_next_token` 取 `logits[:, -1, :]` 得到 `[B, V]`，再 argmax 得到 `[B]`。

**动手前先想清楚这两个问题**：

- 为什么只要最后一个位置？前面 S-1 个位置的 logits 是干什么用的？
- 因果语言模型里，位置 `i` 的 logits 预测的是哪个位置的 token？

想明白之后，再想第三个——这是今天最有价值的一点：

> 既然解码时只用最后一个位置，那 `[B, S, D] @ [D, V]` 这个矩阵乘
> 是不是**从一开始就只该算最后一行**？

在 GPT-2 尺度上这两者差距是：

```text
预填充 [1, 1024, 768] @ [768, 50257] = 79.0   GFLOP
解码   [1,    1, 768] @ [768, 50257] =  0.077 GFLOP   ← 差 1024 倍
```

先算完整个 `[B, S, V]` 再切片，等于白做 1024 倍的计算。
所以 `LMHead` 要留一条只处理最后一位的路径。

---



## 任务 5：Weight tying 与参数量对比（30 分钟）

给 `TinyLM` 加一个 `tie_weights()`，让 `lm_head` 和 `embedding` 共享**同一个张量**。

为什么能直接共享、不用转置：两者权重形状本来就一样。

```text
nn.Embedding(V, D).weight   -> [50257, 768]
nn.Linear(D, V).weight      -> [50257, 768]
```

验证与对比：

1. 用 `is` 确认两个 `weight` 是同一个对象，不只是数值相等
2. 改动其中一个，另一个跟着变
3. 统计参数量，在 GPT-2 尺度上对比

```text
untied  两份权重  77.2M 参数  fp32 294 MB
tied    一份权重  38.6M 参数  fp32 147 MB
省下 38.6M —— GPT-2 small 全模型才 124M
```

`GPT2Config().tie_word_embeddings` 是 `True`，GPT-2 确实这么做。

**想清楚为什么这么做是合理的**：`hidden @ W.T` 的第 `i` 个分数，
就是 hidden 向量和「token i 的 embedding 向量」的点积。
所以 LM Head 在问的是「当前状态和哪个 token 的向量最像」——
用同一张表既合理又省一半参数。

---



## 任务 6：单元测试（30 分钟）

`tests/test_lm_head.py` 至少包含：

1. 输入 `[B,S,D]`，输出是 `[B,S,V]`
2. 输入不是三维时报错
3. 最后一维和 `hidden_size` 不匹配时报错
4. 默认不带 bias（`lm_head.bias is None`）
5. `tie_weights()` 之后两个 weight 是同一个张量
6. tied 之后总参数量确实减少
7. `predict_next_token` 输出形状是 `[B]`，且每个值都落在 `[0, V)`
8. **只算最后一位的路径，结果和「全量算完再切片」完全一致**

第 8 条是重点：它验证的是那个 1024 倍的优化**没有改变语义**。
性能优化必须配一条等价性测试，否则你不知道自己是变快了还是算错了。

Day 01 的教训记得：`test_range_check_can_be_disabled` 曾经在开关根本没接线时也是绿的。
写完之后自己验一遍——**把被测的那段逻辑临时删掉，看测试会不会变红**。不会变红的测试是摆设。

验收：

```bash
pytest -q
```

---



## 任务 7：概念笔记与张量约定收尾（20 分钟）

`[concepts/02-lm-head.md](./concepts/02-lm-head.md)` 补齐，
并在 `[concepts/00-tensor-conventions.md](./concepts/00-tensor-conventions.md)` 的形状表里加一行：

| 张量 | 形状 | 说明 |
|------|------|------|
| `logits` | `[B, S, V]` | 未归一化分数，不是概率 |

记录今天的三条结论。

---



## 今日过关标准

- [ ] 能说清 logits 和概率的区别，以及贪心解码为什么不需要 softmax
- [ ] 完成 `LMHead`，含只算最后一位的路径
- [ ] `TinyLM` 能吃 `input_ids` 吐出 next token
- [ ] 对比了 tied / untied 的参数量，并能说出共享为什么合理
- [ ] `pytest -q` 全部通过
- [ ] 更新张量约定文档
- [ ] 能口述 文本 → `input_ids` → `hidden_states` → `logits` → next token

---



## 今日最重要的面试式问题

**生成第 101 个 token 时，LM Head 需要对多少个位置做矩阵乘？**

答案必须是：**1 个**。

只有最后一个位置的 logits 会被用来预测下一个 token，前面 100 个位置的 logits 在推理阶段
计算了也是丢掉。训练时不一样——teacher forcing 下每个位置的 logits 都要参与算 loss。

**推理和训练在这里的行为不同**，这是很多人第一次写推理引擎时白白浪费算力的地方。
