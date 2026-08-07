# 学习单元 Day 20：最终 logits parity 与第 2 周门禁

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 20**｜学习日 14

第 2 周最后一天。今天把 Day 19 的逐层对齐推到**最终 logits**，然后过门禁。

> **第 2 周门禁**：RMSNorm、RoPE、Attention、MLP、Block 和**最终 logits**
> 全部与 HF 对齐。**这是整个路线最硬的一道门禁。**

过了这道门，你手上就是一个**已验证正确**的 Transformer。
第 3 周之后所有的生成、Cache、优化工作，都建立在这个基础上——
如果基础是错的，后面所有「优化后结果一致」的验证都变成了在验证一个错误实现的自洽性。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：最终 logits parity | `tests/test_hf_parity.py` |
| ⬜ | 任务 2：多配置扫描 | 同上（参数化） |
| ⬜ | 任务 3：反向定位练习 | `docs/concepts/07-hf-llama-attention.md` |
| ⬜ | 任务 4：容差表收尾 | `docs/concepts/05-numerical-tolerance.md` |
| ⬜ | 任务 5：第 2 周门禁与段落验收 | 本文末 |
| ⬜ | 任务 6（可选）：极小规模训练一致性 | `notebooks/day20_tiny_training.ipynb` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [torch.testing](https://docs.pytorch.org/docs/stable/testing.html) | 完成端到端 parity，并能从 HF 结构反向定位到自己的实现 |

---

## 任务 1：最终 logits parity（80 分钟）

```python
def test_parity_7_final_logits():
    load_from_hf(my_model, hf_model)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.inference_mode():
        mine = my_model(input_ids)
        ref  = hf_model(input_ids).logits
    torch.testing.assert_close(mine, ref, rtol=..., atol=...)
```

先用**默认 fp32 容差**试。过了最好；不过的话按 Day 18 的规则放宽，
并在注释里写清「涉及 2 层 × (2 GEMM + softmax + 3 GEMM)，实测最大误差 X，取 3 倍余量」。

### 如果这一级不过、但 Day 19 全过了

Day 19 已经验证到 Block 级别，所以问题只能在这三个地方：

| 可能原因 | 怎么确认 |
|---|---|
| **漏了 final norm** | 打印 `my_model` 的模块列表，找 `norm`；或手动对 hidden states 做一次 norm 看是否对上 |
| **LM Head 权重没复制到 / tie 没生效** | `assert lm_head.weight is embedding.weight`，再和 HF 逐比特比 |
| **多层之间的 `position_ids` 或 mask 传递断了** | 只用 `num_hidden_layers=1` 试，过了说明是层间传递问题 |

第三条的「降到 1 层」是个通用的隔离技巧：**用最小配置区分「单层逻辑错」和「层间装配错」。**

### 一个必须做的对照

对齐通过后，**故意破坏一处**，确认测试会变红：

- 把 final norm 注掉 → 必须失败
- 把 `position_ids` 改成全 0 → 必须失败
- 把 `scaling` 从 `Dh**-0.5` 改成 `D**-0.5` → 必须失败

如果哪一项破坏了测试**依然通过**，说明容差放得太松，测试失去了鉴别力。
**这一步比对齐本身更重要**——它验证的是你的测试有没有用。

## 任务 2：多配置扫描（60 分钟）

单一配置对齐可能是运气。用 `@pytest.mark.parametrize` 扫一批：

| 变量 | 取值 | 想暴露什么 |
|---|---|---|
| `num_hidden_layers` | 1, 2, 4 | 层间传递、误差累积 |
| `num_attention_heads` | 1, 2, 4 | head 拆分/合并（`Hq=1` 时很多 bug 隐身） |
| `B, S` | (1,1), (1,8), (3,5) | 边界形状、非整齐 batch |
| `tie_word_embeddings` | True, False | LM Head 权重路径 |
| `head_dim` | 16, 32 | RoPE 频率数量 |

`B=1, S=1` 这个组合特别值得测——它是 **Decode 阶段的形状**，
提前在这里验证过，Day 26～30 能省很多事。

注意组合会爆炸。**不要做全笛卡尔积**，选十来组有代表性的即可，
控制整个测试套件在几秒内跑完。

### 顺手做一次 bf16 观测

不作为门禁，但记录下来：同一配置在 bf16 下，最终 logits 与 fp32 的最大差异是多少。
这个数字在第 4 周决定「用什么精度做 benchmark」时会用到。

## 任务 3：反向定位练习（50 分钟）

第 2 周的要求里有一条能力目标：

> 能从 HF `LlamaDecoderLayer` 反向定位到自己的实现。

具体练法：**打开 HF 源码，随机指一行，说出它对应你代码的哪一行、以及为什么这么写。**

自测五个点：

1. `self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)`
   —— `layer_idx` 是干什么的？（提示：KV Cache 要用它索引，Day 27 会遇到）
2. `hidden_states, _ = self.self_attn(...)` —— 为什么返回 tuple？
3. `apply_rotary_pos_emb(q, k, cos, sin)` —— 为什么参数里没有 `v`？
4. `repeat_kv(key_states, self.num_key_value_groups)` —— 当前 `Hq=Hkv` 时它做了什么？
5. `self.norm = LlamaRMSNorm(...)` 在 `LlamaModel.__init__` 里
   —— 它和每层的两个 norm 是什么关系？

答不上来的回去看源码。**这个能力的价值在于：以后遇到任何新模型
（Qwen、Mistral、DeepSeek），你能直接读它的 HF 实现并对照到自己的知识框架上。**

把 Day 10 建的那张 HF ↔ 自研对应表更新完整，把「缺口」列里
RoPE、RMSNorm、MLP、Block 这几项划掉（都实现了），
剩下应该只有 **`repeat_kv`（GQA，Day 31）** 和 **`past_key_value`（KV Cache，Day 27）**。

## 任务 4：容差表收尾（30 分钟）

把 Day 18 建的表填完，七级全部有数字：

| 级别 | fp32 最大绝对误差 | fp32 最大相对误差 | 采用 rtol/atol | bf16 最大绝对误差 |
|---|---|---|---|---|
| 权重复制 | 0 | 0 | `0/0` | 0 |
| RMSNorm | | | | |
| RoPE | | | | |
| MLP | | | | |
| Attention | | | | |
| Decoder Block | | | | |
| 最终 logits | | | | |

观察误差随级别的增长趋势，在笔记里写一句结论
（比如「从 Block 到最终 logits 误差增长约 N 倍，与 2 层的累积预期一致」）。

**这张表是第 4 周 benchmark 报告的「数值正确性」章节。** 现在填好，到时候直接引用。

## 任务 5：第 2 周门禁与段落验收（30 分钟）

### 门禁

- [ ] `test_parity_1` ～ `test_parity_7` 全部通过
- [ ] 多配置扫描全部通过（含 `B=1, S=1`）
- [ ] 故意破坏三处，测试都能变红
- [ ] 容差表七级填满

### 第 2 周段落验收

- [ ] 不看资料实现 RMSNorm
- [ ] 不看资料实现 SwiGLU
- [ ] 解释 RoPE 为什么只作用于 Q 和 K
- [ ] 画出 Decoder Block 数据流
- [ ] 与 HF logits 数值对齐

前四条是**闭卷**的。找张白纸，真的写一遍，不要在心里默认「我会」。

> **门禁未过不要进第 3 周。** 第 3 周开始做生成和采样，
> 那些工作全部假设模型的 logits 是对的。基础不对，后面的验证都是在验证错误的自洽。

## 任务 6（可选）：极小规模训练一致性（90 分钟）

这项是**选做**。如果前五个任务提前做完，且今天还有精力，值得做——
它能捕捉一类前面所有测试都测不到的问题。

做法：用几十条极短的重复序列（比如 `"a b c a b c"`），
在自研模型和 HF 模型上各跑同样的几百步训练，对比 loss 曲线。

**能发现什么**：如果两条曲线走势明显不同，说明存在**梯度路径**上的差异——
比如某个 `detach()` 用错了、某处 `no_grad` 范围不对。
这类问题在纯前向的 parity 里完全看不出来。

新建 `notebooks/day20_tiny_training.ipynb`。**如果时间不够就跳过**，
在笔记里记一句「已知未验证：梯度路径一致性」即可。
本项目是纯推理项目，这个风险可以接受。

---

## 过关标准

- [ ] 最终 logits 与 HF 对齐
- [ ] 多配置扫描通过，含 Decode 形状 `B=1, S=1`
- [ ] 三处故意破坏都能让测试变红（测试有鉴别力）
- [ ] HF ↔ 自研对应表更新，缺口只剩 `repeat_kv` 和 `past_key_value`
- [ ] 能回答反向定位练习的五个问题
- [ ] 容差表七级填满，并写了误差增长的结论
- [ ] 第 2 周段落验收五条全部通过（前四条闭卷）
- [ ] `pytest -q` 全部通过

---

## 今日最重要的面试式问题

**你的实现和 Hugging Face 完全对齐了，这说明了什么，不说明什么？**

**说明**：所有算子的数学逻辑、shape 变换、权重布局、位置编码、
归一化细节都正确，而且装配顺序正确。这是一个很强的正确性证据。

**不说明**：
- 性能没问题（对齐只测数值，不测速度和显存）
- Decode 路径正确（parity 测的是 Prefill 形态；`B=1,S=1` 只是形状相同，
  还没有 KV Cache 参与）
- 梯度路径正确（纯前向对齐测不到）
- 长序列下没有数值问题（小配置测不出）

追问：**如果只测最终 logits 就对齐了，还需要逐层测吗？**

需要。原因有两个：

1. **定位能力**。哪天改了某个算子导致 parity 失败，只有最终 logits 的话
   要从头二分查找；分层测试直接告诉你是哪一级。
2. **误差掩盖**。最终 logits 的容差必然比单层宽，
   一个单层误差偏大但被 softmax 前的宽容差吸收掉的 bug，
   只测最终 logits 是抓不到的。
