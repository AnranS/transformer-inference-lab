# 学习单元 Day 19：逐模块复制 HF 权重与逐层对齐

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 19**｜学习日 13

今天是整个第 2 周技术含量最高的一天，也是**最能证明实现正确**的一天。

有一句话值得反复读：

> 这比「看起来能生成文本」更能证明实现正确。

一个 Attention 写错了 mask 维度的模型，照样能生成通顺的句子——
因为语言模型的冗余度极高。**只有逐层数值对齐能抓住这类错误。**

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：建权重名映射表 | `docs/concepts/12-hf-weight-mapping.md` |
| ⬜ | 任务 2：写权重复制函数 | `tests/test_hf_parity.py` 或 `src/mini_transformer/hf_convert.py` |
| ⬜ | 任务 3：对齐 RMSNorm 与 RoPE | `tests/test_hf_parity.py` |
| ⬜ | 任务 4：对齐 Attention 与 MLP | 同上 |
| ⬜ | 任务 5：对齐单个 Decoder Block | 同上 |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Llama 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [PyTorch Module API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) | 完成逐模块权重复制与中间输出对照 |

---

## 五步法（今天的总纲）

1. 创建一个极小的 HF `LlamaConfig`
2. 实例化随机权重的 `LlamaForCausalLM`
3. 将**相同权重**复制到自研模型
4. 传入**完全相同**的 `input_ids` 和 `position_ids`
5. 比较每层输出和最终 logits

第 1、2 步昨天（Day 18 任务 5）已完成。今天做 3、4、5，做到单个 Block；
最终 logits 留给明天（Day 20）。

**方向很重要：HF → 自研**，不是反过来。HF 是参考实现，它是真值。

---

## 任务 1：建权重名映射表（50 分钟）

先把两边的权重名都打出来：

```python
print([n for n, _ in hf_model.named_parameters()])
print([n for n, _ in my_model.named_parameters()])
```

HF 的名字大致是这个结构（以你本地版本为准）：

```text
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

如果你从 Day 9 起就按 HF 的命名习惯给模块起名（`q_proj`、`gate_proj`、
`input_layernorm`……），**映射基本是去掉 `model.` 前缀就完事**。
这就是当初特意跟着 HF 命名的回报。

写进 `docs/concepts/12-hf-weight-mapping.md`，三列：

| HF 权重名 | 我的权重名 | 是否需要变换 |
|---|---|---|

第三列大部分应该是「直接复制」。如果出现「需要转置」，**先停下来想清楚**——
`nn.Linear` 两边都是 `[out, in]`，正常不该需要转置。需要转置说明你某个
`nn.Linear` 的 `in`/`out` 写反了，那是个真 bug，改代码而不是在复制时转置绕过去。

## 任务 2：写权重复制函数（80 分钟）

```text
load_from_hf(my_model, hf_model) -> None
```

实现要点：

**1. 用 `state_dict` 而不是逐个 `named_parameters` 手抄。**

```python
hf_sd = hf_model.state_dict()
my_sd = my_model.state_dict()
new_sd = {my_name: hf_sd[hf_name] for hf_name, my_name in MAPPING.items()}
my_model.load_state_dict(new_sd, strict=True)
```

**2. `strict=True` 必须开。** 它会在有权重没被覆盖、或多出权重时报错。
关掉它等于放弃这个函数的全部价值——你会得到一个部分随机初始化的模型，
然后在对齐失败时怀疑自己的算子实现。

**3. 处理 weight tying。** `tie_word_embeddings=True` 时
`lm_head.weight` 和 `embed_tokens.weight` 是同一个张量，
`state_dict` 里可能只出现一次。加载后**用 `is` 断言它们仍然共享**：

```python
assert my_model.lm_head.weight is my_model.embedding.weight
```

**4. 复制后立刻断言数值相等。**

```python
torch.testing.assert_close(
    my_model.embedding.weight, hf_model.model.embed_tokens.weight, rtol=0, atol=0
)
```

`rtol=0, atol=0` 表示**逐比特相同**——复制不该有任何误差。
这一条能把「复制错了」和「算子写错了」彻底分开。

> 建议把这个函数放 `src/mini_transformer/hf_convert.py` 而不是测试文件里。
> 它以后加载真实预训练权重时还要用。

## 任务 3：对齐 RMSNorm 与 RoPE（60 分钟）

从最简单的两级开始。**一定要按顺序，不要跳级。**

### RMSNorm

```python
x = torch.randn(2, 8, D)                       # 固定 seed
mine = my_model.layers[0].input_layernorm(x)
ref  = hf_model.model.layers[0].input_layernorm(x)
torch.testing.assert_close(mine, ref)
```

用**默认容差**先试。过不了，检查这三处（Day 11 的内容）：

- `eps` 是否在 `sqrt` 内部
- 中间统计是否转了 fp32
- `weight` 相乘是否在**转回原 dtype 之后**

第三条最隐蔽：fp32 下几乎看不出差别，但顺序不同在 bf16 下会有可见误差。
今天在 fp32 下对齐，明天可以顺手验一下 bf16 的差异有多大。

### RoPE

RoPE 的对齐要小心 HF 的接口形状。大致是：

```python
cos, sin = hf_model.model.rotary_emb(v, position_ids)
q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)
```

对齐时确认三件事：

1. **`inv_freq` 逐比特相同**（`rtol=0, atol=0`）——这是纯函数，不该有误差
2. `cos`/`sin` 的**形状约定**一致（HF 是 `[B, S, Dh]`，前后两半重复）
3. 旋转后的 `q`、`k` 都对齐

如果 `inv_freq` 就对不上，检查 `rope_theta` 和 `head_dim` 是否一致。
如果 `inv_freq` 对但旋转结果不对，**几乎一定是布局问题**——
你写成 interleaved 而 HF 用 `rotate_half`（Day 14 的坑）。

### 记录误差

每对齐一级，把实测的最大绝对/相对误差填进 Day 18 建的那张表。
**不要只记「通过」，要记具体数字。**

## 任务 4：对齐 Attention 与 MLP（80 分钟）

### MLP（先做这个，比 Attention 简单）

```python
mine = my_model.layers[0].mlp(x)
ref  = hf_model.model.layers[0].mlp(x)
```

对不上的话，最可能是 **SiLU 加到了 `up` 上而不是 `gate` 上**（Day 12 任务 6 第 5 条）。

### Attention

这一级最容易出问题，因为参数多。**必须传完全相同的 `position_ids` 和 mask。**

HF 的 `LlamaAttention.forward` 签名随版本变化较大（`position_embeddings`、
`attention_mask`、`past_key_value`、`cache_position` 等），
**先在源码里看清当前版本要什么**，再构造调用。

对不上时的排查顺序（从最可能到最不可能）：

1. **`position_ids` 不一致** —— 打印两边实际用的 position
2. **mask 语义/形状不一致** —— HF 的 mask 是 additive 的 4D 张量，
   注意它 `-inf` 的填法和你的是否一致
3. **`scaling` 用了 `D` 而不是 `Dh`** —— 打印两边的 `scaling` 值直接比
4. **`repeat_kv` 的差异** —— 今天 `Hq == Hkv`，这一项应该无影响；
   如果有影响说明你的实现里 KV head 处理有问题
5. **head 拆分/合并顺序** —— Day 9 的 Bug 2/5

一个有效的隔离手段：**把 mask 传成 `None`**，先在无 mask 的情况下对齐。
过了说明算子本体没问题，问题在 mask 构造上；不过说明算子本身有 bug。
**分而治之，别同时怀疑两件事。**

## 任务 5：对齐单个 Decoder Block（40 分钟）

```python
mine = my_model.layers[0](x, position_ids=pos, attn_mask=mask)
ref  = hf_model.model.layers[0](x, position_ids=pos, attention_mask=mask)[0]
```

注意 HF 的 Block 返回 **tuple**，要取 `[0]`。

到这一级如果前四级都过了、这一级不过，问题一定在**装配**而不是算子：

- 两个 norm 用反了（`input_layernorm` 和 `post_attention_layernorm` 互换）
- 残差加错位置（比如第二条残差加的是 norm 之后的值而不是 norm 之前的）
- 少了一条残差

这正是 Day 15 那条「权重置 0 时输出精确等于输入」的测试要防的东西。
如果那条测试当时通过了，这里出问题的概率就很低。

### 测试组织

`tests/test_hf_parity.py` 用**分级命名**，让失败时一眼看出卡在哪级：

```python
def test_parity_1_weight_copy_is_bitwise_exact(): ...
def test_parity_2_rmsnorm(): ...
def test_parity_3_rope(): ...
def test_parity_4_mlp(): ...
def test_parity_5_attention(): ...
def test_parity_6_decoder_block(): ...
# test_parity_7_final_logits 明天写
```

---

## 过关标准

- [ ] 权重名映射表完成，且**没有任何一项需要转置**
- [ ] `load_from_hf` 用 `strict=True`，复制后逐比特相等
- [ ] weight tying 在复制后仍然成立（`is` 断言）
- [ ] RMSNorm 对齐
- [ ] RoPE 对齐（含 `inv_freq` 逐比特相同）
- [ ] MLP 对齐
- [ ] Attention 对齐
- [ ] 单个 Decoder Block 对齐
- [ ] 每一级的实测误差都记进了容差表
- [ ] `pytest -q` 全部通过

---

## 今日最重要的面试式问题

**怎么证明一个自研 Transformer 实现是正确的？**

**逐层与参考实现做数值对齐**：用同一份权重、同一份输入，
从 RMSNorm 开始逐级比较中间输出，直到最终 logits。

关键是**分级**——只测最终 logits 的话，失败了无法定位；
分级测试会直接指出哪一层开始偏离。

「能生成通顺文本」不算证明。语言模型冗余度很高，
mask 维度错、position 错位这类 bug 照样能生成像样的句子。

追问：**权重复制时为什么必须 `strict=True`？**

`strict=False` 会静默跳过没匹配上的权重，那部分保持**随机初始化**。
你会得到一个「一半是 HF 权重、一半是随机」的模型，
然后在对齐失败时误以为算子写错了。`strict=True` 让映射表的错误立刻暴露。
