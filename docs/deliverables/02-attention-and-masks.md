# Attention 与 Mask：第 1 周交付物

> 覆盖 Day 06～10：Q/K/V、Scaled Dot-Product Attention、causal / padding mask、
> Multi-Head Attention，以及与 Hugging Face `LlamaAttention` 的对照。
>
> 实现：`src/mini_transformer/attention.py`  
> 测试：`tests/test_attention.py`、`tests/test_mask.py`  
> 对照笔记：`docs/concepts/07-hf-llama-attention.md`

## 1. Attention 公式与每一步的形状

```text
Attention(Q, K, V) = softmax(QKᵀ / √Dh + mask) V
```

多头完整路径：

```text
hidden_states          [B, S, D]
  → Wq / Wk / Wv       [B, S, D]
  → view + transpose   Q/K/V [B, H, S, Dh]
  → Q @ Kᵀ             scores [B, H, Sq, Skv]
  → / √Dh
  → + additive mask
  → softmax(dim=-1)    probs  [B, H, Sq, Skv]
  → probs @ V          [B, H, Sq, Dh]
  → transpose + reshape [B, S, D]
  → Wo                 [B, S, D]
```

`D = Hq × Dh`。Prefill 时 `Sq = Skv = S`，所以中间分数是 `S × S`。
Decode 有 Cache 时 `Sq = 1`，中间分数是 `1 × Skv`。

## 2. Q / K / V 的语义

三者都由同一份 hidden states 线性投影得到，但角色不同：

| 角色 | 含义 |
|---|---|
| Q | 当前 token 想找什么 |
| K | 每个 token 可被怎样匹配 |
| V | 匹配后真正聚合走的信息 |

`QKᵀ` 只决定「看谁、看多少」；`probs @ V` 才把内容取回来。
K 和 V 不能共用一个向量：匹配特征和传递内容是两件事。

Decoder-only 里还有因果约束：位置 `i` 只能读取 `0..i`，不能看未来。

## 3. 为什么除 `√Dh`

设 `q`、`k` 各维独立、均值 0、方差 1。点积是 `Dh` 项之和：

```text
Var(q · k) = Dh
Std(q · k) = √Dh
```

`Dh` 越大，未缩放分数越极端，softmax 越尖，梯度越容易消失。
除以 `√Dh` 后方差回到约 1，与 head 维度解耦。

必须除 `√Dh`，不是 `√D`。点积发生在一个 head 内部，长度是 `Dh = D / H`。
误用 `√D` 会把权重压得过平。`Hq=1` 时 `D == Dh`，这个 bug 测不出来，必须用 `Hq>1` 对齐 SDPA。

## 4. `[B,S,D]` ↔ `[B,Hq,S,Dh]`

```text
拆： [B, S, D] → view(B, S, H, Dh) → transpose(1, 2) → [B, H, S, Dh]
合： [B, H, S, Dh] → transpose(1, 2) → reshape(B, S, D) → [B, S, D]
```

transpose 是为了让 Attention 的矩阵乘落在 `(S, Dh)` 上，把 `B` 和 `H` 都当成批维。
不转的话会在错误的维度上相乘。

合并必须先 transpose 回来再 `reshape`。`transpose` 后张量通常不连续，
`view` 会报错；`reshape` 会在需要时复制。跳过 transpose 直接 `reshape(B, S, D)`
元素个数对得上，**不会报错，只是把头和位置搅乱**。

## 5. 四种 mask 的组合

用途和表示互相独立：

| | Boolean（SDPA：`True` = 可见） | Additive（本项目） |
|---|---|---|
| **Causal** | 下三角 / 右下角为 True | 可见为 0，未来为 `finfo.min` |
| **Padding** | 真实 token 为 True | 真实 token 为 0，PAD 为 `finfo.min` |

本项目统一用 additive，避免两套 bool 语义混用。形状：

```text
causal    [Sq, Skv]
padding   [B, 1, 1, Skv]
combined  [B, 1, Sq, Skv]   ← minimum，避免两个 finfo.min 相加溢出
```

合成后可广播到 scores `[B, H, Sq, Skv]`。任一来源屏蔽即屏蔽。

## 6. `Sq != Skv` 时必须右下角对齐

Prefill（`Sq = Skv`）的下三角碰巧正确。Decode（`Sq=1, Skv>1`）时，
最新 Query 对应绝对位置 `Skv-1`，必须看见全部历史 Key：

```text
可见条件：key_pos[j] <= query_pos[i]
query_pos[i] = Skv - Sq + i
```

`tril(1, 8)` 会让唯一的 Query 只看见第 0 个 Key。这是 KV Cache 阶段最经典的 bug。

实测：PyTorch SDPA 的 `is_causal=True` 在 `Sq == Skv` 时与下三角一致；
在 `Sq=1, Skv=8` 时**不是**右下角对齐，只看见第 0 个 Key。
Decode 必须自己构造掩码再传入，不能依赖 `is_causal`。

## 7. 五个常见 bug 与检测方法

| Bug | 检测方法 |
|---|---|
| Softmax 维度写错 | `probs.sum(dim=-1)` 必须全为 1 |
| transpose 后直接 reshape 合并 | 恒等投影 + 固定输入手算；`Hq=1` 时这个 bug 隐身 |
| mask 广播维度错误 | 因果 attention 第 0 行必为 `[1, 0, 0, ...]` |
| 缩放用了 `D` 而不是 `Dh` | `Hq>1` 时与 SDPA 对齐（SDPA 默认 `1/√query.size(-1)`） |
| 拆头时 `H` 与 `Dh` 写反 | 同上，恒等投影手算 |

验证「屏蔽生效」最可靠的方法不是看概率，而是改被屏蔽的输入、确认输出不变。

## 8. 与 HF `LlamaAttention` 的对应和缺口

对照本地 transformers **5.14.1**。完整行号表见
[`docs/concepts/07-hf-llama-attention.md`](../concepts/07-hf-llama-attention.md)。

骨架相同：投影 → 拆头 → Attention → 合头 → `Wo`。
HF 在拆头之后多了 RoPE 和 KV Cache，Attention 前可能 `repeat_kv`。

| 已对齐 | 仍缺（按学习单元补） |
|---|---|
| `q/k/v/o` 投影、拆头、`1/√Dh`、`naive_attention` 公式、合头 | Day 14 RoPE（只转 Q/K） |
| | Day 18 softmax 是否升 FP32 |
| | Day 27 `layer_idx` + Cache.update |
| | Day 31 GQA：`Hkv < Hq`、`repeat_kv` |

## 9. 数值对齐结果

测量环境：PyTorch 2.13.0，CPU，`torch.manual_seed(0)`。
门禁测试：`test_week1_gate_naive_attention_matches_sdpa`（含 `Sq=1, Skv=8`）。

| 对照 | 最大绝对误差 | 最大相对误差 | 说明 |
|---|---|---|---|
| `naive_attention` vs SDPA，fp32，无 mask | **2.38×10⁻⁷** | 6.26×10⁻⁵ | 多组 `(B,H,Sq,Skv,Dh)`，含 Decode |
| `naive_attention` + causal vs SDPA `is_causal`，fp32，Prefill | **2.38×10⁻⁷** | 6.26×10⁻⁵ | 仅 `Sq == Skv` |
| `MultiHeadAttention` vs 同投影 SDPA，fp32 | **7.45×10⁻⁸** | 2.22×10⁻⁵ | 含 `Hq>1` 与 `Hq=1` |
| `naive_attention` bf16 vs 同输入 fp32 | **7.33×10⁻³** | 2.11×10⁻¹ | 精度损失，不是实现错误 |
| `naive_attention` bf16 vs SDPA bf16 | **7.81×10⁻³** | 9.42×10⁻² | 两条 bf16 路径彼此也不比特级一致 |

第 4 周做 HF parity 时，fp32 可以用 `atol=1e-5` / `rtol=1.3e-6`（`assert_close` 默认）过门禁；
bf16 需要把容差放到 **10⁻²** 量级，并且不要拿 bf16 去对 fp32 参考。

## 第 1 周门禁

- [x] `naive_attention` 与 SDPA 在 fp32 下对齐
- [x] `MultiHeadAttention`（`Hq>1`）与 SDPA 对齐
- [x] causal mask：Prefill 下三角 + Decode `Sq=1, Skv>1` 右下角对齐
- [x] padding mask 把 PAD 概率压到 0
- [x] causal + padding 合成后可广播到 `[B, H, Sq, Skv]`
- [x] 因果 attention 第 0 行概率为 `[1, 0, 0, ...]`
- [x] `pytest -q`：96 passed
