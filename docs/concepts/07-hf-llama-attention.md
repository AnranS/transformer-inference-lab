# HF LlamaAttention ↔ 自研 MultiHeadAttention

> Day 10。对照本地 transformers **5.14.1** 的 `LlamaAttention`，
> 标出已经对齐的部分和后面几周要补的缺口。
>
> 源码：`.venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py`
> 自研：`src/mini_transformer/attention.py`

到 Day 19 做 HF parity 时，用这张表逐项核对，补上一项就划掉一行。

## 1. 两条流水线

```text
自研：X → Wq/Wk/Wv → 拆头 → naive_attention → 合头 → Wo
HF  ：X → Wq/Wk/Wv → 拆头 → RoPE(Q,K) → Cache.update → attention → 合头 → Wo
```

骨架相同。HF 在「拆头」和「算 attention」之间多了 RoPE 与 KV Cache；
算 attention 之前还可能把 KV 头复制到 Query 头数（GQA）。

## 2. 对应表

行号以本地 transformers 5.14.1 为准。

| HF `LlamaAttention` | 行号 | 我的实现 | 差异与原因 |
|---|---|---|---|
| `layer_idx` | 231 | **缺** | Cache 按层分槽。Day 27 接入 Cache 时补 |
| `head_dim` | 232 | `hidden_size // num_heads` | 语义相同。HF 允许 config 单独指定 `head_dim` |
| `num_key_value_heads` / `num_key_value_groups` | 233, 241–246 | **缺**（默认 `Hkv = Hq`） | GQA：Query 头可以多于 KV 头。Day 31 补 |
| `self.scaling = head_dim ** -0.5` | 234 | `/ math.sqrt(head_dim)` | **已对齐**。都是 `1/√Dh`，不是 `1/√D` |
| `q_proj` 输出 `Hq × Dh` | 238–240 | `nn.Linear(D, D)` | MHA 时 `D = Hq × Dh`，两边一样 |
| `k_proj` / `v_proj` 输出 `Hkv × Dh` | 241–246 | `nn.Linear(D, D)` | 自研尚未拆开 KV 宽度；GQA 时 HF 这里更窄。Day 31 |
| `o_proj` | 247–249 | `out_proj` | **已对齐**，只是名字不同。bias 默认都是 False |
| `.view(B,S,-1,Dh).transpose(1,2)` | 260–264 | `_split_heads` | **已对齐**。HF 用 `-1` 让 Q/K 共用一句 view，头数可以不同 |
| `apply_rotary_pos_emb` | 146, 266–267 | **缺** | 投影之后、attention 之前，只转 Q/K，不转 V。Day 14 补 |
| `past_key_values.update` | 269–270 | **缺** | RoPE 之后写入本层 K/V，Decode 时 `Sq=1, Skv>1`。Day 27 补 |
| `ALL_ATTENTION_FUNCTIONS` / `eager_attention_forward` | 199, 272–285 | `naive_attention` | **公式已对齐**：`QKᵀ × scale → +mask → softmax → @V`。HF 还可切 SDPA/Flash |
| `repeat_kv` | 187, 209–210 | **缺** | 把 `[B,Hkv,S,Dh]` 复制成 `[B,Hq,S,Dh]`。MHA 时 `n_rep=1` 等于空操作。Day 31 补 |
| `softmax(..., dim=-1)` 先转 FP32 | 216 | `scaled_scores.softmax(dim=-1)` | 自研当前在输入 dtype 上做 softmax。Day 18 定容差时再核对 |
| `transpose(1,2)` 发生在 eager 内部 | 219 | `_merge_heads` 里 transpose | 出口形状都是 `[B,S,D]`；HF 的 transpose 提前到 `eager_attention_forward` |
| `reshape` + `o_proj` | 287–288 | `_merge_heads` + `out_proj` | **已对齐** |

## 3. 当前缺口（按学习单元）

- [ ] **Day 14**：`apply_rotary_pos_emb`，只作用于 Q/K
- [ ] **Day 18**：softmax 是否升 FP32，以及和 HF 的数值容差
- [ ] **Day 27**：`layer_idx` + `past_key_values.update`
- [ ] **Day 31**：`num_key_value_heads`、`k_proj`/`v_proj` 更窄的输出、`repeat_kv`

没有勾上的项，Day 19 做逐模块对齐时不要假设「已经和 HF 一样」。

## 4. 今天必须记住的两件事

1. RoPE 插在**投影之后、attention 之前**，而且**不转 V**。
2. `repeat_kv` 说明 HF 的 GQA 是「把 KV 头复制到 `Hq` 份」。这不是唯一做法，Day 31 会对照 SDPA 的 `enable_gqa`。

## 5. 整理测试时发现的坑

PyTorch SDPA 的 `is_causal=True` 在 `Sq == Skv` 时和下三角一致，可以当 Prefill 参考。
`Sq=1, Skv=8` 时它**不是**右下角对齐：最新 Query 只看见第 0 个 Key，而不是全部历史。

Decode 必须自己构造右下角 additive mask，再传给 `naive_attention` / `attn_mask`。
门禁测试 `test_week1_gate_naive_attention_matches_sdpa` 覆盖无 mask 的 fp32 对齐，
含 `Sq != Skv`；因果 Decode 对齐见 `tests/test_mask.py`。
