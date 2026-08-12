# Attention Mask：从语义到实现

## 1. 两个用途、两种表示

Mask 有两个互相独立的用途：

- **Causal mask**：阻止 Query 关注未来的 Key，由自回归任务决定。
- **Padding mask**：阻止 Query 关注 PAD Key，由 batch 中的真实序列长度决定。

它们都可以使用两种表示：

- Boolean：在 PyTorch SDPA 中，`True` 表示可见，`False` 表示屏蔽。
- Additive：可见位置为 `0`，屏蔽位置为一个极小值，在 softmax 前加到 attention scores。

本项目的构造函数统一返回 **additive mask**，避免后续接口混用两套 bool 语义。

## 2. 形状

```text
scores         [B, H, Sq, Skv]
causal mask       [Sq, Skv]
padding mask   [B, 1,  1, Skv]
combined mask  [B, 1, Sq, Skv]
```

- `Sq`：本轮正在计算的 Query 数量。
- `Skv`：可供查询的 Key/Value 数量。
- padding mask 必须显式变成 `[B, 1, 1, Skv]`，不能依赖偶然广播。

## 3. Causal mask 必须右下角对齐

Prefill 时 `Sq == Skv`，普通下三角恰好正确：

```text
0  x  x  x
0  0  x  x
0  0  0  x
0  0  0  0
```

其中 `0` 表示可见，`x` 表示屏蔽值。

使用 KV Cache Decode 时，通常 `Sq=1`、`Skv>1`。最新 Query 可以看见全部历史 Key：

```text
Sq=1, Skv=5
0  0  0  0  0
```

因此不能直接使用 `torch.ones(Sq, Skv).tril()`。正确的绝对位置关系是：

```text
query_position = Skv - Sq + i
key_position   = j
visible        = key_position <= query_position
```

## 4. Task 5 的接口契约

在 `src/mini_transformer/attention.py` 中实现：

```python
def build_causal_mask(
    query_length: int,
    key_value_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """返回 [Sq, Skv] additive causal mask。"""


def build_padding_mask(
    attention_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把 [B, Skv] 的 1/0 mask 转成 [B, 1, 1, Skv] additive mask。"""


def combine_attention_masks(
    causal_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """合成 [B, 1, Sq, Skv] additive mask。"""
```

随后扩展：

```python
def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
```

`attention_bias` 必须在缩放后、softmax 前加入：

```text
QKᵀ → /sqrt(Dh) → + attention_bias → softmax → @V
```

## 5. TDD 顺序

### Red 1：Decode causal mask

先断言 `Sq=1, Skv=5` 返回全零的一行。它能防止错误使用左上角对齐的 `tril(1, 5)`。

### Green 1：按绝对位置构造可见矩阵

分别生成 Query 和 Key 的绝对位置，通过比较得到 boolean 可见矩阵，再转换为 additive mask。

### Red 2：Prefill causal mask

断言 `Sq=Skv=4` 时：

- 对角线及下三角为 `0`；
- 上三角为 `torch.finfo(dtype).min`；
- 输出的 dtype 和 device 与参数一致。

### Red 3：Padding mask

输入：

```text
[[1, 1, 1, 0],
 [1, 1, 0, 0]]
```

断言输出形状为 `[2, 1, 1, 4]`，真实 token 对应 `0`，PAD 对应屏蔽值。

### Red 4：合成与广播

将 `[Sq, Skv]` causal mask 与 `[B, 1, 1, Skv]` padding mask 逐元素取最小值，断言：

```text
combined.shape == [B, 1, Sq, Skv]
(scores + combined).shape == [B, H, Sq, Skv]
```

不能直接把两个 additive mask 相加：同一位置被两者同时屏蔽时，
`finfo.min + finfo.min` 会溢出成 `-inf`。`torch.minimum` 表达的语义是
「任一 mask 屏蔽即屏蔽」，同时保持屏蔽值有限。

### Red 5：接入 Attention

扩展 `naive_attention`，验证：

- 被屏蔽 Key 的 attention 概率为 0；
- causal 输出与 `F.scaled_dot_product_attention(..., is_causal=True)` 在 fp32 下对齐。

## 6. 数值注意事项

不要硬编码 `-1e9`，使用：

```python
torch.finfo(dtype).min
```

但要注意：极小有限值只能让正常行中的屏蔽位置概率接近 0。若一整行都被屏蔽，
softmax 会得到没有语义的分布；使用 `-inf` 时则会得到 NaN。因此当前实现必须保证
每个有效 Query 至少有一个可见 Key，后续处理任意 padding batch 时再显式解决全屏蔽行。

## 7. 完成标准

- Prefill 和 Decode causal mask 都正确。
- padding mask 的形状固定为 `[B, 1, 1, Skv]`。
- causal 与 padding 可以合成并广播到所有 head。
- mask 在 softmax 前生效。
- fp32 输出与 PyTorch SDPA 对齐。
- 全量测试通过。
