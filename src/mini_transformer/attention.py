import math

import torch
from torch import nn


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """支持 additive attention bias 的单头/多头 Attention。

    query: [B, H, Sq, Dh]
    key:   [B, H, Skv, Dh]
    value: [B, H, Skv, Dh]
    output:[B, H, Sq, Dh]
    attention_bias: 可广播到 [B, H, Sq, Skv] 的 additive mask；
                    0 表示可见，极小值表示屏蔽。
    """
    if query.dim() != 4:
        raise ValueError(
            f"query must be a [B, H, Sq, Dh] tensor, "
            f"but got shape {tuple(query.shape)}"
        )
    if key.dim() != 4:
        raise ValueError(
            f"key must be a [B, H, Skv, Dh] tensor, "
            f"but got shape {tuple(key.shape)}"
        )
    if value.dim() != 4:
        raise ValueError(
            f"value must be a [B, H, Skv, Dh] tensor, "
            f"but got shape {tuple(value.shape)}"
        )
    if query.size(-1) != key.size(-1):
        raise ValueError(
            f"query and key must have the same head dimension, "
            f"but got {query.size(-1)} and {key.size(-1)}"
        )
    if key.size(-2) != value.size(-2):
        raise ValueError(
            f"key and value must have the same sequence length, "
            f"but got {key.size(-2)} and {value.size(-2)}"
        )

    scores = query @ key.transpose(-2, -1)
    head_dim = query.size(-1)
    scaled_scores = scores / math.sqrt(head_dim)
    if attention_bias is not None:
        scaled_scores += attention_bias
    probabilities = scaled_scores.softmax(dim=-1)
    return probabilities @ value


def build_causal_mask(
    query_len: int,
    key_value_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构建右下角对齐的 [Sq, Skv] additive causal mask。

    可见位置为 0，未来位置为 torch.finfo(dtype).min。
    Query 对应最后 Sq 个绝对位置，所以 Decode 的 Sq=1、Skv>1
    会返回全 0 的一行，让最新 Query 看见全部历史 Key。
    """
    mask = torch.zeros(query_len, key_value_len, device=device, dtype=dtype)
    query_position = torch.arange(key_value_len - query_len, key_value_len, device=device)
    key_position = torch.arange(key_value_len, device=device)
    is_available = key_position[None, :] <= query_position[:, None]
    mask.masked_fill_(~is_available, torch.finfo(dtype).min)
    return mask


def build_padding_mask(
    attention_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把 [B, Skv] 的 1/0 mask 转成 [B, 1, 1, Skv] additive mask。

    attention_mask 中 1 表示真实 token，0 表示 PAD。输出中真实 token
    对应 0，PAD 对应 torch.finfo(dtype).min。
    """
    expanded_mask = attention_mask[:, None, None, :]
    additive_mask = torch.zeros_like(expanded_mask, dtype=dtype)
    additive_mask.masked_fill_(expanded_mask == 0, torch.finfo(dtype).min)
    return additive_mask


def combine_attention_masks(
    causal_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """合成 causal [Sq, Skv] 与 padding [B, 1, 1, Skv] mask。

    返回可广播到 [B, H, Sq, Skv] 的 [B, 1, Sq, Skv] additive mask。
    使用 minimum 表达“任一来源屏蔽即屏蔽”，并避免两个 finfo.min
    相加溢出成 -inf。
    """
    return torch.minimum(causal_mask, padding_mask)


class MultiHeadAttention(nn.Module):
    """最小多头自注意力模块：[B, S, D] → [B, S, D]。

    Q/K/V 都由同一份 hidden_states 投影得到；四个投影分别是 Wq、Wk、
    Wv 和 Wo。模块不会自动创建 causal/padding mask，调用方应先合成
    additive mask，再通过 attn_mask 传入。
    """

    def __init__(self, hidden_size: int, num_heads: int, *, bias: bool = False):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, "
                f"but got {hidden_size}"
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, "
                f"but got {num_heads}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算多头自注意力。

        attn_mask 应可广播到 attention scores [B, H, S, S]；
        常用形状为 [S, S] 或 [B, 1, S, S]。
        """
        # 提前验证输入契约，避免让投影层或 shape 解包抛出难以定位的底层错误。
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be a [B, S, D] tensor, "
                f"but got shape {tuple(hidden_states.shape)}"
            )
        # D 必须与初始化时的 hidden_size 一致，四个线性投影才能使用同一特征宽度。
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must match "
                f"hidden_size={self.hidden_size}, "
                f"but got {hidden_states.size(-1)}"
            )
        # 先做 Q/K/V 投影，再把 D 拆成 H 个 Dh，得到 [B, H, S, Dh]。
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        # 每个 head 独立计算 Attention；attn_mask 在 softmax 前屏蔽不可见位置。
        attention_output = naive_attention(query, key, value, attn_mask)

        # 将多个 head 按原顺序合并回 [B, S, D]，再用 Wo 混合各头的信息。
        merged_output = self._merge_heads(attention_output)
        return self.out_proj(merged_output)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, S, D] → [B, H, S, Dh]。"""
        b, s, _ = hidden_states.shape
        split = hidden_states.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        return split

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, H, S, Dh] → [B, S, D]。"""
        b, _, s, _ = hidden_states.shape
        merged = hidden_states.transpose(1, 2).reshape(b, s, self.hidden_size)
        return merged