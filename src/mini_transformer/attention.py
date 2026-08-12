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
    """
    if query.dim() != 4:
        raise ValueError(
            f"query must be a [B, H, S, Dh] tensor, "
            f"but got shape {tuple(query.shape)}"
        )
    if key.dim() != 4:
        raise ValueError(
            f"key must be a [B, H, S, Dh] tensor, "
            f"but got shape {tuple(key.shape)}"
        )
    if value.dim() != 4:
        raise ValueError(
            f"value must be a [B, H, S, Dh] tensor, "
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
    """构建因果掩码。
    query_len: 查询序列长度。
    key_value_len: 键值序列长度。
    device: 设备。
    dtype: 数据类型。
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
    attention_mask: [B, Skv]
    dtype: 数据类型。
    """
    expanded_mask = attention_mask[:, None, None, :]
    additive_mask = torch.zeros_like(expanded_mask, dtype=dtype)
    additive_mask.masked_fill_(expanded_mask == 0, torch.finfo(dtype).min)
    return additive_mask


def combine_attention_masks(
    causal_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """把因果掩码和加性掩码结合成一个掩码。
    causal_mask: [Sq, Skv]
    padding_mask: [B, 1, 1, Skv]
    """
    # 两个 mask 的可见位置都是 0，取 minimum 表达“任一 mask 屏蔽即屏蔽”
    return torch.minimum(causal_mask, padding_mask)

class MultiHeadAttention(nn.Module):
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
        pass

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