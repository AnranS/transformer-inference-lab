import math

import torch


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """不带 mask 的单头/多头 Attention。

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
    probabilities = scaled_scores.softmax(dim=-1)
    return probabilities @ value