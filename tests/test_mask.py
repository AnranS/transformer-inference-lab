"""Attention mask 的 Prefill、Decode、Padding 与合成契约。

重点验证 causal mask 在 Prefill 时是下三角，而在单步 Decode 时必须
右下角对齐，让最新 Query 看见全部缓存 Key。
"""

import torch
import torch.nn.functional as F

from mini_transformer.attention import (
    build_causal_mask,
    build_padding_mask,
    combine_attention_masks,
    naive_attention,
)


def test_decode_causal_mask_allows_all_cached_keys():
    """Sq=1、Skv=5 模拟单步 Decode：最新 Query 可以读取全部缓存 Key。"""
    mask = build_causal_mask(
        query_len=1,
        key_value_len=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.zeros((1, 5), dtype=torch.float32)
    torch.testing.assert_close(mask, expected)


def test_prefill_causal_mask_blocks_future_keys():
    """Sq=Skv=4 模拟 Prefill：每个 Query 只能看到自己和更早的 Key。"""
    dtype = torch.float32
    blocked = torch.finfo(dtype).min
    expected = torch.tensor(
        [
            [0.0, blocked, blocked, blocked],
            [0.0, 0.0, blocked, blocked],
            [0.0, 0.0, 0.0, blocked],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    mask = build_causal_mask(
        query_len=4,
        key_value_len=4,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    torch.testing.assert_close(mask, expected)


def test_causal_mask_uses_requested_dtype():
    mask = build_causal_mask(
        query_len=2,
        key_value_len=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert mask.dtype == torch.bfloat16


def test_padding_mask_expands_and_blocks_pad_keys():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    dtype = torch.float32
    blocked = torch.finfo(dtype).min
    expected = torch.tensor(
        [
            [[[0.0, 0.0, 0.0, blocked]]],
            [[[0.0, 0.0, blocked, blocked]]],
        ],
        dtype=dtype,
    )

    actual = build_padding_mask(attention_mask, dtype=dtype)

    torch.testing.assert_close(actual, expected)


def test_combined_mask_broadcasts_without_overflow():
    """minimum 合成保持有限值，避免两个 finfo.min 相加溢出成 -inf。"""
    dtype = torch.float32
    causal_mask = build_causal_mask(
        query_len=4,
        key_value_len=4,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    padding_mask = build_padding_mask(
        torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ]
        ),
        dtype=dtype,
    )

    combined = combine_attention_masks(causal_mask, padding_mask)

    assert combined.shape == (2, 1, 4, 4)
    assert torch.isfinite(combined).all()

    blocked = torch.finfo(dtype).min
    assert combined[0, 0, 3, 3] == 0
    assert combined[1, 0, 3, 2] == blocked
    assert combined[1, 0, 0, 3] == blocked


def test_attention_bias_zeroes_blocked_key_probabilities():
    query = torch.zeros((1, 1, 1, 4))
    key = torch.zeros((1, 1, 4, 4))

    # 用单位矩阵作为 V，Attention 输出就等于概率向量，方便直接观察 mask 效果。
    value = torch.eye(4).view(1, 1, 4, 4)
    padding_mask = build_padding_mask(
        torch.tensor([[1, 1, 0, 0]]),
        dtype=torch.float32,
    )

    probabilities = naive_attention(
        query,
        key,
        value,
        attention_bias=padding_mask,
    )

    expected = torch.tensor([[[[0.5, 0.5, 0.0, 0.0]]]])
    torch.testing.assert_close(probabilities, expected)


def test_causal_attention_matches_sdpa():
    torch.manual_seed(0)
    query = torch.randn((2, 3, 4, 8))
    key = torch.randn((2, 3, 4, 8))
    value = torch.randn((2, 3, 4, 8))
    causal_mask = build_causal_mask(
        query_len=4,
        key_value_len=4,
        device=query.device,
        dtype=query.dtype,
    )

    actual = naive_attention(
        query,
        key,
        value,
        attention_bias=causal_mask,
    )
    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )

    torch.testing.assert_close(actual, expected)


def test_first_causal_attention_row_only_sees_first_key():
    torch.manual_seed(0)
    query = torch.randn((1, 1, 4, 4))
    key = torch.randn((1, 1, 4, 4))

    # 单位矩阵让输出直接暴露每个 Key 的 Attention 概率。
    value = torch.eye(4).view(1, 1, 4, 4)
    causal_mask = build_causal_mask(
        query_len=4,
        key_value_len=4,
        device=query.device,
        dtype=query.dtype,
    )

    probabilities = naive_attention(
        query,
        key,
        value,
        attention_bias=causal_mask,
    )

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(probabilities[0, 0, 0], expected)
