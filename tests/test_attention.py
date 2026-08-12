import torch
import torch.nn.functional as F
import pytest

import mini_transformer.attention as attention
from mini_transformer.attention import naive_attention


def test_naive_attention_matches_sdpa_in_fp32():
    torch.manual_seed(0)

    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 6, 8)

    actual = naive_attention(query, key, value)
    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )

    torch.testing.assert_close(actual, expected)


def test_attention_probabilities_sum_to_one():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 4)
    key = torch.randn(2, 3, 4, 4)
    # 每个 batch、每个 head 都使用 4×4 单位矩阵
    value = torch.eye(4).expand(2, 3, 4, 4).clone()
    probabilities = naive_attention(query, key, value)
    expected_sums = torch.ones(2, 3, 4)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        expected_sums,
    )

def test_rejects_non_4d_inputs():
    query = torch.randn(3, 4, 8)  # 少了 batch 维
    key = torch.randn(3, 6, 8)
    value = torch.randn(3, 6, 8)

    with pytest.raises(ValueError, match=r"\[B, H, S, Dh\]"):
        naive_attention(query, key, value)


def test_rejects_mismatched_query_key_head_dimensions():
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 7)    # Dh=7，与 query 的 8 不同
    value = torch.randn(2, 3, 6, 8)
    with pytest.raises(ValueError, match="head dimension"):
        naive_attention(query, key, value)

def test_rejects_mismatched_key_value_sequence_lengths():
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)    # Skv=6
    value = torch.randn(2, 3, 5, 8)  # Skv=5，错误
    with pytest.raises(ValueError, match="key and value.*sequence length"):
        naive_attention(query, key, value)

def test_records_bfloat16_attention_error():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 6, 8)
    output_fp32 = naive_attention(query, key, value)
    output_bf16 = naive_attention(
        query.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
    ).float()
    max_absolute_error = (output_fp32 - output_bf16).abs().max().item()
    print(f"fp32 vs bf16 最大绝对误差: {max_absolute_error:.6f}")
    assert torch.isfinite(output_bf16).all()
    assert max_absolute_error > 0


def test_multi_head_attention_rejects_indivisible_hidden_size():
    assert hasattr(attention, "MultiHeadAttention")

    with pytest.raises(ValueError, match="hidden_size.*num_heads"):
        attention.MultiHeadAttention(hidden_size=10, num_heads=3)


def test_multi_head_attention_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="hidden_size"):
        attention.MultiHeadAttention(hidden_size=0, num_heads=2)

    with pytest.raises(ValueError, match="num_heads"):
        attention.MultiHeadAttention(hidden_size=8, num_heads=0)


def test_multi_head_attention_has_four_bias_free_projections():
    model = attention.MultiHeadAttention(
        hidden_size=12,
        num_heads=3,
    )

    for projection in (
        model.q_proj,
        model.k_proj,
        model.v_proj,
        model.out_proj,
    ):
        assert projection.weight.shape == (12, 12)
        assert projection.bias is None


def test_split_heads_moves_head_dimension_before_sequence():
    model = attention.MultiHeadAttention(
        hidden_size=12,
        num_heads=3,
    )
    hidden_states = torch.randn(2, 4, 12)

    split = model._split_heads(hidden_states)

    assert split.shape == (2, 3, 4, 4)


def test_merge_heads_reverses_split_heads():
    model = attention.MultiHeadAttention(
        hidden_size=12,
        num_heads=3,
    )
    hidden_states = torch.arange(2 * 4 * 12).reshape(2, 4, 12)

    split = model._split_heads(hidden_states)
    merged = model._merge_heads(split)

    assert merged.shape == (2, 4, 12)
    assert torch.equal(merged, hidden_states)