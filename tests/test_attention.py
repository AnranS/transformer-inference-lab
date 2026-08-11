import torch
import torch.nn.functional as F
import pytest

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