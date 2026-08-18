"""Naive attention 与 MultiHeadAttention 的契约、数值对齐和第 1 周门禁。"""

import math

import pytest
import torch
import torch.nn.functional as F

import mini_transformer.attention as attention
from mini_transformer.attention import naive_attention

WEEK1_GATE_SHAPES = [
    (2, 3, 4, 4, 8),
    (1, 1, 1, 1, 4),
    (1, 2, 1, 8, 8),
    (2, 1, 4, 6, 8),
    (2, 3, 4, 6, 8),
]

MHA_SHAPES = [
    (2, 4, 3, 4),
    (1, 1, 1, 8),
    (1, 8, 2, 4),
    (2, 1, 4, 4),
    (2, 4, 1, 8),
]


@pytest.mark.parametrize(
    "batch_size, num_heads, query_len, key_value_len, head_dim",
    WEEK1_GATE_SHAPES,
)
def test_week1_gate_naive_attention_matches_sdpa(
    batch_size,
    num_heads,
    query_len,
    key_value_len,
    head_dim,
):
    """第 1 周门禁：naive attention 必须与 PyTorch SDPA 数值对齐（fp32）。"""
    torch.manual_seed(0)
    query = torch.randn(batch_size, num_heads, query_len, head_dim)
    key = torch.randn(batch_size, num_heads, key_value_len, head_dim)
    value = torch.randn(batch_size, num_heads, key_value_len, head_dim)

    actual = naive_attention(query, key, value)
    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "batch_size, num_heads, query_len, key_value_len",
    [
        (2, 3, 4, 4),
        (1, 1, 1, 8),
        (1, 2, 1, 4),
    ],
)
def test_attention_probabilities_sum_to_one(
    batch_size,
    num_heads,
    query_len,
    key_value_len,
):
    torch.manual_seed(0)
    head_dim = key_value_len
    query = torch.randn(batch_size, num_heads, query_len, head_dim)
    key = torch.randn(batch_size, num_heads, key_value_len, head_dim)
    value = (
        torch.eye(key_value_len)
        .expand(batch_size, num_heads, key_value_len, head_dim)
        .clone()
    )

    probabilities = naive_attention(query, key, value)
    expected_sums = torch.ones(batch_size, num_heads, query_len)
    torch.testing.assert_close(probabilities.sum(dim=-1), expected_sums)


def test_rejects_non_4d_inputs():
    query = torch.randn(3, 4, 8)
    key = torch.randn(3, 6, 8)
    value = torch.randn(3, 6, 8)

    with pytest.raises(ValueError, match=r"\[B, H, Sq, Dh\]"):
        naive_attention(query, key, value)


def test_rejects_mismatched_query_key_head_dimensions():
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 7)
    value = torch.randn(2, 3, 6, 8)

    with pytest.raises(ValueError, match="head dimension"):
        naive_attention(query, key, value)


def test_rejects_mismatched_key_value_sequence_lengths():
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 5, 8)

    with pytest.raises(ValueError, match="key and value.*sequence length"):
        naive_attention(query, key, value)


def test_bfloat16_attention_has_measurable_numerical_drift():
    """记录 bf16 相对 fp32 的精度损失；这里的 error 指数值误差，不是异常。"""
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
    with pytest.raises(ValueError, match="hidden_size.*num_heads"):
        attention.MultiHeadAttention(hidden_size=10, num_heads=3)


def test_multi_head_attention_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="hidden_size"):
        attention.MultiHeadAttention(hidden_size=0, num_heads=2)

    with pytest.raises(ValueError, match="num_heads"):
        attention.MultiHeadAttention(hidden_size=8, num_heads=0)


def test_multi_head_attention_has_four_bias_free_projections():
    model = attention.MultiHeadAttention(hidden_size=12, num_heads=3)

    for projection in (model.q_proj, model.k_proj, model.v_proj, model.out_proj):
        assert projection.weight.shape == (12, 12)
        assert projection.bias is None


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_heads, head_dim",
    MHA_SHAPES,
)
def test_split_heads_moves_head_dimension_before_sequence(
    batch_size,
    sequence_length,
    num_heads,
    head_dim,
):
    hidden_size = num_heads * head_dim
    model = attention.MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

    split = model._split_heads(hidden_states)

    assert split.shape == (batch_size, num_heads, sequence_length, head_dim)


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_heads, head_dim",
    MHA_SHAPES,
)
def test_merge_heads_reverses_split_heads(
    batch_size,
    sequence_length,
    num_heads,
    head_dim,
):
    hidden_size = num_heads * head_dim
    model = attention.MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    hidden_states = torch.arange(
        batch_size * sequence_length * hidden_size,
        dtype=torch.float32,
    ).reshape(batch_size, sequence_length, hidden_size)

    merged = model._merge_heads(model._split_heads(hidden_states))

    assert merged.shape == (batch_size, sequence_length, hidden_size)
    assert torch.equal(merged, hidden_states)


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_heads, head_dim",
    MHA_SHAPES,
)
def test_multi_head_attention_preserves_hidden_state_shape(
    batch_size,
    sequence_length,
    num_heads,
    head_dim,
):
    hidden_size = num_heads * head_dim
    model = attention.MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

    output = model(hidden_states)

    assert output.shape == hidden_states.shape


def test_multi_head_attention_rejects_non_3d_input():
    model = attention.MultiHeadAttention(hidden_size=12, num_heads=3)
    invalid_hidden_states = torch.randn(2, 12)

    with pytest.raises(ValueError, match=r"\[B, S, D\]"):
        model(invalid_hidden_states)


def test_multi_head_attention_rejects_wrong_hidden_size():
    model = attention.MultiHeadAttention(hidden_size=12, num_heads=3)
    invalid_hidden_states = torch.randn(2, 4, 10)

    with pytest.raises(ValueError, match="last dimension.*hidden_size"):
        model(invalid_hidden_states)


def test_single_head_matches_naive_attention_with_identity_projections():
    """H=1 且所有投影为单位矩阵时，MHA 应退化为 naive_attention。"""
    torch.manual_seed(0)
    model = attention.MultiHeadAttention(hidden_size=4, num_heads=1)
    identity = torch.eye(4)
    with torch.no_grad():
        model.q_proj.weight.copy_(identity)
        model.k_proj.weight.copy_(identity)
        model.v_proj.weight.copy_(identity)
        model.out_proj.weight.copy_(identity)

    hidden_states = torch.randn(2, 3, 4)
    actual = model(hidden_states)
    single_head = hidden_states.unsqueeze(1)
    expected = naive_attention(single_head, single_head, single_head).squeeze(1)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "batch_size, sequence_length, num_heads, head_dim",
    MHA_SHAPES,
)
def test_multi_head_attention_matches_sdpa_with_multiple_heads(
    batch_size,
    sequence_length,
    num_heads,
    head_dim,
):
    """完整 MHA 路径必须与使用相同投影的 PyTorch SDPA 对齐。"""
    torch.manual_seed(0)
    hidden_size = num_heads * head_dim
    model = attention.MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

    actual = model(hidden_states)

    query = (
        model.q_proj(hidden_states)
        .reshape(batch_size, sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    key = (
        model.k_proj(hidden_states)
        .reshape(batch_size, sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    value = (
        model.v_proj(hidden_states)
        .reshape(batch_size, sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    expected_heads = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )
    expected = model.out_proj(
        expected_heads.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            hidden_size,
        )
    )

    torch.testing.assert_close(actual, expected)


def test_multi_head_attention_ignores_changes_to_masked_token():
    """修改 padding mask 屏蔽的 Key/Value，不应影响其他有效 Query。"""
    torch.manual_seed(0)
    model = attention.MultiHeadAttention(hidden_size=8, num_heads=2)
    hidden_states = torch.randn(1, 4, 8)
    padding_mask = attention.build_padding_mask(
        torch.tensor([[1, 1, 1, 0]]),
        dtype=hidden_states.dtype,
    )
    modified_hidden_states = hidden_states.clone()
    modified_hidden_states[:, 3, :] += 1000

    original_output = model(hidden_states, attn_mask=padding_mask)
    modified_output = model(modified_hidden_states, attn_mask=padding_mask)

    torch.testing.assert_close(
        original_output[:, :3, :],
        modified_output[:, :3, :],
    )


def test_multi_head_attention_runs_under_inference_mode():
    model = attention.MultiHeadAttention(hidden_size=8, num_heads=2)
    hidden_states = torch.randn(2, 4, 8)

    with torch.inference_mode():
        output = model(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.requires_grad is False
    assert torch.is_inference(output)


def test_multi_head_attention_matches_hand_calculation():
    """单位投影下，两个 head 的结果必须按原顺序正确合并。"""
    model = attention.MultiHeadAttention(hidden_size=4, num_heads=2)
    identity = torch.eye(4)
    with torch.no_grad():
        model.q_proj.weight.copy_(identity)
        model.k_proj.weight.copy_(identity)
        model.v_proj.weight.copy_(identity)
        model.out_proj.weight.copy_(identity)

    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    same_token_weight, other_token_weight = torch.softmax(
        torch.tensor([1 / math.sqrt(2), 0.0]),
        dim=0,
    )
    expected = torch.tensor(
        [
            [
                [
                    same_token_weight,
                    other_token_weight,
                    other_token_weight,
                    same_token_weight,
                ],
                [
                    other_token_weight,
                    same_token_weight,
                    same_token_weight,
                    other_token_weight,
                ],
            ]
        ]
    )

    torch.testing.assert_close(model(hidden_states), expected)
