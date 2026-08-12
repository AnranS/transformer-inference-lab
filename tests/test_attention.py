import math

import pytest
import torch
import torch.nn.functional as F

import mini_transformer.attention as attention
from mini_transformer.attention import naive_attention


def test_naive_attention_matches_sdpa_in_fp32():
    """Sq=4、Skv=6，覆盖 Decode 中 Query 与 Key/Value 长度不同的合法路径。"""
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

    with pytest.raises(ValueError, match=r"\[B, H, Sq, Dh\]"):
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


def test_multi_head_attention_preserves_hidden_state_shape():
    """完整 MHA 必须在内部拆分、计算并合并 head，最终保持 [B, S, D]。"""
    model = attention.MultiHeadAttention(
        hidden_size=12,
        num_heads=3,
    )
    # B=2、S=4、D=12；内部应拆成 H=3、Dh=4。
    hidden_states = torch.randn(2, 4, 12)

    output = model(hidden_states)

    # 输出还要交给残差连接，因此形状必须与输入完全一致。
    assert output.shape == hidden_states.shape

def test_multi_head_attention_rejects_non_3d_input():
    model = attention.MultiHeadAttention(
        hidden_size=12,
        num_heads=3,
    )
    invalid_hidden_states = torch.randn(2, 12)  # 缺少序列维 S
    with pytest.raises(ValueError, match=r"\[B, S, D\]"):
        model(invalid_hidden_states)

def test_multi_head_attention_rejects_wrong_hidden_size():
    """输入的 D 必须与模块初始化时声明的 hidden_size 相同。"""
    model = attention.MultiHeadAttention(
        hidden_size=12,
        num_heads=3,
    )
    # 实际 D=10，与模型要求的 D=12 不一致。
    invalid_hidden_states = torch.randn(2, 4, 10)
    with pytest.raises(ValueError, match="last dimension.*hidden_size"):
        model(invalid_hidden_states)

def test_single_head_matches_naive_attention_with_identity_projections():
    """H=1 且所有投影为单位矩阵时，MHA 应退化为 naive_attention。"""
    torch.manual_seed(0)
    model = attention.MultiHeadAttention(
        hidden_size=4,
        num_heads=1,
    )

    # 单位投影不改变输入，方便隔离并验证 Attention 主路径。
    identity = torch.eye(4)
    with torch.no_grad():
        model.q_proj.weight.copy_(identity)
        model.k_proj.weight.copy_(identity)
        model.v_proj.weight.copy_(identity)
        model.out_proj.weight.copy_(identity)

    hidden_states = torch.randn(2, 3, 4)

    actual = model(hidden_states)

    # naive_attention 需要 [B,H,S,Dh]，单头时只需插入 H=1。
    single_head = hidden_states.unsqueeze(1)
    expected = naive_attention(
        single_head,
        single_head,
        single_head,
    ).squeeze(1)

    torch.testing.assert_close(actual, expected)

def test_multi_head_attention_matches_sdpa_with_multiple_heads():
    """多头完整路径必须与使用相同投影的 PyTorch SDPA 对齐。"""
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 4
    hidden_size = 12
    num_heads = 3
    head_dim = hidden_size // num_heads

    model = attention.MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
    )

    actual = model(hidden_states)

    # 独立构造参考路径，不调用 _split_heads，避免实现与测试犯同一个错误。
    query = model.q_proj(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).transpose(1, 2)
    key = model.k_proj(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).transpose(1, 2)
    value = model.v_proj(hidden_states).reshape(
        batch_size, sequence_length, num_heads, head_dim
    ).transpose(1, 2)

    expected_heads = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )

    # 独立完成 head 合并和 Wo 投影。
    expected_merged = expected_heads.transpose(1, 2).reshape(
        batch_size,
        sequence_length,
        hidden_size,
    )
    expected = model.out_proj(expected_merged)

    torch.testing.assert_close(actual, expected)


def test_multi_head_attention_ignores_changes_to_masked_token():
    """修改 padding mask 屏蔽的 Key/Value，不应影响其他有效 Query。

    MHA 不会自动生成 causal mask；需要因果约束时，调用方应先将 causal
    与 padding mask 合成后传入。
    """
    torch.manual_seed(0)

    model = attention.MultiHeadAttention(
        hidden_size=8,
        num_heads=2,
    )
    hidden_states = torch.randn(1, 4, 8)

    # 最后一个 token 是 PAD，任何 Query 都不能关注它。
    padding_mask = attention.build_padding_mask(
        torch.tensor([[1, 1, 1, 0]]),
        dtype=hidden_states.dtype,
    )

    modified_hidden_states = hidden_states.clone()
    # 大幅修改被屏蔽 token，确保 mask 失效时能产生明显差异。
    modified_hidden_states[:, 3, :] += 1000

    original_output = model(
        hidden_states,
        attn_mask=padding_mask,
    )
    modified_output = model(
        modified_hidden_states,
        attn_mask=padding_mask,
    )

    # 前三个 Query 本身没有变化，而且看不到最后一个 Key/Value。
    torch.testing.assert_close(
        original_output[:, :3, :],
        modified_output[:, :3, :],
    )

def test_multi_head_attention_runs_under_inference_mode():
    """MHA 在推理模式下应正常运行，并且输出不构建计算图。"""
    model = attention.MultiHeadAttention(
        hidden_size=8,
        num_heads=2,
    )
    hidden_states = torch.randn(2, 4, 8)

    with torch.inference_mode():
        output = model(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.requires_grad is False
    assert torch.is_inference(output)

def test_multi_head_attention_matches_hand_calculation():
    """单位投影下，两个 head 的结果必须按原顺序正确合并。"""
    model = attention.MultiHeadAttention(
        hidden_size=4,
        num_heads=2,
    )

    identity = torch.eye(4)
    with torch.no_grad():
        model.q_proj.weight.copy_(identity)
        model.k_proj.weight.copy_(identity)
        model.v_proj.weight.copy_(identity)
        model.out_proj.weight.copy_(identity)

    # H=2、Dh=2，每个 head 都看到一组互相正交的向量。
    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )

    # 每个 head 的缩放 scores 都是 [[1/√2, 0], [0, 1/√2]]。
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

    actual = model(hidden_states)

    torch.testing.assert_close(actual, expected)