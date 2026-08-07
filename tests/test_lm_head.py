import re

import torch
import pytest

from mini_transformer.lm_head import LMHead

D, V = 16, 128


def test_weight_shape_is_vocab_by_hidden():
    head = LMHead(hidden_size=D, vocab_size=V)

    # 和 TokenEmbedding 的权重形状一致，tie_weights() 才能直接共享、不用转置
    assert head.weight.shape == (V, D)


def test_bias_defaults_to_false():
    assert LMHead(hidden_size=D, vocab_size=V).proj.bias is None
    assert LMHead(hidden_size=D, vocab_size=V, bias=True).proj.bias is not None


def test_rejects_non_positive_hidden_size():
    with pytest.raises(ValueError, match="hidden_size"):
        LMHead(hidden_size=0, vocab_size=V)


def test_rejects_non_positive_vocab_size():
    with pytest.raises(ValueError, match="vocab_size"):
        LMHead(hidden_size=D, vocab_size=0)


def test_forward_shape():
    head = LMHead(hidden_size=D, vocab_size=V)

    assert head(torch.randn(2, 5, D)).shape == (2, 5, V)


def test_last_position_shape():
    head = LMHead(hidden_size=D, vocab_size=V)

    # 解码路径把 S 维吃掉：[B, S, D] → [B, V]
    assert head.forward_last_position(torch.randn(2, 5, D)).shape == (2, V)


@pytest.mark.parametrize("shape", [(2, 5, D), (1, 1, D), (4, 128, D)])
def test_two_paths_agree(shape):
    head = LMHead(hidden_size=shape[-1], vocab_size=V)
    hidden_states = torch.randn(*shape)

    # 用 assert_close 而不是 torch.equal：两种输入形状走不同的 BLAS 归约顺序，
    # 浮点加法不满足结合律，逐比特相等会偶发失败
    with torch.no_grad():
        torch.testing.assert_close(
            head.forward_last_position(hidden_states), head(hidden_states)[:, -1, :]
        )


def test_single_position_is_consistent():
    head = LMHead(hidden_size=D, vocab_size=V)
    hidden_states = torch.randn(3, 1, D)

    # S=1 时两条路径的语义必须重合，这是解码第一步的情形
    with torch.no_grad():
        torch.testing.assert_close(
            head.forward_last_position(hidden_states), head(hidden_states).squeeze(1)
        )


@pytest.mark.parametrize("bad_shape", [(5, D), (2, 3, 4, D)])
@pytest.mark.parametrize("path", ["forward", "forward_last_position"])
def test_rejects_non_3d_input(bad_shape, path):
    head = LMHead(hidden_size=D, vocab_size=V)
    hidden_states = torch.randn(*bad_shape)

    # 报错信息要带上实际形状，否则和 torch 的兜底没区别。
    # match 走 re.search，元组的圆括号在正则里是分组符号，得先转义
    with pytest.raises(ValueError, match=re.escape(str(bad_shape))):
        getattr(head, path)(hidden_states)


@pytest.mark.parametrize("path", ["forward", "forward_last_position"])
def test_rejects_wrong_hidden_size(path):
    head = LMHead(hidden_size=D, vocab_size=V)
    hidden_states = torch.randn(2, 5, D + 1)

    with pytest.raises(ValueError, match=f"must be {D}"):
        getattr(head, path)(hidden_states)


def test_rejects_empty_sequence_when_decoding():
    head = LMHead(hidden_size=D, vocab_size=V)
    hidden_states = torch.randn(2, 0, D)

    # S=0 没有「最后一个位置」可取，切片会得到空张量而不是报错，所以要显式拦
    with pytest.raises(ValueError, match="at least one position"):
        head.forward_last_position(hidden_states)


def test_slicing_happens_before_the_matmul():
    """解码路径省下的计算，只体现在喂给矩阵乘的张量形状上。

    `proj(x[:, -1, :])` 和 `proj(x)[:, -1, :]` 数值完全等价，
    任何数值断言都区分不了两者，但后者一分计算都没省。
    这里用 pre-hook 拦下真正到达 nn.Linear 的输入，把「省了」变成可断言的事实。
    """
    head = LMHead(hidden_size=D, vocab_size=V)
    seen_shapes = []
    head.proj.register_forward_pre_hook(
        lambda _module, args: seen_shapes.append(tuple(args[0].shape))
    )

    head.forward_last_position(torch.randn(2, 5, D))

    # 到达矩阵乘的必须是 [B, D]，而不是带着 5 个位置的 [B, 5, D]
    assert seen_shapes == [(2, D)]


def test_accepts_empty_batch():
    # B=0 是合法的（对齐 embedding.py 允许空 batch），只有 S=0 才拦
    head = LMHead(hidden_size=D, vocab_size=V)

    assert head.forward_last_position(torch.randn(0, 5, D)).shape == (0, V)


def test_runs_under_inference_mode():
    head = LMHead(hidden_size=D, vocab_size=V)

    with torch.inference_mode():
        logits = head.forward_last_position(torch.randn(2, 5, D))

    assert logits.requires_grad is False
