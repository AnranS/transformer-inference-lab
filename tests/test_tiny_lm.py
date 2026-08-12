"""TinyLM 端到端 shape、最后位置预测与 weight tying 契约。"""

import torch

from mini_transformer.tiny_lm import TinyLM


def test_forward_maps_input_ids_to_vocab_logits():
    model = TinyLM(vocab_size=128, hidden_size=16)
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    logits = model(input_ids)

    assert logits.shape == (2, 3, 128)

def test_predict_next_token_uses_last_position_argmax():
    model = TinyLM(vocab_size=128, hidden_size=16)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    next_ids = model.predict_next_token(input_ids)
    expected = model(input_ids)[:, -1, :].argmax(dim=-1)
    assert next_ids.shape == (2,)
    assert torch.equal(next_ids, expected)

def test_predict_next_token_slices_before_lm_head():
    model = TinyLM(vocab_size=128, hidden_size=16)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    seen_shapes = []
    model.lm_head.proj.register_forward_pre_hook(
        lambda _module, args: seen_shapes.append(tuple(args[0].shape))
    )
    model.predict_next_token(input_ids)
    assert seen_shapes == [(2, 16)]

def test_tie_weights_shares_the_same_parameter():
    model = TinyLM(vocab_size=128, hidden_size=16)

    model.tie_weights()

    assert model.lm_head.weight is model.embedding.embedding.weight

def test_tie_weights_removes_duplicate_parameter():
    """共享前有两份 [V,D]，共享后参数遍历只计同一个 [V,D] Parameter。"""
    model = TinyLM(vocab_size=128, hidden_size=16)
    untied_count = sum(parameter.numel() for parameter in model.parameters())
    model.tie_weights()
    tied_count = sum(parameter.numel() for parameter in model.parameters())
    assert untied_count == 2 * 128 * 16
    assert tied_count == 128 * 16

def test_tied_lm_head_observes_embedding_weight_updates():
    model = TinyLM(vocab_size=128, hidden_size=16)
    model.tie_weights()

    with torch.no_grad():
        model.embedding.embedding.weight[7, 3] = 42.0

    assert model.lm_head.weight[7, 3].item() == 42.0