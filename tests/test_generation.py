import torch
from torch import nn

from mini_transformer.generate import generate_greedy


class AlwaysTokenModel(nn.Module):
    """无论输入是什么，都让 token_id 成为最后位置最优选择。"""

    def __init__(self, vocab_size: int, token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_id = token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.vocab_size)
        logits[..., self.token_id] = 1.0
        return logits


def test_generates_exactly_max_new_tokens_without_eos():
    model = AlwaysTokenModel(vocab_size=10, token_id=7)
    input_ids = torch.tensor([[1, 2], [3, 4]])

    output_ids = generate_greedy(
        model,
        input_ids,
        max_new_tokens=3,
        eos_token_id=None,
    )

    expected = torch.tensor(
        [
            [1, 2, 7, 7, 7],
            [3, 4, 7, 7, 7],
        ]
    )
    assert torch.equal(output_ids, expected)


def test_stops_immediately_after_generating_eos():
    """当前 baseline 只有整个 batch 同时命中 EOS 才停止。

    Day 21 会加入 per-sequence finished mask，让 batch 内每条序列独立停止。
    """
    model = AlwaysTokenModel(vocab_size=10, token_id=7)
    input_ids = torch.tensor([[1, 2], [3, 4]])

    output_ids = generate_greedy(
        model,
        input_ids,
        max_new_tokens=3,
        eos_token_id=7,
    )

    expected = torch.tensor(
        [
            [1, 2, 7],
            [3, 4, 7],
        ]
    )
    assert torch.equal(output_ids, expected)


def test_generation_is_deterministic_for_fixed_model_and_input():
    model = AlwaysTokenModel(vocab_size=10, token_id=7)
    input_ids = torch.tensor([[1, 2], [3, 4]])

    first_output = generate_greedy(model, input_ids, max_new_tokens=3)
    second_output = generate_greedy(model, input_ids, max_new_tokens=3)

    assert torch.equal(first_output, second_output)


class InferenceModeCheckingModel(nn.Module):
    """记录 forward 执行时是否处于 inference_mode。"""

    def __init__(self):
        super().__init__()
        self.inference_mode_enabled: bool | None = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        batch_size, sequence_length = input_ids.shape
        return torch.zeros(batch_size, sequence_length, 10)


def test_generation_runs_model_under_inference_mode():
    model = InferenceModeCheckingModel()

    generate_greedy(model, torch.tensor([[1, 2]]), max_new_tokens=1)

    assert model.inference_mode_enabled is True