import torch
import pytest

from mini_transformer.embedding import TokenEmbedding


def test_weight_shape_is_vocab_by_hidden():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)

    assert model.embedding.weight.shape == (128, 16)


def test_rejects_non_positive_vocab_size():
    with pytest.raises(ValueError, match="vocab_size"):
        TokenEmbedding(vocab_size=0, hidden_size=16)


def test_rejects_non_positive_hidden_size():
    with pytest.raises(ValueError, match="hidden_size"):
        TokenEmbedding(vocab_size=128, hidden_size=0)


def test_embedding_shape():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    output = model(token_ids)

    assert output.shape == (2, 4, 16)


def test_same_token_has_same_embedding():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor(
        [
            [7, 1],
            [7, 2],
        ]
    )

    output = model(token_ids)

    torch.testing.assert_close(output[0, 0], output[1, 0])


def test_forward_with_inference_mode():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor([[1, 2, 3]])

    with torch.inference_mode():
        output = model(token_ids)

    assert output.shape == (1, 3, 16)
    assert output.requires_grad is False


def test_rejects_non_2d_input():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError, match=r"\[B, S\]"):
        model(token_ids)


def test_rejects_non_long_dtype():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    with pytest.raises(TypeError, match="torch.long"):
        model(token_ids)


def test_rejects_out_of_vocab_token_id():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor([[1, 127, 128]])

    with pytest.raises(IndexError):
        model(token_ids)


def test_rejects_negative_token_id():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.tensor([[0, -1]])

    with pytest.raises(IndexError):
        model(token_ids)


def test_accepts_empty_batch():
    model = TokenEmbedding(vocab_size=128, hidden_size=16)
    token_ids = torch.empty((0, 4), dtype=torch.long)

    output = model(token_ids)

    assert output.shape == (0, 4, 16)


def test_range_check_can_be_disabled():
    model = TokenEmbedding(vocab_size=128, hidden_size=16, check_token_range=False)
    token_ids = torch.tensor([[1, 2, 3]])

    output = model(token_ids)

    assert output.shape == (1, 3, 16)
