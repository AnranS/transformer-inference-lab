import torch
from torch import nn

from mini_transformer.embedding import TokenEmbedding
from mini_transformer.lm_head import LMHead


class TinyLM(nn.Module):
    """TokenEmbedding + LMHead，能端到端跑通的最小语言模型。

    注意：这里**没有位置信息、也没有 Attention**（RoPE 在 Day 14，Attention 在 Day 6～9），
    所以预测出的 token 没有实际意义。现在验证的是管道通不通，不是输出好不好。

    契约（任务 9 落到 tests/test_lm_head.py）：

      形状   forward(input_ids [B, S]) → logits [B, S, V]
      解码   predict_next_token(input_ids [B, S]) → next_ids [B]，取值落在 [0, V)
      共享   tie_weights() 后 lm_head 与 embedding 的 weight 是同一个张量，
             总参数量随之减少
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, S] → [B, S, V]。"""
        raise NotImplementedError

    def predict_next_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """取最后一个位置的 logits，argmax 得到下一个 token：[B, S] → [B]。"""
        raise NotImplementedError

    def tie_weights(self) -> None:
        """让 lm_head 和 embedding 共享同一个权重张量。

        两者形状本来就都是 [V, D]，可以直接共享，不需要转置。
        """
        raise NotImplementedError
