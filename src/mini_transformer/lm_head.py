import torch
from torch import nn


class LMHead(nn.Module):
    """把每个位置的隐藏向量映射回词表上的分数。

    契约（任务 9 把它落成 tests/test_lm_head.py 里的断言）：

      形状   权重 [V, D]；输入 hidden_states [B, S, D] → 输出 logits [B, S, V]
      bias   默认 False，对齐 GPT-2
      构造   hidden_size / vocab_size 不是正数 → ValueError
      输入   不是三维 → ValueError；最后一维 != hidden_size → ValueError
      解码   forward_last_position 只对最后一位做矩阵乘，
             结果与 forward(...)[:, -1, :] 完全一致
    """

    def __init__(self, hidden_size: int, vocab_size: int, *, bias: bool = False):
        super().__init__()
        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """全量路径：[B, S, D] → [B, S, V]。"""
        raise NotImplementedError

    def forward_last_position(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """解码路径：[B, S, D] → [B, V]，只对最后一个位置做矩阵乘。

        切片必须发生在矩阵乘**之前**，否则省不下任何计算。
        """
        raise NotImplementedError
