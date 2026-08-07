import torch
from torch import nn


class LMHead(nn.Module):
    """把每个位置的隐藏向量映射回词表上的分数。

    契约由 tests/test_lm_head.py 定义，跑 `pytest -q` 检验：

      形状   权重 [V, D]；输入 hidden_states [B, S, D] → 输出 logits [B, S, V]
      bias   默认 False，对齐 GPT-2
      构造   hidden_size / vocab_size 不是正数 → ValueError
      输入   不是三维 → ValueError；最后一维 != hidden_size → ValueError
      解码   forward_last_position 只对最后一位做矩阵乘，
             结果与 forward(...)[:, -1, :] 在浮点容差内一致（不是逐比特相等，
             两种形状走不同的 BLAS 归约顺序，GPT-2 尺度实测差到 ~3e-6）
      省算   到达矩阵乘的张量必须是 [B, D]，用 pre-hook 断言，
             因为把切片写在乘之后数值完全等价、只有形状能揭穿

    省下多少时间在 notebooks/day03_logits_and_softmax.ipynb 第 6 节测。
    """

    def __init__(self, hidden_size: int, vocab_size: int, *, bias: bool = False):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, but got {hidden_size}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, but got {vocab_size}")
        self.proj = nn.Linear(hidden_size, vocab_size, bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        """[V, D]，和 TokenEmbedding 的权重形状相同，供 tie_weights() 共享。"""
        return self.proj.weight

    def _validate(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be a 3D tensor [B, S, D], "
                f"but got shape {tuple(hidden_states.shape)}"
            )
        hidden_size = self.proj.in_features
        if hidden_states.size(-1) != hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {hidden_size}, "
                f"but got shape {tuple(hidden_states.shape)}"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """全量路径：[B, S, D] → [B, S, V]。"""
        self._validate(hidden_states)
        return self.proj(hidden_states)

    def forward_last_position(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """解码路径：[B, S, D] → [B, V]，只对最后一个位置做矩阵乘。

        切片必须发生在矩阵乘**之前**，否则省不下任何计算。
        """
        self._validate(hidden_states)
        if hidden_states.size(1) == 0:
            raise ValueError(
                f"hidden_states must have at least one position to decode from, "
                f"but got shape {tuple(hidden_states.shape)}"
            )
        # 先切成 [B, D] 再乘，省下前 S-1 个位置的矩阵乘；
        # 反过来写 self.proj(hidden_states)[:, -1, :] 结果一样但一分计算都不省
        return self.proj(hidden_states[:, -1, :])
