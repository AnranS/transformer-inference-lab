import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        check_token_range: bool = True,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.check_token_range = check_token_range
        # weight shape: [V, D]，每一行对应一个 token 的向量
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def _check_token_range(self, token_ids: torch.Tensor) -> None:
        if token_ids.numel() == 0:
            return

        min_id = int(token_ids.min())
        max_id = int(token_ids.max())
        if min_id < 0 or max_id >= self.vocab_size:
            raise IndexError(
                f"token_ids out of range [0, {self.vocab_size}), "
                f"got min={min_id}, max={max_id}"
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, S]")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
        # 读 min/max 会把 GPU 上的值同步回 CPU，卡住流水线；
        # 跑性能测试时用 check_token_range=False 关掉，越界仍由 nn.Embedding 兜底报错
        if self.check_token_range:
            self._check_token_range(token_ids)

        # 按 token_ids 从 [V, D] 中取行 → [B, S, D]
        return self.embedding(token_ids)
