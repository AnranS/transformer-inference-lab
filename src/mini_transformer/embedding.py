import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """按 token ID 查表，把离散索引变成连续向量。

    契约由 tests/test_embedding.py 定义，跑 `pytest -q` 检验：

      形状   权重 [V, D]；输入 token_ids [B, S] → 输出 [B, S, D]
      一致性 相同 token ID 在任何位置都查到相同向量
      构造   vocab_size / hidden_size 不是正数 → ValueError
      输入   不是二维 → ValueError；dtype 不是 torch.long → TypeError
      越界   token ID 为负或 >= vocab_size → IndexError
      边界   空 batch [0, S] 不报错，输出 [0, S, D]
      开关   check_token_range=False 时跳过越界检查
      推理   torch.inference_mode() 下可运行，输出 requires_grad 为 False
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        check_token_range: bool = True,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, but got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, but got {hidden_size}")
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.check_token_range = check_token_range

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be a 2D tensor [B, S], but got shape {tuple(token_ids.shape)}"
            )
        if token_ids.dtype != torch.long:
            raise TypeError(f"token_ids must use torch.long, but got {token_ids.dtype}")
        # min/max 会把 GPU 上的值同步回 CPU，解码时开销明显，所以做成可关闭；
        # 空张量没有最小值，min() 会直接抛 RuntimeError，而且没有元素也无所谓越界
        if self.check_token_range and token_ids.numel() > 0:
            vocab_size = self.embedding.num_embeddings
            min_id = int(token_ids.min())
            max_id = int(token_ids.max())
            if min_id < 0 or max_id >= vocab_size:
                raise IndexError(
                    f"token_ids out of range [0, {vocab_size}), "
                    f"but got min={min_id}, max={max_id}"
                )
        return self.embedding(token_ids)
