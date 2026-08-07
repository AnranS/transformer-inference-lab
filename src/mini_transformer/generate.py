import torch
from torch import nn


def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """无 KV Cache 的贪心生成 baseline：每轮都把全部历史重新 forward 一遍。

    这是刻意保留的低效实现——Day 26～30 实现 KV Cache 时要拿它当对照。

    契约（任务 9 落到 tests/test_generation.py）：

      长度   未命中 EOS 时，输出长度 = 输入长度 + max_new_tokens
      确定性 固定权重与输入，两次生成结果完全相同
      停止   eos_token_id 命中时提前返回，输出长度小于上限；None 表示不启用
      梯度   全程在 torch.inference_mode() 下运行
    """
    raise NotImplementedError
