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

    model(input_ids) 必须返回 [B, S, V] logits。生成循环的契约由
    tests/test_generation.py 定义：

      长度   未命中 EOS 时，输出长度 = 输入长度 + max_new_tokens
      确定性 固定权重与输入，两次生成结果完全相同
      停止   eos_token_id 命中时提前返回，输出长度小于上限；None 表示不启用
      梯度   全程在 torch.inference_mode() 下运行
      调试   当前教学 baseline 每轮打印 next token；这不是稳定 API 保证
    """
    output_ids = input_ids
    with torch.inference_mode():
        for step in range(max_new_tokens):
            logits = model(output_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            print(f"step {step + 1}: next_token={next_token.squeeze(1).tolist()}")

            # Day 4 的简化版：仅当整个 batch 都生成 EOS 才停止。
            # 部分序列先结束仍会继续生成；Day 21 用 finished mask 单独处理。
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
    return output_ids
