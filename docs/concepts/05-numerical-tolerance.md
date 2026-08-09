# 数值容差与 `assert_close`

> 对应 [`day05.md`](../day05.md) 的任务 3。后续与 Hugging Face 对齐时，
> 测试要判断的是“数值是否在合理误差内”，不是“浮点位模式是否逐位相同”。

## 为什么不用 `==`

整数 token ID 应该精确相同；浮点结果通常不应使用 `==` 或 `torch.equal`。
相同数学表达式可能因计算顺序、硬件后端或 dtype 的舍入而产生末位差异。

项目中优先使用：

```python
torch.testing.assert_close(actual, expected)
```

## 判据

`assert_close` 对每个元素检查：

```text
|actual - expected| <= atol + rtol × |expected|
```

- `atol`（absolute tolerance）保护接近 0 的值；
- `rtol`（relative tolerance）随 `expected` 的量级放宽允许误差。

例如 fp32 默认 `rtol=1.3e-6`、`atol=1e-5` 时，`expected=1000` 的总允许误差约为
`1e-5 + 1.3e-6 × 1000 = 0.00131`。

## PyTorch 默认容差

本项目环境实测的 `torch.testing.assert_close` 默认值：

| dtype | rtol | atol |
|---|---:|---:|
| `float64` | `1e-7` | `1e-7` |
| `float32` | `1.3e-6` | `1e-5` |
| `float16` | `1e-3` | `1e-5` |
| `bfloat16` | `1.6e-2` | `1e-5` |

bf16 的默认相对容差比 fp32 宽约 12,000 倍。bf16 用更少的有效尾数换取更大的指数范围，
因此很适合推理的吞吐与内存场景，却不适合作为严格数值对齐的依据。

## 项目规则

> **正确性对齐一律在 fp32 下做；bf16 只用于性能、内存和溢出行为实验。**

如果测试失败，不要先盲目调大容差。先定位：shape 是否错位、mask 是否广播到错误维度、
权重映射是否错误，或实现是否与参考模型真的不等价。容差只能吸收正常浮点舍入误差，
不能掩盖算法错误。
