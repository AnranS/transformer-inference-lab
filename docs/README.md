# 学习导航

这里是 `Transformer Inference From Scratch` 的学习入口。项目包含两套编号：

- **学习单元 Day 1～40**：知识与实现的依赖顺序。
- **28 个学习日**：实际执行节奏；一天通常完成 1～2 个学习单元。

例如 `day07-08.md` 表示一个学习日内完成学习单元 Day 7 和 Day 8。

## 当前进度

- 已完成：Day 1～10
- 下一单元：[Day 11～12：RMSNorm 与 SwiGLU](./day11-12.md)
- 总计划：[roadmap.md](./roadmap.md)
- 全局 shape 约定：[00-tensor-conventions.md](./00-tensor-conventions.md)

## 每天怎么学

按下面顺序使用仓库：

1. 打开当天的 `dayXX.md`，确认任务、资料和过关标准。
2. 从 [concepts/README.md](./concepts/README.md) 找到对应概念笔记。
3. 运行对应的 `notebooks/dayNN_*.ipynb`，观察 shape 和数值。
4. 在 `src/mini_transformer/` 实现功能，在 `tests/` 用 TDD 固化契约。
5. 每周门禁完成后，把结论整理到 [deliverables/README.md](./deliverables/README.md)。

## 四周路线

- 第 1 周：Tokenizer、Embedding、LM Head、Attention、Mask、Multi-Head Attention
- 第 2 周：RMSNorm、SwiGLU、RoPE、Decoder Block、Llama 对齐
- 第 3 周：生成、采样、Padding、Prefill、Decode、KV Cache
- 第 4 周：GQA、benchmark、profiler、显存分析与可复现报告

具体学习日映射、必读资料和验收门禁以 [roadmap.md](./roadmap.md) 为准。

## 文档分工

- `day*.md`：当天做什么
- `concepts/`：一个知识点的细粒度解释
- `deliverables/`：跨多个 Day 的阶段总结
- `roadmap.md`：完整课程结构、依赖和毕业标准

未来任务在日文档和 roadmap 中会提前出现；对应源码、测试或专题文档只在学到该单元时创建。
