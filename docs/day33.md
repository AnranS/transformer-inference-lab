# 学习单元 Day 33：Prefill Benchmark 与可靠测量方法

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 33**｜学习日 23

今天学**怎么测得准**。这比测什么更重要——
方法不对的话，后面几天的所有数据都是噪声。

阅读目标就是四个词：**warmup、同步、重复测量、结果比较**。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习可靠测量的四要素 | `docs/concepts/19-benchmark-method.md` |
| ⬜ | 任务 2：搭测量工具 | `benchmarks/timer.py` |
| ⬜ | 任务 3：Prefill benchmark（四维扫描） | `benchmarks/bench_prefill.py` |
| ⬜ | 任务 4：结果记录与初步分析 | `benchmarks/results/prefill.csv` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [PyTorch Benchmark 教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html) | 学会 warmup、同步、重复测量和结果比较 |

---

## 任务 1：学习可靠测量的四要素（40 分钟）

### 1. Warmup

第一次运行必然慢，原因有好几层：

| 原因 | 影响 |
|---|---|
| CUDA context 初始化 | 首次调用几百 ms |
| kernel 的 JIT / 自动调优 | cuBLAS 会试几种算法选最快 |
| 内存池冷启动 | 首次分配要向驱动申请 |
| CPU 侧的 Python / 导入开销 | 首次进函数慢 |

**丢掉前 3～5 次**。`torch.utils.benchmark.Timer` 会自动 warmup，
手写的话必须自己加。

### 2. 同步

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
fn()
torch.cuda.synchronize()      # ← 这一句最容易漏
dt = time.perf_counter() - t0
```

CUDA 调用是**异步**的：`fn()` 只是把 kernel 提交到流里就返回了。
不加第二个 `synchronize()`，你测到的是「提交耗时」，通常是真实时间的百分之一。

**这是 GPU benchmark 最常见的错误**，会让你得出「这个算子快得不可思议」的结论。

Day 34 会更深入讲 CUDA 异步语义。

### 3. 重复测量与统计量

单次测量没有意义。至少跑 10～30 次，然后：

**报告中位数，不是平均值。** 理由：偶发的 OS 调度、其他进程抢占会产生
个别极大值，平均值被它们拉偏，中位数稳健得多。

同时记录**最小值**和**四分位距**：

- 最小值 ≈ 理论最好情况（无干扰）
- 四分位距大 → 测量环境不稳定，结论不可信

### 4. 结果比较

比较两个实现时，**不能只看两个中位数谁大**。
如果差异小于测量波动，那就是没有差异。

`torch.utils.benchmark` 的 `Compare` 类会帮你做这个。
手写的话，简单规则：**差异小于 5% 且波动大于 5% → 判定为「无显著差异」**。

### 额外两条实践规则

**5. 控制变量。** 一次只改一个维度。同时改 batch 和 dtype，
你无法归因是哪个带来的变化。

**6. 记录环境。** 每份结果都要带上：PyTorch 版本、CUDA 版本、GPU 型号、
dtype、是否 `inference_mode`。**否则一周后的数据无法和今天的比。**

## 任务 2：搭测量工具（50 分钟）

`benchmarks/timer.py`，一个可复用的计时器：

```text
benchmark(fn, *, warmup=5, repeats=20, device) -> BenchResult
```

`BenchResult` 至少包含：`median_ms`、`min_ms`、`p25_ms`、`p75_ms`、`repeats`。

实现要点：

- **自动处理 CPU / CUDA 的同步差异**（CPU 上不需要 `synchronize`）
- warmup 的结果**丢弃**，不参与统计
- 提供 `peak_memory_mb`（CUDA 上用 `max_memory_allocated`，
  调用前先 `reset_peak_memory_stats()`）
- 提供一个 `环境信息` 函数，返回 PyTorch / CUDA 版本、设备名

**建议同时用 `torch.utils.benchmark.Timer` 交叉验证一次**：
用两种方式测同一个函数，结果应该接近。差得多说明你的实现有问题
（大概率是同步或 warmup）。

### 关于 CPU 上跑 benchmark

前 4 周 CPU 即可，之后用 RTX 5070 Ti。如果你现在在 CPU 上：

- **同步相关的代码依然要写**（为将来准备），只是运行时跳过
- CPU 上的数据仍然有意义：**相对**趋势（batch 翻倍时间怎么变）是可比的
- 但**绝对**数字和 GPU 上完全不同，尤其「访存密集 vs 计算密集」的对比
  在 CPU 上会弱化很多（CPU 的算力/带宽比和 GPU 差别大）

在结果文件里**明确标注设备**，不要把 CPU 和 GPU 的数据混在一张表里。

## 任务 3：Prefill benchmark（四维扫描）（60 分钟）

`benchmarks/bench_prefill.py`。四个维度：

| 维度 | 取值 |
|---|---|
| batch | 1, 2, 4, 8 |
| prompt 长度 | 128, 512, 1024, 2048 |
| dtype | fp32, fp16, bf16 |
| attention 实现 | naive vs SDPA |

**全笛卡尔积是 `4×4×3×2 = 96` 组**，每组 20 次重复。
如果单次 Prefill 是 50 ms，总时间约 96×20×0.05 = 96 秒——可以接受。

但要**控制模型规模**。用玩具配置（`L=2, D=128`）测不出任何有意义的趋势，
建议用一个「小但真实」的配置：

```text
L=12, D=768, Hq=12, Hkv=12, I=3072, V=32000      （GPT-2 small 量级）
```

CPU 上这个规模的 2048 长度 Prefill 可能要几秒，那就把 prompt 上限降到 1024，
或者减少 repeats。**在文件里记录你实际用的配置。**

### 记录的指标（三项）

| 指标 | 计算方式 |
|---|---|
| latency | 中位数耗时（ms） |
| tokens/s | `batch × prompt_len / latency` |
| 峰值显存 | `max_memory_allocated`（CUDA） |

### 预期会看到什么（先猜再测）

**动手前先写下预测**，测完对比。这是最有效的学习方式。

我的预测，供你对照：

1. **latency 随 `batch × S` 近似线性增长**（Prefill 是计算密集，
   算力打满后时间正比于总工作量）
2. **tokens/s 在小 batch 时随 batch 上升，然后饱和**
   （小 batch 时算力没打满，有提升空间；打满后不再改善）
3. **fp16/bf16 明显快于 fp32**（GPU 上有 Tensor Core 加速；
   **CPU 上可能反而更慢**，因为缺乏原生 bf16 支持要来回转换）
4. **SDPA 快于 naive**，且差距随 `S` 增大而扩大
   （naive 要实体化 `[B,Hq,S,S]` 的中间矩阵，`S²` 显存和带宽开销；
   SDPA 的 flash 后端不实体化它）
5. **naive 的峰值显存随 `S²` 增长**，SDPA 随 `S` 增长——
   这是最能说明 FlashAttention 价值的一条

第 5 条特别值得测：`S=2048` 时 `[8, 12, 2048, 2048]` 的 fp32 中间矩阵
是 1.5 GiB，naive 实现很可能直接 OOM。**OOM 本身就是一个有价值的数据点**，
记下来。

## 任务 4：结果记录与初步分析（30 分钟）

存成 CSV：`benchmarks/results/prefill.csv`，列包括所有维度 + 所有指标 + 环境信息。

**用 CSV 而不是打印到终端**，因为 Day 39 要把四周的数据汇总成实验矩阵。
现在存成结构化格式，到时候直接读。

写三条结论进 `docs/concepts/19-benchmark-method.md`，每条都要有数据支撑：

1. 关于 batch 和 tokens/s 的关系
2. 关于 dtype 的影响
3. 关于 naive vs SDPA（时间和显存两方面）

**同时记录哪些预测错了。** 预测错的地方才是真正学到东西的地方——
在笔记里写清「我以为 X，实际是 Y，原因是 Z」。

---

## 过关标准

- [ ] 能说出 warmup 的四个原因
- [ ] 知道漏掉第二个 `synchronize()` 会导致什么（测到提交时间）
- [ ] 报告中位数而非平均值，并知道为什么
- [ ] `timer.py` 能自动处理 CPU/CUDA 差异，且与 `torch.utils.benchmark` 交叉验证过
- [ ] 四维扫描完成，结果存为 CSV 且含环境信息
- [ ] **测量前写下了预测**，测量后对比并记录了偏差
- [ ] 验证了 naive 的显存随 `S²` 增长、SDPA 随 `S` 增长
- [ ] 三条带数据的结论写进笔记

---

## 今日最重要的面试式问题

**GPU 上做性能测量必须注意什么？**

四件事：

1. **Warmup**：丢掉前几次（CUDA context 初始化、kernel 自动调优、内存池冷启动）
2. **同步**：计时的**前后都要** `torch.cuda.synchronize()`。
   CUDA 是异步的，不同步只会测到「提交 kernel 的时间」
3. **重复测量**：跑 10～30 次报**中位数**（平均值会被偶发的调度抖动拉偏）
4. **控制变量 + 记录环境**：一次只改一个维度；记下 PyTorch/CUDA 版本、GPU 型号、dtype

追问：**为什么 naive attention 在长序列上会 OOM，而 SDPA 不会？**

因为 naive 会**实体化** `[B, Hq, S, S]` 的注意力矩阵，显存随 `S²` 增长。
`B=8, Hq=12, S=2048` 的 fp32 中间矩阵就是 1.5 GiB。

SDPA 的 FlashAttention 后端**从不实体化完整的注意力矩阵**——
它分块计算并用在线 softmax 累积结果，显存只随 `S` 线性增长。
这正是 FlashAttention 的核心价值：它不是算得更快，而是**不存那个矩阵**。
