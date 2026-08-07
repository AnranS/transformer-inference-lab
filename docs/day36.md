# 学习单元 Day 36：显存快照与 Cache 增长分析

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 36**｜学习日 25

进入毕业项目段（Day 36～40）。今天把**显存**这条线彻底搞清——
它是长上下文推理的第一约束，也是最终报告里最有说服力的一节。

Day 31～32 算的是**理论值**，今天测**实际值**，并解释两者的差距。

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习三种显存口径 | `docs/concepts/21-memory.md` |
| ⬜ | 任务 2：显存构成拆解 | `benchmarks/bench_memory_breakdown.py` |
| ⬜ | 任务 3：Cache 增长实测 vs 理论 | `notebooks/day26_cache_growth.ipynb` |
| ⬜ | 任务 4：生成内存快照 | `benchmarks/results/memory_snapshot.pickle` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [PyTorch CUDA Memory](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | [torch.cuda.memory_snapshot](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html) | 区分 allocated、reserved、峰值显存，并生成内存快照 |

---

## 任务 1：学习三种显存口径（30 分钟）

Day 34 提过，今天讲透。**报告显存时说错口径，结论会差一倍以上。**

### 三个层次

```text
┌─ nvidia-smi 看到的（进程总占用）───────────────────┐
│  CUDA context（几百 MB，固定开销）                  │
│  ┌─ reserved（PyTorch 缓存池向驱动申请的）────────┐ │
│  │  ┌─ allocated（张量实际占用）───────────────┐  │ │
│  │  │  权重 + Cache + 激活                     │  │ │
│  │  └─────────────────────────────────────────┘  │ │
│  │  碎片 + 空闲块（不还给驱动，留着复用）          │ │
│  └───────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

| API | 含义 | 什么时候用它 |
|---|---|---|
| `memory_allocated()` | 当前张量占用 | 算「我的数据占多少」 |
| `max_memory_allocated()` | 张量占用峰值 | **报告里主要用这个** |
| `memory_reserved()` | 缓存池总量 | 判断碎片化程度 |
| `max_memory_reserved()` | 缓存池峰值 | 估算「实际需要多大显卡」 |

### 三个必须知道的事实

**1. `reserved > allocated` 是正常的。** PyTorch 的缓存分配器
从驱动大块申请、内部细分，释放张量时**不还给驱动**（为了下次分配快）。
差值就是缓存池里的空闲块。

**2. `nvidia-smi` 还要再大一截。** 它包含 CUDA context
（驱动和运行时的固定开销，几百 MB）。**所以 `nvidia-smi` 的数字
不能用来验证你的显存计算**——它包含了和你的模型无关的部分。

**3. `reserved` 远大于 `allocated` 说明碎片化。** 典型原因是
反复分配释放不同大小的张量——**正是 `DynamicCache` 的行为**（Day 29）。
这是预分配的又一个好处：碎片少。

### `reset_peak_memory_stats()` 必须用

峰值统计是**累积**的，不重置的话你测的是「从进程启动至今的峰值」，
包含了 warmup 和之前的实验。

```python
torch.cuda.reset_peak_memory_stats()
# ... 测量目标 ...
peak = torch.cuda.max_memory_allocated()
```

**每次测量前都要重置。** 忘了这一步是显存测量最常见的错误。

## 任务 2：显存构成拆解（60 分钟）

`benchmarks/bench_memory_breakdown.py`。把显存拆成四块，分别测：

| 组成 | 怎么测 | 特征 |
|---|---|---|
| **权重** | 加载模型后的 `memory_allocated()` | 固定 |
| **KV Cache** | 分配 Cache 前后的差值 | 随 `B × S` 增长 |
| **激活（Prefill）** | Prefill 期间的峰值 − 权重 − Cache | 随 `B × S`（naive 时 `S²`） |
| **激活（Decode）** | Decode 期间的峰值 − 权重 − Cache | 随 `B`，很小 |

测量顺序很重要：

```python
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
model = build_model(config).to("cuda")
weights = torch.cuda.memory_allocated()

cache = StaticCache(...)
cache_mem = torch.cuda.memory_allocated() - weights

torch.cuda.reset_peak_memory_stats()
prefill(input_ids, cache=cache)
prefill_peak = torch.cuda.max_memory_allocated()
prefill_activations = prefill_peak - weights - cache_mem
```

### 三条会看到的结论

**1. 权重和 Cache 的比例随 context 反转。**
Day 31～32 算过：Llama-3 8B 在 `B=16, S=8K` 时 Cache 超过权重。
你的小模型上也会有这个转折点，算出它并实测验证。

**2. Prefill 的激活峰值可能超过权重和 Cache。**
naive attention 的 `[B,Hq,S,S]` 中间矩阵是 `S²`——
`B=8, Hq=12, S=2048` 的 fp32 就是 1.5 GiB。
这也是**Prefill 常需要分块（chunked prefill）**的原因。

**3. Decode 的激活极小。** 只处理 1 个位置，激活是 `[B, 1, *]` 量级。
所以 Decode 阶段的显存**几乎全是权重 + Cache**——
这让 Decode 的显存需求非常好预测，是调度器能精确计算容量的基础。

## 任务 3：Cache 增长实测 vs 理论（50 分钟）

### 对比表

对每一步生成，同时记录：

| | 来源 |
|---|---|
| 理论值 | `2 × L × B × S × Hkv × Dh × bytes`（Day 32 的公式） |
| 实测值 | Cache 张量的 `numel() × element_size()` 之和 |
| `memory_allocated` 增量 | 相邻两步的差 |

**理论值和实测值应该完全相等**（这是 Day 32 已经验证过的）。
如果不等，检查是不是把 `Hq` 当成了 `Hkv`（GQA 实现的坑，Day 31）。

**`memory_allocated` 的增量会大于理论值**，因为还有：

- 分配器的对齐（按块分配，比如 512 字节对齐）
- `DynamicCache` 的 concat 临时张量

### 两种 Cache 的显存曲线

画在同一张图上，横轴生成步数，纵轴显存：

```text
DynamicCache   阶梯上升，且 concat 瞬间有尖峰（新旧两份并存）
StaticCache    一开始就是水平线（预分配满额）
```

**这张图是 Day 29 那个权衡的最佳可视化**，放进最终报告：

- `StaticCache` 起点高但可预测、无尖峰
- `DynamicCache` 起点低但有尖峰、且长期看会超过

两条线的**交点**就是「预分配开始划算」的生成长度。算出这个交点。

### 顺手回答一个实际问题

给定显存预算（比如 24 GB 的 5070 Ti），能跑多大的配置？

```text
可用 = 显存总量 − CUDA context(~0.5G) − 权重 − Prefill 激活峰值
max_tokens = 可用 / per_token_kv_bytes
```

用 Day 32 推的「每 token KV 成本」（8B 模型 128 KiB/token）算一遍。
**这个估算能力是推理工程最实用的技能之一**，写进笔记。

## 任务 4：生成内存快照（40 分钟）

PyTorch 的内存快照能可视化**每一块显存是谁分配的**。

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... 跑一次完整生成 ...
torch.cuda.memory._dump_snapshot("benchmarks/results/memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)   # 关掉
```

然后把 pickle 拖到 <https://docs.pytorch.org/memory_viz> 看（纯前端，不上传数据）。

### 在快照里找这四样

1. **权重那几个大块**（进程开始就分配，一直不释放）
2. **Cache 的块**：`StaticCache` 是几个大块；`DynamicCache` 是很多逐渐变大的块
3. **激活的反复分配释放**（Prefill 期间密集，Decode 期间稀疏）
4. **碎片**：块之间的空隙。`DynamicCache` 下应该明显更多

### 注意

- `_record_memory_history` 本身有开销，**不要在 benchmark 时开着**
- 记录会产生大文件（几十 MB），**加进 `.gitignore`**，
  只把截图或结论放进报告
- 这些 API 带下划线前缀（私有），跨版本可能变。**在笔记里记下你用的 PyTorch 版本**

如果没有 GPU，这个任务跳过，在笔记里记一句「待有 GPU 时补」。
前三个任务在 CPU 上也能做（用 `sys.getsizeof` 或张量的
`numel() × element_size()` 手工统计）。

---

## 过关标准

- [ ] 能画出 allocated / reserved / nvidia-smi 三层的包含关系
- [ ] 知道 `nvidia-smi` 的数字不能用来验证显存计算，以及为什么
- [ ] 每次测量前都调了 `reset_peak_memory_stats()`
- [ ] 显存拆成权重 / Cache / Prefill 激活 / Decode 激活四块并实测
- [ ] 找到了 Cache 超过权重的转折点（理论 + 实测）
- [ ] 理论 Cache 值与实测完全相等
- [ ] 画出两种 Cache 的显存曲线，并算出交点
- [ ] 能估算「给定显存能缓存多少 token」
- [ ] 生成了内存快照并在其中找到四类块（有 GPU 的话）

---

## 今日最重要的面试式问题

**推理时显存都花在哪了？**

四块：

| 组成 | 特征 |
|---|---|
| **权重** | 固定，`参数量 × dtype 字节数` |
| **KV Cache** | 随 `B × S` 线性增长，长上下文时会**超过权重** |
| **Prefill 激活** | 随 `B × S`（naive attention 下是 `S²`），常是峰值来源 |
| **Decode 激活** | 极小（只处理 1 个位置） |

加上 CUDA context 的几百 MB 固定开销。

追问：**`nvidia-smi` 显示 20 GB，`memory_allocated()` 只有 12 GB，差的 8 GB 在哪？**

三部分：

1. **CUDA context**（几百 MB）：驱动和运行时的固定开销
2. **PyTorch 缓存池的空闲块**：`memory_reserved() - memory_allocated()`。
   PyTorch 释放张量时不还给驱动，留着复用
3. **碎片**：反复分配不同大小的张量导致。`DynamicCache` 的 concat 是典型来源

所以报告显存时必须说清口径。一般用 `max_memory_allocated()` 表示「数据需要多少」，
用 `max_memory_reserved()` 估算「需要多大的显卡」。
