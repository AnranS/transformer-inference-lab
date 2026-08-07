# 学习单元 Day 39：完整实验矩阵与复现检查

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 39**｜学习日 27

前六天的 benchmark 数据散落在各个脚本和 CSV 里。今天**统一口径、重跑一遍、
确认可复现**，为明天的最终报告准备好干净的数据。

核心要求：**别人按你的 README 跑，能得到同样的结论。**

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习可复现性的要求 | `docs/concepts/23-reproducibility.md` |
| ⬜ | 任务 2：统一 benchmark 入口 | `benchmarks/run_all.py` |
| ⬜ | 任务 3：定义并跑完实验矩阵 | `benchmarks/results/*.csv` |
| ⬜ | 任务 4：复现检查（跑两遍对比） | 同上 |
| ⬜ | 任务 5：数据整理成报告素材 | `benchmarks/results/summary.md` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | 整理 benchmark 方法，保证结果可复现且控制变量明确 |

---

## 任务 1：学习可复现性的要求（40 分钟）

### 两种「可复现」，要求完全不同

| 类型 | 含义 | 难度 |
|---|---|---|
| **数值可复现** | 同样输入 → **逐比特**同样输出 | 需要努力，且有性能代价 |
| **结论可复现** | 别人跑出的**趋势和量级**一致 | 这才是 benchmark 的目标 |

benchmark 追求的是**结论可复现**。绝对数字必然因硬件而异，
但「Decode 的 tokens/s 随 batch 近似线性上升」这个结论必须能复现。

### 数值可复现需要什么

```python
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False           # 关掉算法自动选择
```

外加环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（cuBLAS 确定性需要）。

**代价**：确定性算法通常更慢，`cudnn.benchmark=False` 也放弃了自动调优。
所以**不要在 benchmark 时开这些**——测性能和测确定性是两件事。

我们的做法：

- **正确性测试**（`tests/`）：固定 seed，但不强制 deterministic algorithms
  （我们的算子都是确定性的，不需要）
- **benchmark**：固定 seed（保证输入一致），**允许非确定性**（要测真实性能）

在笔记里写清这个区分。

### 结论可复现需要什么

**每份结果都必须带完整的环境信息**：

| 项 | 怎么拿 |
|---|---|
| PyTorch 版本 | `torch.__version__` |
| CUDA 版本 | `torch.version.cuda` |
| GPU 型号 | `torch.cuda.get_device_name()` |
| CPU 型号 | `platform.processor()` 或读 `/proc/cpuinfo` |
| dtype | 显式记录 |
| 模型配置 | 全部字段 |
| 随机种子 | 显式记录 |
| 是否 `inference_mode` | 显式记录 |
| 日期 | 便于追溯 |

**没有这些，一周后的数据就和今天的不可比了。** 这不是形式主义——
你在 CPU 上测的第 1 周数据和有 GPU 后测的数据，必须能区分开。

## 任务 2：统一 benchmark 入口（70 分钟）

现在有六七个 `bench_*.py`，各自的计时方式、输出格式可能不一致。统一它们。

### 统一三件事

**1. 都用 `benchmarks/timer.py`**（Day 33 写的）。
如果某个脚本当时手写了计时，改掉。

**2. 统一输出格式。** 一个 CSV schema，所有实验共用：

```text
experiment, device, dtype, batch, seq_len, config_name, variant,
latency_ms_median, latency_ms_min, latency_ms_p25, latency_ms_p75,
tokens_per_s, peak_mem_mb, repeats,
torch_version, cuda_version, device_name, seed, date
```

`variant` 列放「naive / sdpa」「dynamic / static」「mha / gqa / mqa」这类维度，
`experiment` 列区分是哪个实验。**一张宽表装所有数据**，
Day 40 做报告时直接用 pandas 筛选。

**3. 统一模型配置。** 定义两三个命名配置，所有实验都用它们：

```text
tiny    L=2,  D=128, Hq=4,  Hkv=4, I=384,  V=256      正确性测试用
small   L=12, D=768, Hq=12, Hkv=12, I=3072, V=32000   benchmark 主力（GPT-2 small 量级）
gqa     L=12, D=768, Hq=12, Hkv=3,  I=3072, V=32000   GQA 对照
```

**同一份配置贯穿所有实验**，这样各实验的数据才能横向比较。
之前几天如果用了不同配置，今天重跑统一。

### `run_all.py`

```text
python benchmarks/run_all.py --device cuda --output benchmarks/results/
python benchmarks/run_all.py --quick        # 减少 repeats 和维度，用于快速验证
```

`--quick` 模式很重要：完整矩阵可能跑几十分钟，
开发时需要一个能几十秒跑完的版本。

## 任务 3：定义并跑完实验矩阵（100 分钟）

报告必须包含六项：**Prefill、Decode、GQA、Cache、SDPA、BF16**。
按这六项定义矩阵：

| 实验 | 变化维度 | 固定 | 输出的核心结论 |
|---|---|---|---|
| **Prefill** | batch 1/2/4/8 × S 128/512/1024/2048 | small, bf16, sdpa | latency ~ `B×S`；tokens/s 饱和点 |
| **Decode** | batch 1/2/4/8/16 × ctx 128/1024/4096 | small, bf16, static | tokens/s 随 batch 线性上升 |
| **GQA** | mha / gqa / mqa | ctx 4096, batch 4 | Cache 大小与 decode 延迟的关系 |
| **Cache** | dynamic / static | 生成 100 步 | 分配次数、拷贝量、显存曲线 |
| **SDPA** | naive / math / flash / efficient | S 128～4096 | 显存 `O(S²)` vs `O(S)` |
| **BF16** | fp32 / fp16 / bf16 | prefill + decode | 速度、显存、数值误差三方面 |

### 每个实验都要有一个「主结论」

不要只出数据表。每个实验对应**一句可以写进报告的结论**，
而且要能用表里的数字支撑。上面最后一列就是。

如果跑完发现数据不支持你预期的结论，**那才是最有价值的发现**——
在笔记里如实记录，并试着解释为什么。

### BF16 那一项要测三方面

这一项容易只测速度，需要的是完整对照：

1. **速度**：Prefill 和 Decode 分别
2. **显存**：权重 + Cache + 激活
3. **数值误差**：与 fp32 的最大差异（用 Day 20 那张容差表的方法）

第 3 点让这个实验有了正确性维度——**「bf16 快一倍但误差多大」
才是完整的工程判断。**

### 时间管理

100 分钟要跑完六组实验。**先用 `--quick` 跑一遍确认脚本没问题**，
再跑完整版。完整版跑的时候去做任务 5 的整理工作，别干等。

如果时间不够，**优先保证 Prefill / Decode / Cache 三项**（最核心），
GQA / SDPA / BF16 可以缩减维度。

## 任务 4：复现检查（40 分钟）

### 跑两遍，对比

```bash
python benchmarks/run_all.py --output results/run1/
python benchmarks/run_all.py --output results/run2/
```

对比两次的中位数：

| 差异 | 判断 |
|---|---|
| < 3% | 正常波动，结论可信 |
| 3% ～ 10% | 环境有干扰，增加 repeats 重测 |
| > 10% | **有问题**，检查 warmup、后台进程、温度节流 |

**任何小于波动幅度的「性能差异」都不能写进报告。**
比如你测出 `inference_mode` 快 2%，但两次运行本身就差 3%，
那结论只能是「无显著差异」。

写一个小脚本自动做这个对比，输出每一行的差异百分比。

### 检查 README 的复现说明

**自己按 README 走一遍**，假装是第一次接触这个项目：

- [ ] 环境安装步骤完整（`uv sync` 能装齐）
- [ ] 跑 benchmark 的命令能直接复制粘贴
- [ ] 说明了需要多少显存 / 多长时间
- [ ] 说明了在 CPU 上哪些实验会跳过
- [ ] 结果文件的位置和格式有说明

这是最终门禁「README 可让别人复现」的预演。明天（Day 40）正式做。

## 任务 5：数据整理成报告素材（50 分钟）

`benchmarks/results/summary.md`。为明天的报告准备好**图和表**。

### 六张核心图（Day 40 直接用）

1. **Prefill：tokens/s vs batch**（不同 `S` 几条线）→ 饱和点
2. **Decode：tokens/s vs batch** → 近似线性
3. **Decode：latency vs context** → 转折点
4. **Attention 显存 vs S（对数-对数）** → 斜率 2 vs 1
5. **Cache 显存曲线：dynamic vs static** → 交点
6. **GQA：Cache 大小 + decode latency 双轴** → 对应关系

### 三张核心表

1. **正确性汇总**：HF parity 各级误差（Day 20 那张容差表）
2. **优化收益汇总**：Day 37～38 那张含「代价/风险」列的表
3. **KV Cache 显存矩阵**：理论 + 实测（Day 32、36）

**图表的标题和坐标轴要写清单位和配置。** 一张没标注 dtype 和模型配置的图
在报告里是没用的。

### 顺手记下「异常数据」

跑这么多组合，一定会有几个不符合预期的点。列一个清单：

| 现象 | 可能原因 | 是否需要进一步查 |
|---|---|---|

不要隐藏它们。**报告里如实写出「这一项与预期不符，原因待查」
比假装一切完美更有价值**，也是 Day 40「下一阶段瓶颈清单」的素材来源。

---

## 过关标准

- [ ] 能区分「数值可复现」和「结论可复现」，知道 benchmark 追求哪个
- [ ] 知道为什么不该在 benchmark 时开 deterministic algorithms
- [ ] 所有脚本统一用 `timer.py`、统一 CSV schema、统一模型配置
- [ ] `run_all.py` 支持 `--quick`
- [ ] 六项实验矩阵跑完，每项有一句数据支撑的主结论
- [ ] BF16 实验测了速度、显存、数值误差三方面
- [ ] **跑两遍对比，波动 < 3%**（或说明原因并增加 repeats）
- [ ] 每份结果都带完整环境信息
- [ ] 自己按 README 走了一遍复现流程
- [ ] 六张图三张表准备完成
- [ ] 异常数据清单记录完成

---

## 今日最重要的面试式问题

**怎么保证 benchmark 结果可信？**

三层：

1. **测量方法正确**：warmup、前后同步、重复 20+ 次报中位数（Day 33 的四要素）
2. **控制变量 + 记录环境**：一次只改一个维度；每份数据带 PyTorch/CUDA 版本、
   设备型号、dtype、模型配置、seed
3. **跑两遍验证波动幅度**：任何小于波动幅度的「差异」都不能当结论。
   测出 2% 的提升但运行间波动有 3%，结论只能是「无显著差异」

追问：**为什么 benchmark 时不该开 `torch.use_deterministic_algorithms(True)`？**

因为确定性算法通常更慢，而且 `cudnn.benchmark=False` 会放弃算法自动调优。
开了之后测到的是「确定性模式下的性能」，不是用户实际会遇到的性能。

**确定性和性能是两个正交的目标**：
正确性测试需要可复现的数值（固定 seed 就够，我们的算子本身是确定性的），
benchmark 需要真实的性能。分开设置。
