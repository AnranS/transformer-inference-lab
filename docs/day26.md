# 学习单元 Day 26：无 Cache baseline 与 Prefill / Decode 分解

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 26**｜学习日 18

进入 KV Cache 这一段。今天**不写 Cache**，先做两件必要的准备：

1. 搞清 Cache 到底缓存**什么**（很多人这里理解就是错的）
2. 用数据**证明**当前实现在浪费计算——有了 baseline，后面的优化才有意义

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：学习 Cache 缓存的是什么 | `docs/concepts/16-kv-cache-basics.md` |
| ⬜ | 任务 2：给 baseline 加计量 | `benchmarks/bench_no_cache.py` |
| ⬜ | 任务 3：量化浪费 | `notebooks/03_cache_growth.ipynb` |
| ⬜ | 任务 4：Prefill / Decode 概念分解 | 同上笔记 |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Caching 原理](https://huggingface.co/docs/transformers/main/cache_explanation) | [HF Generation Utilities](https://huggingface.co/docs/transformers/internal/generation_utils) | 明确缓存的是**每一层的 K/V**，而不是 token 或 Attention score |

---

## 任务 1：学习 Cache 缓存的是什么（30 分钟）

### 先排除三个常见误解

| 误解 | 为什么不对 |
|---|---|
| 「缓存 token」 | token id 本来就在 `input_ids` 里，缓存它毫无意义 |
| 「缓存 attention score」 | score 是 `[B,Hq,Sq,Skv]`，**每轮都在变**（新 query 对所有 key），缓存不了 |
| 「缓存 hidden states」 | 接近但不对。hidden states 确实不变，但 attention 需要的是投影后的 K/V，缓存 hidden 还得每轮重新投影 |

### 正确答案：每一层的 K 和 V

```text
每层缓存两个张量：
  K  [B, Hkv, S, Dh]
  V  [B, Hkv, S, Dh]

共 L 层 → 2 × L 个张量
```

**注意「每一层」。** 一个 32 层模型有 64 个缓存张量，不是 2 个。
这是 Day 16 那个公式里 `2 × L` 的来源。

### 为什么 K/V 可以缓存（核心）

因为**因果性**。位置 `i` 的 `K_i`、`V_i` 只依赖 token `i` 自己
（经过前面若干层的 hidden state，而那些 hidden state 又只依赖 `0..i`）。

**新增一个 token 不会改变任何历史位置的 K/V。** 所以算过一次就永久有效。

反过来想：如果模型是双向的（BERT 那样），新增 token 会改变所有位置的表示，
KV Cache 就完全不成立。**KV Cache 是因果语言模型的专属优化。**

### 为什么 Q 不需要缓存

这个问题很少有人问，但答案很能说明问题。

Decode 时我们只要**最后一个位置**的输出（Day 3 的结论）。
最后一个位置的输出需要：它自己的 `Q`，以及**所有位置**的 `K`、`V`。

历史位置的 `Q` 呢？它们对应的输出在**上一轮已经算过并且丢掉了**——
我们当时只要了那一轮的最后一位。所以历史 `Q` 永远不会再被用到。

```text
需要缓存：K、V   （历史的会被反复用）
不需要缓存：Q     （只用当前这一个）
```

这也解释了为什么叫「KV Cache」而不是「QKV Cache」。

## 任务 2：给 baseline 加计量（50 分钟）

在 `benchmarks/bench_no_cache.py` 里包一层计量，记录每轮四项
（四项）：

| 指标 | 怎么拿 |
|---|---|
| 输入长度 | `input_ids.shape[1]` |
| forward 时间 | `time.perf_counter()` 前后差；**GPU 上必须先 `torch.cuda.synchronize()`** |
| 计算的 token 数 | 等于输入长度（无 Cache 时每个位置都重算） |
| 峰值显存 | `torch.cuda.max_memory_allocated()`；CPU 上跳过 |

### 关于计时的两个坑（Day 33 会正式学，今天先按规矩做）

**1. CUDA 是异步的。** `model(x)` 立刻返回，kernel 还在排队。
不 `synchronize()` 就计时，量到的是「提交任务的时间」，几乎是 0。

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
out = model(x)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
```

**2. 第一次调用要 warmup。** 首次会触发 kernel 编译、内存池初始化、
cuDNN 算法选择，可能比稳态慢几十倍。**先跑 3～5 次丢掉**，再开始测。

CPU 上第 1 条不需要，但第 2 条依然需要（首次有内存分配和缓存冷启动）。

## 任务 3：量化浪费（60 分钟）

新建 `notebooks/03_cache_growth.ipynb`。这是**「证明」**环节。

### 第一步：观察输入长度单调递增

在循环里打印，你会看到：

```text
第 1 轮 forward 10 个 token → 产出第 11 个
第 2 轮 forward 11 个 token → 产出第 12 个   ← 前 10 个又算了一遍
第 3 轮 forward 12 个 token → 产出第 13 个   ← 前 11 个又算了一遍
```

### 第二步：证明重算结果完全相同

这是「证明浪费」最直接的方式：**把第 1 轮和第 2 轮前 10 个位置的
K/V（或 hidden states）抓出来对比**。

```python
# 用 hook 抓第 0 层的 k_proj 输出
# 第 2 轮的 k[:, :, :10, :] 必须和第 1 轮的 k 完全相同
torch.testing.assert_close(k_round2[:, :, :10, :], k_round1, rtol=0, atol=0)
```

`rtol=0, atol=0` 逐比特相同——**这就是「白算了」的严格证明**，
也顺手验证了任务 1 里「新 token 不改变历史 K/V」那个论断。

**这一条比任何时间数据都更有说服力**，一定要做。

### 第三步：累加总量并对比

累加所有轮次处理的 token-位置总数：

```text
无 Cache 累计 = Σ(prompt_len + i)，i 从 0 到 n-1
有 Cache 累计 = prompt_len + (n - 1)
```

| prompt 长度 | 生成数 | 无 Cache 累计 | 有 Cache 累计 | 倍数 |
|---|---|---|---|---|
| 10 | 20 | 390 | 29 | 13.4× |
| 128 | 50 | 7,625 | 177 | 43.1× |
| 1024 | 100 | 107,350 | 1,123 | 95.6× |

**规律：prompt 越长、生成越多，浪费越大。** 无 Cache 是 `O(n²)`，
有 Cache 是 `O(n)`。这就是 KV Cache 的全部价值。

自己用公式算一遍上表，别直接抄——`Σ(p+i)` 的求和要会推。

### 第四步：画图

画两条曲线：横轴生成步数，纵轴累计处理的 token 数。
一条是抛物线（无 Cache），一条是直线（有 Cache）。
**这张图放进第五篇专题文档和最终报告。**

## 任务 4：Prefill / Decode 概念分解（40 分钟）

这是整个推理优化领域最重要的一个概念划分。写进
`docs/concepts/16-kv-cache-basics.md`。

| | **Prefill** | **Decode** |
|---|---|---|
| 输入 | 完整 prompt，`S` 个 token | 1 个 token |
| `Sq` | `S` | 1 |
| `Skv` | `S` | `S + t` |
| 矩阵乘形状 | 大 GEMM `[S, D] @ [D, *]` | 小 GEMM `[1, D] @ [D, *]` |
| 瓶颈 | **计算密集**（算力打满） | **访存密集**（要读全部权重 + 全部 Cache） |
| 次数 | 1 次 | `n-1` 次 |
| 对应指标 | **TTFT**（首 token 延迟） | **TPOT**（每 token 延迟） |
| 优化方向 | 提高算力利用率、分块 | 提高访存效率、增大 batch |

### 为什么 Decode 是访存密集（关键洞察）

Decode 时每一层要做的事：

```text
读取：该层全部权重（几十 MB）+ 该层全部 KV Cache（随 context 增长）
计算：几个 [1, D] × [D, *] 的矩阵向量乘
```

**读的数据量远大于计算量。** 一个 `[1,4096] @ [4096,4096]` 的乘法
只有 33 MFLOP，但要读 32 MB 的权重（bf16）。
在现代 GPU 上（算力 TFLOP 级、带宽 TB/s 级），时间几乎全花在读上。

**推论：Decode 阶段增大 batch 几乎不增加延迟**——
权重只读一次，被整个 batch 共享。这是所有推理服务做 continuous batching 的原因。

Prefill 相反：`S` 个 token 一起算，`[S,4096] @ [4096,4096]` 是真正的 GEMM，
算力能打满，增大 batch 会线性增加时间。

**这两种截然相反的性能特征，是第 4 周全部 benchmark 的主题。**
今天先在概念上分清，Day 33～34 会实测。

---

## 过关标准

- [ ] 能说出缓存的是**每一层的 K 和 V**，形状 `[B, Hkv, S, Dh]`
- [ ] 能解释为什么 K/V 可以缓存（因果性），以及为什么 Q 不需要
- [ ] 知道为什么 attention score 不能缓存
- [ ] baseline 计量记录了四项指标，且计时前做了 warmup 和 synchronize
- [ ] **用 `rtol=0, atol=0` 证明了历史 K/V 每轮完全相同**
- [ ] 算出了浪费倍数表，能推导 `Σ(p+i)` 的求和
- [ ] 画出了 `O(n²)` vs `O(n)` 的对比曲线
- [ ] 能说清 Prefill 与 Decode 的六项差异，尤其「计算密集 vs 访存密集」

---

## 今日最重要的面试式问题

**KV Cache 缓存的到底是什么？**

**每一层的 K 和 V 张量**，各为 `[B, Hkv, S, Dh]`。`L` 层模型有 `2L` 个缓存张量。

不是 token（本来就有），不是 attention score（每轮都变，缓存不了），
也不是 hidden states（缓存了还得重新投影）。

追问：**为什么 K/V 能缓存，Q 不能？**

K/V 能缓存是因为**因果性**：位置 `i` 的 K/V 只依赖 `0..i`，
新增 token 不改变任何历史位置的 K/V，算一次永久有效。

Q 不需要缓存是因为 Decode 只要**最后一个位置**的输出，
只用到当前 token 的 Q；历史 Q 对应的输出在上一轮就算完丢掉了，永远不会再用。

再追问：**Decode 阶段为什么是访存密集的？**

因为每层要读全部权重（几十 MB）+ 全部 KV Cache，
而计算只是几个 `[1,D] @ [D,*]` 的矩阵向量乘。读的字节数远超算的 FLOP 数。

推论是**Decode 增大 batch 几乎不增加延迟**（权重读一次被全 batch 共享），
这正是推理服务做 continuous batching 的根本原因。
