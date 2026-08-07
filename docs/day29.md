# 学习单元 Day 29：预分配 KV Cache 与原地写入

> [`roadmap.md`](./roadmap.md)：**学习单元 Day 29**｜学习日 20｜预算 5 小时

昨天的 `DynamicCache` 正确但低效——每轮 `concat` 都要重新分配和复制。
今天做 `StaticCache`：一次分配到位，按位置原地写入。

Notion 的硬要求：

> **禁止每轮重新分配整个缓存。**

## 进度与落点

| 进度 | 任务 | 落点 |
| --- | --- | --- |
| ⬜ | 任务 1：量化 concat 的代价 | `notebooks/03_cache_growth.ipynb` |
| ⬜ | 任务 2：实现 `StaticCache` | `src/mini_transformer/cache.py` |
| ⬜ | 任务 3：有效长度管理与 view 返回 | 同上 |
| ⬜ | 任务 4：与 `DynamicCache` 等价性测试 | `tests/test_cache.py` |
| ⬜ | 任务 5：测量分配次数与拷贝量 | `benchmarks/bench_cache_memory.py` |

## 学习资料

| 必读 | 选读 | 阅读目标 |
|------|------|---------|
| [HF Static Cache](https://huggingface.co/docs/transformers/kv_cache#fixed-size-cache) | [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) | 设计预分配缓存和按 position 原地写入，避免逐步 concat |

---

## 任务 1：量化 concat 的代价（40 分钟）

先证明问题存在，再解决它（和 Day 26 同样的方法论）。

### `torch.cat` 到底做了什么

**它总是分配一个新张量，并复制两边的全部数据。** 没有「原地追加」这回事——
PyTorch 张量的内存是连续块，末尾没有预留空间。

所以第 `t` 轮 concat 要：

```text
分配   [B, Hkv, S+t, Dh] 的新内存
复制   S+t-1 个位置的历史 + 1 个新位置
释放   旧张量
```

### 累计拷贝量是 `O(n²)`

生成 `n` 个 token，每层每轮复制约 `S+t` 个位置：

```text
累计复制 = Σ(S + t) ≈ n·S + n²/2      每层每个 K/V
× 2 (K和V) × L 层
```

**注意这和 Day 26 那个「重复计算」的 `O(n²)` 是两笔不同的账。**
KV Cache 消掉了重复**计算**，但 `DynamicCache` 又引入了重复**拷贝**。
虽然拷贝比计算便宜得多（没有矩阵乘），但量级同样是 `O(n²)`。

### 实测

在 notebook 里测三件事：

1. **分配次数**：CUDA 上用 `torch.cuda.memory_stats()["allocation.all.count"]`
   前后差；CPU 上可以数 `cat` 调用次数（`2 × L × n`）
2. **每轮 concat 的耗时**：应该随轮次**线性增长**
3. **峰值显存**：concat 瞬间**同时存在**新旧两份张量，
   峰值约是稳态的 2 倍（对单层而言）

第 3 条是很多人忽略的：`DynamicCache` 的峰值显存高于它「应该」占的空间。

把「每轮 concat 耗时 vs 轮次」画成曲线，和 Day 26 那张图放一起。

## 任务 2：实现 `StaticCache`（100 分钟）

### 数据结构

```python
class StaticCache:
    def __init__(self, num_layers, max_batch_size, max_seq_len,
                 num_kv_heads, head_dim, *, dtype, device):
        shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.keys   = [torch.zeros(shape, dtype=dtype, device=device)
                       for _ in range(num_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device)
                       for _ in range(num_layers)]
        self.cur_len = 0
```

**一次性分配 `2 × L` 个固定形状张量，之后再也不分配。**

用 `torch.zeros` 而不是 `torch.empty`——`empty` 是未初始化内存，
万一有 bug 读到了未写区域，`zeros` 的表现是确定的（贡献 0 注意力），
`empty` 可能是 NaN 或巨大值，调试起来痛苦得多。**用 `zeros` 换可调试性。**

### 原地写入

```python
def update(self, layer_idx, new_k, new_v, cache_position):
    self.keys[layer_idx][:, :, cache_position, :] = new_k
    self.values[layer_idx][:, :, cache_position, :] = new_v
    ...
```

`cache_position` 可以是 int、slice 或 `[S]` 的 long 张量（Prefill 时写多个位置）。

**用 `index_copy_` 或直接切片赋值都可以**，关键是不产生新张量。
验证方法：写入前后 `tensor.data_ptr()` 不变。

### off-by-one：今天最大的坑

Notion 明确把「Cache 写入位置 off-by-one」列为必查错误。三个高危点：

| 场景 | 正确 | 错误写法 | 后果 |
|---|---|---|---|
| Prefill 写 `S` 个位置 | `[0 : S]` | `[0 : S-1]` 或 `[1 : S+1]` | 少写/错位一个位置 |
| Decode 第 `t` 步 | `cur_len`（写入后 `+1`） | `cur_len - 1`（覆盖上一个） | 覆盖历史，输出退化 |
| 返回有效区间 | `[: cur_len]` | `[: cur_len - 1]` 或 `[: cur_len + 1]` | 漏掉最新 token / 读到未写的 0 |

**防御手段：让「写入」和「递增」在同一个方法里完成**，
调用方永远不手动管 `cur_len`：

```python
def update(self, layer_idx, new_k, new_v, cache_position):
    # 写入
    ...
    # 只在最后一层更新长度，避免 L 次重复递增
    if layer_idx == self.num_layers - 1:
        self.cur_len = int(cache_position.max()) + 1 if torch.is_tensor(cache_position) \
                       else cache_position + 1
    return self.get_kv(layer_idx)
```

注意 `if layer_idx == num_layers - 1` 这个判断——**每轮 `L` 层都会调 `update`，
但 `cur_len` 只该增加一次。** 忘了这一点会让长度增长 `L` 倍，
是个特别容易犯又特别难查的错。

（或者更干净：把 `cur_len` 的更新完全移出 `update`，
由 `decode()` 在处理完所有层之后调一次 `cache.advance()`。推荐这种。）

## 任务 3：有效长度管理与 view 返回（60 分钟）

### 必须返回有效 view，不能返回整个 buffer

```python
def get_kv(self, layer_idx):
    return (self.keys[layer_idx][:, :, : self.cur_len, :],
            self.values[layer_idx][:, :, : self.cur_len, :])
```

**如果返回整个 `[B, Hkv, max_S, Dh]`**，attention 会对
`max_S - cur_len` 个**未写入的零向量**计算注意力。

后果分析（值得想清楚）：零向量的 `K` 点积出的 score 是 0，
`exp(0) = 1`，**不是 0**！所以这些位置会分到可观的注意力权重，
而它们的 `V` 是零向量——等于给输出掺入了一个「什么都不是」的分量。

模型不会崩，输出也不会 NaN，只是**质量下降**。这类 bug 极其难发现。

**两个解法**，都要理解：

| 解法 | 做法 | 优劣 |
|---|---|---|
| **返回 view** | `[:, :, :cur_len, :]` | 简单，Decode 时 `Skv` 变化 |
| **mask 掉未写区** | 返回全量 + 把 `>= cur_len` 的位置加 `-inf` | `Skv` 固定，`torch.compile` 友好 |

第二种是 HF `StaticCache` 的做法，也是为什么它能配合 `torch.compile`
（Day 38 会验证）——**形状固定是编译的前提**。

我们先用第一种（简单、正确），在笔记里记下第二种的动机。

### 切片是 view 不是 copy

```python
t = torch.zeros(2, 4, 100, 8)
v = t[:, :, :10, :]
v.data_ptr() == t.data_ptr()      # True，共享内存
```

这是「预分配 + view」能省开销的技术基础。
但要小心：**这个 view 是可写的**，写它会改到 buffer。
本项目里这正是我们想要的，但要清楚这一点。

### 容量检查

```python
if cur_len + new_len > max_seq_len:
    raise ValueError(
        f"Cache 容量不足：已用 {cur_len}，本轮 {new_len}，上限 {max_seq_len}。"
        f"请增大 max_seq_len 或减少 max_new_tokens。"
    )
```

预分配的代价就是有上限。**明确报错**比静默截断或越界好——
沿用 Day 01～04 的习惯，错误信息带上所有实际数值。

## 任务 4：与 `DynamicCache` 等价性测试（60 分钟）

`StaticCache` 是 `DynamicCache` 的优化版，**结果必须完全相同**。
这是今天最重要的测试。

`tests/test_cache.py` 新增：

1. **同一序列，两种 Cache 的每层 K/V 逐比特相同**（`rtol=0, atol=0`）
2. **每步 logits 逐比特相同**
3. **最终 token 序列相同**
4. **`data_ptr` 不变**：`update` 前后 buffer 的指针相同（证明真的原地）
5. **分配次数**：整个生成过程中 CUDA 分配次数不随生成步数增长
6. `cur_len` 每轮**只增加 1**（不是 `L`）——**专门防那个 bug**
7. 超出 `max_seq_len` 时报错，错误信息含三个实际数值
8. `get_kv` 返回的形状是 `[B, Hkv, cur_len, Dh]`，**不是 `max_seq_len`**
9. **未写区域不参与计算**：把 buffer 的未写区填成巨大值（比如 `1e6`），
   结果**必须不变**

第 9 条是验证任务 3 那个陷阱的**唯一可靠方法**。
如果填了 `1e6` 结果变了，说明你返回了整个 buffer。
`zeros` 初始化会掩盖这个 bug（零向量的影响小到看不出来），
故意填大值才能暴露它。**这条一定要写。**

第 6 条也很关键——`cur_len` 增长 `L` 倍的 bug 在小配置（`L=2`）下
表现为「长度翻倍」，容易被误认为是别的问题。

## 任务 5：测量分配次数与拷贝量（40 分钟）

写 `benchmarks/bench_cache_memory.py`，对比两种 Cache：

| 指标 | DynamicCache | StaticCache |
|---|---|---|
| 总分配次数 | `~2·L·n` | `2·L`（只在初始化） |
| 累计拷贝字节数 | `O(n²)` | `O(n)` |
| 峰值显存 | 高（concat 瞬间双份） | 等于预分配大小 |
| 稳态显存 | 随 `n` 增长 | **固定**（一开始就占满） |

最后一行是 `StaticCache` 的**代价**：它一开始就占满 `max_seq_len` 的空间，
即使实际只用了几十个 token。

**这是一个真实的权衡**，在笔记里写清楚：

- 生成长度**已知或有上限** → `StaticCache`（可预测、可编译）
- 生成长度**差异极大**、要服务很多短请求 → 预分配会浪费大量显存

真实推理引擎（vLLM 等）的解法是 **PagedAttention**——
按固定大小的块分配，兼得两者优点。本项目不实现，但在笔记里记一句
「下一阶段方向」，这也是 Day 40 那份「下一阶段瓶颈清单」的素材。

---

## 过关标准

- [ ] 量化了 `concat` 的分配次数、累计拷贝量和峰值显存
- [ ] `StaticCache` 一次分配，之后不再分配（`data_ptr` 不变）
- [ ] 用 `zeros` 而非 `empty` 初始化，且知道为什么
- [ ] `cur_len` 每轮只增加 1（不随层数放大）
- [ ] `get_kv` 返回有效 view，形状是 `cur_len` 而非 `max_seq_len`
- [ ] **未写区域填巨大值后结果不变**（第 9 条测试）
- [ ] 与 `DynamicCache` 的 K/V、logits、token 序列全部逐比特相同
- [ ] 容量超限时报错且信息完整
- [ ] 记录了预分配的显存代价与 PagedAttention 的方向
- [ ] `pytest -q` 全部通过

---

## 今日最重要的面试式问题

**预分配 KV Cache 比 concat 好在哪，代价是什么？**

**好处**：① 只分配一次，避免 `2·L·n` 次分配；
② 累计拷贝从 `O(n²)` 降到 `O(n)`；
③ 形状固定，是 `torch.compile` / CUDA Graph 的前提；
④ 峰值显存可预测（`concat` 瞬间会同时存在新旧两份）。

**代价**：一开始就占满 `max_seq_len` 的空间。服务大量短请求时浪费严重，
而且生成长度不能超过预设上限。真实引擎用 PagedAttention（按块分配）解决这个矛盾。

追问：**预分配的 Cache 返回给 attention 时要注意什么？**

**必须返回 `[:, :, :cur_len, :]` 的有效 view，不能返回整个 buffer。**

否则 attention 会对未写入的零向量计算注意力。关键是零向量的点积是 0，
而 `exp(0) = 1` **不为 0**——这些位置会分到可观的注意力权重，
把零向量的 `V` 掺进输出。模型不崩、不 NaN，只是质量下降，极难发现。

要测出这个 bug，得**故意把未写区填成巨大值**再看结果是否变化；
用 `zeros` 初始化会把它掩盖掉。
