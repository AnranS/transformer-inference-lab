# Tokenizer 常见问题

## Token 与汉字、单词是否一一对应？

**不是。**

Token 是 tokenizer 词表里的最小单元，不等于自然语言里的「字」或「词」。

| 情况 | 例子 |
|------|------|
| 一个英文单词 → 多个 token | `Transformer` → `transform` + `##er`（WordPiece） |
| 一个汉字 → 一个 token | 部分中文 tokenizer 对常见字如此 |
| 一个汉字 → 多个 token | 罕见字、生僻字可能被拆成字节 / 子词 |
| 多个字 / 词 → 一个 token | 高频短语有时被合并成单个 token |

结论：token 是为模型服务的离散符号，和语言学上的字/词没有强制一一对应。

---

## Token ID 的数值大小有没有语义？

**没有。**

`input_ids` 里的数字只是词表里的**查表下标**，不表示重要性、频率或语义远近。

- `101` 并不比 `50`「更大」或「更重要」
- `2478` 和 `2479` 相邻，也不代表语义相近
- 语义由 embedding 向量承载，不由 ID 数值本身承载

可以把 ID 想成字典页码：页码大小无意义，翻到那一页拿到的词向量才有意义。

---

## `tokenize` 和 `encode` 有什么区别？

都以「分词」为核心，但返回内容不同：

| 方法 | 典型返回 | 作用 |
|------|----------|------|
| `tokenize(text)` | 字符串列表，如 `['using', 'a', 'transform', '##er', ...]` | 看文本被怎么切 |
| `encode(text)` / `__call__` | 整数 ID 列表（或带 `input_ids` 等的字典） | 真正喂给模型的输入 |

常见对应关系：

```text
文本  --tokenize-->  token 字符串  --convert_tokens_to_ids-->  token ids
文本  -------------- encode / __call__ ---------------->  token ids
```

补充：

- `tokenizer(text)`（`__call__`）通常还会附带 `attention_mask`、`token_type_ids` 等
- `encode` 一般只关心 ID；`__call__` 更适合作为模型输入的标准入口
- `decode(ids)` 是反向：ID → 可读文本

---

## `input_ids` 和 `attention_mask` 分别表示什么？

### `input_ids`

每个 token 在词表中的整数编号，是模型的主输入。

```text
"Using a Transformer network is simple"
→ [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102]
```

（BERT 还会自动加 `[CLS]=101`、`[SEP]=102`。）

### `attention_mask`

标记哪些位置是真实内容、哪些是 padding：

- `1`：有效 token，参与 attention
- `0`：padding，计算时应被忽略

单句、无 padding 时，mask 往往全是 `1`。batch 里长短不一时，短句补 PAD，对应位置 mask 为 `0`。

---

## PAD、BOS、EOS、UNK 各自解决什么问题？

| 特殊 token | 常见写法 | 解决的问题 |
|------------|----------|------------|
| **PAD** | `[PAD]` / `<pad>` | batch 对齐：短序列补齐到同一长度，配合 `attention_mask` 忽略填充位 |
| **BOS** | `<s>` / `<bos>` / 有时用 `[CLS]` | 标记序列开始，给解码/生成一个明确起点 |
| **EOS** | `</s>` / `<eos>` / 有时用 `[SEP]` | 标记序列结束，生成时作为停止信号 |
| **UNK** | `[UNK]` / `<unk>` | 词表外（OOV）字符或子词的兜底；无法表示时落到 UNK |

注意：不同模型命名不完全统一。例如 BERT 常用 `[CLS]` / `[SEP]`，不一定显式叫 BOS/EOS；GPT-2 早期甚至没有专用 PAD。要以具体 tokenizer 的 `special_tokens_map` 为准。

---

## 为什么两个 tokenizer 处理同一句中文会产生不同数量的 token？

因为**切分算法、词表、训练语料**都不同，同一句中文会被映射到不同的离散序列。

常见原因：

1. **算法不同**：WordPiece（BERT）、BPE（GPT）、SentencePiece/Unigram（T5、很多多语言模型）切法不同  
2. **词表不同**：词表大小、是否包含整字/整词、中文覆盖度不同  
3. **预分词规则不同**：是否按空格、标点、字符边界先切一刀  
4. **特殊 token 策略不同**：是否自动加 BOS/EOS/CLS/SEP，会额外增减长度  
5. **未知字处理不同**：有的拆成字节级（byte-level BPE），有的落成 UNK，token 数差很多  

直观例子（示意）：

```text
句子：今天天气很好

tokenizer A（偏整字） → [今天, 天气, 很, 好]          → 4 tokens
tokenizer B（偏细切） → [今, 天, 天, 气, 很, 好]      → 6 tokens
tokenizer C（含短语） → [今天天气, 很好]                → 2 tokens
```

因此比较「序列长度 / 上下文窗口」时，必须指定是**哪一个 tokenizer** 下的 token，不能直接按汉字个数或英文单词数估算。

---

## BPE、WordPiece、SentencePiece 的共同目标是什么？

**用有限词表覆盖开放文本，并尽量降低未知词（OOV）问题。**

它们都不追求「一个汉字/单词 = 一个 token」，而是把文本切成可复用的子词单元：常见词保持完整，罕见词拆开。差别在具体算法与实现细节，今天不必深入训练过程。

---

## Padding side：left vs right

`padding_side` 决定 PAD 补在序列哪一侧：

| `padding_side` | 短句形态（示意） | 常见场景 |
|----------------|------------------|----------|
| `"right"` | `[有效 tokens..., PAD, PAD]` | BERT 类编码 / 训练默认 |
| `"left"` | `[PAD, PAD, ..., 有效 tokens]` | 因果 LM 生成时 batch 推理（让真实 token 靠右对齐） |

对应地，`attention_mask` 的 `1` 会跟着有效 token 移动：右侧 padding 时 `1` 在左；左侧 padding 时 `1` 在右。

**关键结论：** Padding 改变批处理中的 token 位置和 attention mask，但不应改变原始文本的有效 token。  
（用 `mask == 1` 取出的 `input_ids`，在 left/right 两种设置下应相同。）
