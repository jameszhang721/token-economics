# 从一段文本到 Transformer：token、ID 与 embedding 的逐步观察

> 配套脚本：[`../scripts/inspect_token_pipeline.py`](../scripts/inspect_token_pipeline.py)。本文以 decoder-only 模型 **Qwen2.5-0.5B** 为例；不同模型的切分、特殊 token 和位置编码可能不同。

## 先记住这张图

```text
原始文本（Python 字符串）
    ↓ tokenizer：依据词表和切分规则编码
token（便于观察的片段；不保证是完整的词）
    ↓ 查 tokenizer 的词表
整数 token ID，例如 [id₀, id₁, id₂, …]
    ↓ 模型 embedding 层按 ID 取权重矩阵中的行
初始向量，形状 [batch_size, seq_len, hidden_size]
    ↓ Transformer：结合位置、注意力与前馈网络，逐层处理
上下文化的隐藏状态
    ↓ 语言模型输出层 → logits → 选择下一个 token ID → decode
续写的文字
```

**纠偏**：token 不是向量；token ID 是整数，embedding 才是浮点向量。“embedding 层”是模型中的可训练查表层，不是模型在推理时访问的外部向量数据库。

## 如何运行：先不下载模型，再按需加载模型

在仓库根目录启动 conda 环境。首次在 NVIDIA 机器上按仓库根目录 `environment.yml` 创建环境，版本为 Python 3.9 + PyTorch CUDA 11.8；旧环境不会因为修改配置文件而自动更新，若已有旧版 `token-econ`，先在 README 查看用新名字创建环境的命令：

```powershell
cd D:\zhangjiabao1\Desktop\zhangjiabao\idea-project\token-economics
conda env create -f environment.yml
conda activate token-econ

# 第一阶段：只下载 tokenizer 文件（不下载模型权重，不需要 GPU）
python scripts/inspect_token_pipeline.py --stage tokenizer --text "你好，token！"

# 第二阶段：下载 Qwen2.5-0.5B 模型权重，看查表、前向传播与下一 token
python scripts/inspect_token_pipeline.py --stage model --text "你好，token！"
```

默认端点为 `https://hf-mirror.com`，且尊重事先设置的 `HF_ENDPOINT`。也可用 `--mirror official` 切回官方，或用 `--model D:\模型目录` 读取已经下载到本地的同一模型；完全离线再加 `--local-only`。模型运行会自动选择 CUDA（有 NVIDIA GPU 时），否则用 CPU；第二阶段在 CPU 上也能运行，只是更慢。**Tokenizer 与模型必须来自同一模型目录/版本**，否则 ID 与向量表的行可能对不上。`--stage tokenizer` 仍需要 tokenizer 文件，首次运行可能联网。

下文第 2～3 步的四个 ID 是在本机用 Qwen2.5-0.5B 的 tokenizer **实际运行得到的**。其他输入和不同版本的 tokenizer 可能产生不同结果；向量数值、模型输出维度及概率尚未在本文实测，须运行第二阶段查看。

## 第 1 步：原始文本是什么？

在 Python 中，输入首先是一段字符串：

```python
text = "你好，token！"
print(repr(text))                # '你好，token！'
print(len(text.encode("utf-8"))) # 原文占多少 UTF-8 字节
```

“字符数”“UTF-8 字节数”“词数”“token 数”不是同一回事。一个汉字通常对应多个 UTF-8 字节；它在某个 tokenizer 下可能是一个 token，也可能需要多个 token。不要用“每个 token 等于一个汉字/单词”推算开销。

**脚本对应**：`args.text` 是原始字符串；第 ① 行打印它。

## 第 2 步：tokenizer 如何切分文本？

Tokenizer 在模型之外运行。它读取配套的词表/合并规则，将文本编码为若干**离散单元**。可把 token 直观地理解成“这个 tokenizer 认识的一块文本”，但它未必等于一个完整单词：可能是词的一部分、标点、空格前缀，或者字节级片段。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
text = "你好，token！"
ids = tokenizer.encode(text, add_special_tokens=False)
pieces = tokenizer.convert_ids_to_tokens(ids)
print(list(zip(pieces, ids)))
```

上面示例为解释 API；配套脚本已经在导入 `transformers` **之前**配置了镜像，所以网络受限时请优先运行脚本。本机在 `--stage tokenizer` 下实际得到：

| 位置 | 词表内部表示（原样） | token ID | 单独解码 |
|---:|---|---:|---|
| 0 | `ä½łå¥½` | 108386 | `你好` |
| 1 | `ï¼Į` | 3837 | `，` |
| 2 | `token` | 5839 | `token` |
| 3 | `ï¼ģ` | 6313 | `！` |

因此该样本的 `ids == [108386, 3837, 5839, 6313]`，`T = 4`。中文的内部表示看似乱码，是 byte-level tokenizer 的可视化形式，**不是文本被损坏**。换模型或 tokenizer 版本后以你实际运行的输出为准。`convert_ids_to_tokens` 显示的是**词表中的内部表示**，不一定是人类可读文本；一个 token 单独解码甚至可能出现替代字符。观察恢复的整段文本时，应使用：

```python
print(tokenizer.decode(ids))
```

注意：`add_special_tokens=False` 是为了只观察原文产生的 token。实际聊天模板、BOS/EOS 等**特殊 token**可作为额外 ID 出现；应用中的 token 数因此可能比单独编码文本的 token 数多。

**脚本对应**：`tokenizer.encode` 得到 `token_ids`；`convert_ids_to_tokens` 得到用于展示的 `pieces`；逐项打印片段和 ID，再整体 `decode` 检查。

## 第 3 步：token ID 是什么？存在哪里？

Tokenizer 的词表建立“单元 ↔ 整数编号”的约定。一次输入编码完成后，在程序内表现为整数列表：

```python
# 上面已用该样本实际运行，得到如下 ID；其他输入需由 tokenizer.encode 重新生成。
ids = [108386, 3837, 5839, 6313]
```

交给 PyTorch 模型时，它们通常构成整数张量，形状 `[batch_size, seq_len]`：

```python
import torch
input_ids = torch.tensor([ids], dtype=torch.long, device=device)
# 一段输入：batch_size = 1，seq_len = 这段输入的 token 数
```

**存储位置要区分**：

| 对象 | 常见存储/运行位置 | 用途 |
|---|---|---|
| tokenizer 词表与规则 | 模型配套的 tokenizer 文件；运行时由 tokenizer 进程加载 | 文本与 ID 之间的映射 |
| 本次输入的 token IDs | 进程内存中的整数；前向时为 CPU/GPU 张量 | 告诉模型查哪些行 |
| embedding 权重矩阵 | 模型权重文件（通常包括 `safetensors`），加载后在内存/显存 | 存储各 ID 的**初始**可学习向量 |
| 外部向量数据库 | 可选的独立检索系统 | 存文档等内容的向量，**不是** token embedding 的必经存储 |

磁盘上的模型权重文件与运行时加载的矩阵是一回事的两种形态；并不存在“每次来一个 token 就去向量数据库搜索词向量”这一步。

**脚本对应**：`input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)`。

## 第 4 步：embedding 层怎样查表？

假设词表大小为 `V`，模型隐藏维度为 `d`，输入有 `T` 个 token，模型的 embedding 权重可写成矩阵 `E ∈ ℝ^(V×d)`：

```text
E 的第 0 行 → token ID 0 的初始向量
E 的第 1 行 → token ID 1 的初始向量
…
E 的第 k 行 → token ID k 的初始向量
```

对每个位置的 ID `tᵢ`，取 `xᵢ = E[tᵢ]`。这叫“查表”，本质是**按整数索引读取权重行**，不是查询向量数据库：

```python
embedding_layer = model.get_input_embeddings()
vectors = embedding_layer(input_ids)

print(embedding_layer.weight.shape)   # [V, d]
print(input_ids.shape)                # [1, T]
print(vectors.shape)                  # [1, T, d]
assert torch.equal(vectors[0, 0], embedding_layer.weight[input_ids[0, 0]])
```

相同 token ID 在同一套权重下，**初始向量**是相同的；之后 Transformer 为它算出的**上下文表示**会随上下文改变。这区分了“词表里的向量”和“模型读懂句子之后的表示”。这一步无需训练，只是在查看已有模型权重。

**脚本对应**：`model.get_input_embeddings()`、`vectors = embedding_layer(input_ids)` 与断言；打印实际矩阵/向量形状和第一个向量的前 8 维。

## 第 5 步：向量如何进入 Transformer？位置在哪里？

光知道“有哪些 token”还不够，模型也要区分它们在序列中的位置。不同架构的实现不同：有的模型在输入向量上加位置向量；本文使用的 **Qwen2.5 在注意力计算中使用 RoPE（旋转位置编码）**，所以不能笼统地说“给 embedding 拼一个序号”，也不能只凭上一步的向量值看出位置信息。

推理时把 ID 传给模型，模型内部执行 embedding、各层注意力和前馈网络等计算：

```python
model.eval()
with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    contextual = outputs.hidden_states[-1]
    logits = outputs.logits

print(contextual.shape)  # [1, T, d]：最后一层的上下文化表示
print(logits.shape)      # [1, T, V]：各位置对候选下一个 token 的分数
```

Decoder-only LLM 使用**因果注意力**：位置 `i` 只能利用当前位置及其前面的信息。以末位置 `logits[0, -1]` 为例，经过 softmax 可得到“接下来各候选 token ID 的概率”；选出一个 ID，再用 tokenizer 解码成文字。连续生成时通常还会用 **KV cache** 暂存注意力所需的中间结果，它不是 token embedding 表，更不是向量数据库。

**脚本对应**：`model(..., output_hidden_states=True)` 与 `torch.softmax(outputs.logits[0, -1].float(), dim=-1)`；打印最后一层表示、logits 的形状和 5 个候选 token。这里的输出是**当前文本之后**的候选续写，不是对输入的逐词翻译。

## 一次运行时该怎样对照输出？

以下 ①～③ 的片段、ID 和解码来自本机 tokenizer 阶段的实际输出；④～⑥ 是**尚未执行模型阶段**的形状说明，不能当成实测向量或概率：

```text
① 原始文本: '你好，token！'
② tokenizer 切分后的片段 / token ID:
   位置 0: 原始词表表示='ä½łå¥½'  ID=108386  单独解码='你好'
   位置 1: 原始词表表示='ï¼Į'     ID=3837    单独解码='，'
   位置 2: 原始词表表示='token'   ID=5839    单独解码='token'
   位置 3: 原始词表表示='ï¼ģ'     ID=6313    单独解码='！'
③ token IDs: [108386, 3837, 5839, 6313]
   整体解码: '你好，token！'
④ 输入 ID 张量: shape=(1, 4)                  ← 按上述四个 ID 推出
   模型 embedding 表: shape=(V, d)            ← 需运行模型阶段读出 V、d
   查表后的向量: shape=(1, 4, d)
⑤ Transformer 最后一层输出: shape=(1, 4, d)
⑥ 末位置预测下一个 token: logits shape=(1, 4, V)
```

`V` 和 `d` 由**模型配置**决定；运行第二阶段后请对照它打印的具体值，不要抄形状变量填实验记录。此实验只做一次前向，不会训练模型、写入向量数据库或生成完整回答。

## 与向量数据库 / RAG 的关系

RAG 中另有一条可选流程：`文档 → 独立 embedding 模型 → 文档向量 → 向量数据库`。提问时将问题编码成向量、检索相关文档，取回**原文**拼到提示词中；LLM 然后仍按本页描述的 `文本 → token ID → 模型内部 embedding → Transformer` 运行。一般情况下，检索库里的文档向量**不会被当作 token ID 或 token embedding 直接喂进 LLM**。

## 自测（看完代码后回答）

1. 对同一段文本，把 `add_special_tokens=False` 改为 `True`，token 数是否变化？为什么要区分原文 token 和聊天模板 token？
2. 若 `input_ids` 是 `[1, T]`，为什么查表后的 `vectors` 是 `[1, T, d]`？表的第一维 `V` 去哪里了？
3. 同一个 ID 出现在句中不同位置时，初始 embedding 会不会变化？最后一层隐藏状态会不会变化？
4. 为什么 `logits[0, -1]` 对应的是“下一个 token”，而不是已经输入的最后一个 token 的文字？
5. 如果加载 A 模型的 tokenizer、B 模型的权重，即使程序不报错，实验结论可信吗？

> 答题提示：先跑第一阶段亲眼看 ID，再跑第二阶段看张量形状；若暂时无法下载模型，第 1～3 步仍可以独立完成。
