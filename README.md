# token-economics

八周研究:**token 是一种编码,它编得好不好、贵不贵、丢了多少信息。**

所有结论只靠三把尺子说话:**BPB**(bits per byte)、**fertility**(词→token 数)、**overhead gap**(BPB − 经验熵)。

## 环境(conda + GPU)

```bash
conda env create -f environment.yml
conda activate token-econ

# 验证 GPU 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> `pytorch-cuda=12.1` 需匹配你的驱动。驱动较老改成 `11.8`;`nvidia-smi` 右上角 CUDA Version 是上限。

### 在 NVIDIA 机器上复现(模型计算在这里跑)

本仓库的开发/采样可在任意机器完成;**算 BPB 需要 NVIDIA GPU**。在 GPU 机器上:

```bash
git clone https://github.com/zhangjiabao-nudt/token-economics.git
cd token-economics
conda env create -f environment.yml && conda activate token-econ

# 必须看到 CUDA: True
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -m src.metrics --selftest       # 纯数学,先确认尺子可用
python scripts/download_corpora.py     # 采样六语料(可在任意机器先跑)
python scripts/compression_vs_model.py # GPU:model BPB 应 < zstd
```

`src/metrics.py` 会自动选 `cuda`(可用时)否则 `cpu` —— 无需改代码。
若 `CUDA: False`,检查 `nvidia-smi` 是否可用、`pytorch-cuda` 版本是否匹配驱动。

## 第 0 周:先造尺子(照这个顺序跑)

```bash
# 1. 采样六语料,各 10MB,锁 seed 存进 corpora/
python scripts/download_corpora.py

# 2. 三把尺子的 sanity check(不需要 GPU,先跑通)
python -m src.metrics --selftest

# 3. 第 1 周动手项:zstd vs 小模型,比 bits/char(需要 GPU)
python scripts/compression_vs_model.py

# 4. 第 3 周对照表:三种 tokenizer × 六语言的 BPB/fertility/成本
python scripts/week3_cost_table.py
```

## 三把尺子(src/metrics.py)

| 指标 | 定义 | 为什么用它 |
|---|---|---|
| `entropy(corpus, n)` | 经验 n-gram 熵,bits/char | BPB 的理论下界参照 |
| `bpb(model, tok, text)` | 总交叉熵(nats) / (ln2 × **字节数**) | 分母是字节数,跨 tokenizer 唯一公平 |
| `fertility(tok, text)` | 词→token 数的分布(p90/p99/max) | 成本黑洞在尾部,看均值会骗自己 |

## 目录

```
corpora/     采样语料(gitignore,靠脚本复现)
src/         metrics.py — 全程复用的尺子
scripts/     每周的动手脚本
results/     图表与 CSV 输出
notes/       每周一页:核心主张 / 假设 / 能否证伪
```

## 每周笔记纪律

`notes/weekNN.md` 每周只写三行:**本周核心主张 / 它的假设 / 我能不能证伪它**。八周后这八页就是研究提纲。
