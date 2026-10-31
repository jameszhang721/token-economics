# token-economics

八周研究:**token 是一种编码,它编得好不好、贵不贵、丢了多少信息。**

所有结论只靠三把尺子说话:**BPB**(bits per byte)、**fertility**(词→token 数)、**overhead gap**(BPB − 经验熵)。

入门配套材料：[原始文本 → token → ID → embedding → Transformer（逐步讲解与运行示例）](docs/text_to_transformer.md)。

## 环境（Python 3.9 + PyTorch CUDA 11.8）

在装有 NVIDIA GPU、且驱动支持 CUDA 11.8 的机器上：

```bash
nvidia-smi                                # 先确认驱动和 GPU 可用
conda env create -f environment.yml
conda activate token-econ
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无 NVIDIA GPU')"
```

预期 Python 为 `3.9.x`，`torch.version.cuda` 为 `11.8`，GPU 机器上 `torch.cuda.is_available()` 为 `True`。`nvidia-smi` 显示的 CUDA Version 是**驱动能支持的最高 CUDA 版本**，并非本环境实际使用的 CUDA 版本；GPU 驱动需要足够新。Windows 无 NVIDIA GPU 的本机上，即使安装了 CUDA 版 PyTorch，`torch.cuda.is_available()` 仍会是 `False`。

若本机已经有旧的 `token-econ`（Python 3.11 / CUDA 12.1），**创建新环境，不要覆盖旧环境**：

```bash
conda env create -n token-econ-py39-cu118 -f environment.yml
conda activate token-econ-py39-cu118
```

只修改 `environment.yml` 不会自动改变已经建好的环境；以下其他命令均在**当前激活的环境**里运行。

### 在 NVIDIA 机器上复现(模型计算在这里跑)

本仓库的开发/采样可在任意机器完成；若希望用 CUDA 加速模型计算，需要 NVIDIA GPU。在 GPU 机器上：

```bash
git clone https://github.com/jameszhang721/token-economics.git
cd token-economics
conda env create -f environment.yml && conda activate token-econ

# 必须看到 Python 3.9、PyTorch CUDA 11.8 且 CUDA: True
python -c "import sys, torch; print(sys.version); print('PyTorch CUDA:', torch.version.cuda, '可用:', torch.cuda.is_available())"

python -m src.metrics --selftest       # 纯数学,先确认尺子可用
python scripts/download_corpora.py     # 采样六语料(可在任意机器先跑)
python scripts/compression_vs_model.py # GPU:model BPB 应 < zstd
```

`src/metrics.py` 会自动选 `cuda`(可用时)否则 `cpu` —— 无需改代码。
若 `CUDA: False`,检查 `nvidia-smi` 是否可用、`pytorch-cuda` 版本是否匹配驱动。

## 访问不了 HuggingFace 怎么办

国内网络常连不上 `huggingface.co`。三个脚本都已内置 **HF 镜像**支持,默认走 `hf-mirror.com`,
无需手动设环境变量。`src/hf_config.py` 在 import 库之前设好 `HF_ENDPOINT`。

**方案 A(默认,推荐):hf-mirror.com 镜像**
```bash
python scripts/download_corpora.py        # 默认即用镜像
python scripts/compression_vs_model.py
```
需要切回官方或自定义时:
```bash
python scripts/download_corpora.py --mirror official            # 走 huggingface.co
python scripts/download_corpora.py --mirror https://your-url    # 自定义端点
```
或全局设一次(对所有脚本和手动下载都生效):
```bash
# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
# bash
export HF_ENDPOINT=https://hf-mirror.com
```
装了 `huggingface_hub` 后还可用镜像版 CLI 预下载模型:
```bash
pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B
```

**方案 B:ModelScope(魔搭,阿里)—— 模型走这条最稳**
Qwen 系列在魔搭有官方镜像,`modelscope.cn` 国内直连快。模型名换成魔搭 ID:
```bash
pip install modelscope
python scripts/compression_vs_model.py --model Qwen/Qwen2.5-0.5B --mirror official
# 或先从魔搭下载到本地,再用本地路径:
modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ./models/qwen2.5-0.5b
python scripts/compression_vs_model.py --model ./models/qwen2.5-0.5b --mirror official
```

**方案 C:语料绕开 HF —— 直接下维基原始 dump**
`dumps.wikimedia.org` 国内可直连。要彻底不依赖 HF,可改成从维基 dump 采样
(脚本已支持 `--source wikipedia`;若镜像也慢,手动下 dump 后本地解析)。

> Llama-3 tokenizer 是 gated repo,镜像也需授权 token。跑不了就先跳过那一列,
> cl100k(tiktoken,不走 HF)和 Qwen2.5 两列不受影响。

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
