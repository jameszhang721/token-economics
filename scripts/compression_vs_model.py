"""第 1 周动手项:压缩 = 建模。

对同一份语料,比较:
  (a) zstd / gzip 的 bits/char  —— 经典通用压缩器
  (b) 小语言模型的 BPB          —— 学出来的"压缩器"

预期:两者同量级,且模型优于 zstd。对不上就查 bug(多半是分母用错、
或没在 GPU 上、或文本编码问题)。

用法:
    python scripts/compression_vs_model.py
    python scripts/compression_vs_model.py --langs en zh --sample-kb 512
"""
import argparse
import gzip
import math
import os
import sys

import zstandard as zstd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hf_config import setup_hf_endpoint, report as hf_report  # noqa: E402
from src.metrics import bpb, entropy  # noqa: E402

CORPORA = os.path.join(os.path.dirname(__file__), "..", "corpora")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def compressor_bits_per_char(text: str) -> dict:
    raw = text.encode("utf-8")
    n_chars = len(text)
    gz = len(gzip.compress(raw, 9))
    zs = len(zstd.ZstdCompressor(level=19).compress(raw))
    return {
        "gzip_bpc": gz * 8 / n_chars,
        "zstd_bpc": zs * 8 / n_chars,
        "n_chars": n_chars,
        "n_bytes": len(raw),
    }


def load_sample(lang: str, sample_kb: int) -> str:
    path = os.path.join(CORPORA, f"{lang}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在 —— 先跑 download_corpora.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 取前 sample_kb 作评测样本(BPB 前向开销大,不必全跑)
    budget = sample_kb * 1024
    if len(text.encode("utf-8")) > budget:
        # 按字节裁剪但不切碎多字节字符
        text = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample-kb", type=int, default=512)
    ap.add_argument("--mirror", default="__unset__",
                    help="HF 端点。默认 hf-mirror.com;official 用官方;或自定义 URL")
    args = ap.parse_args()

    setup_hf_endpoint(None if args.mirror == "official" else args.mirror)
    hf_report()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    if device == "cpu":
        print("  警告:未检测到 GPU,0.5B 模型在 CPU 上会很慢。")

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    print(f"\n{'lang':<6}{'H1(bpc)':>10}{'gzip':>10}{'zstd':>10}{'model BPB':>12}")
    print("-" * 48)
    for lang in args.langs:
        text = load_sample(lang, args.sample_kb)
        comp = compressor_bits_per_char(text)
        h1 = entropy(text, 1)
        model_bpb = bpb(model, tok, text, device=device)
        print(f"{lang:<6}{h1:>10.3f}{comp['gzip_bpc']:>10.3f}"
              f"{comp['zstd_bpc']:>10.3f}{model_bpb:>12.3f}")

    print("\n读法:")
    print("  H1 是单字符经验熵(粗下界);gzip/zstd 是通用压缩器;")
    print("  model BPB 应低于 zstd —— 学出来的压缩器打败通用压缩器,")
    print("  这就是 '语言建模 = 压缩' 的实证。若 model > zstd,查 bpb() 分母。")


if __name__ == "__main__":
    main()
