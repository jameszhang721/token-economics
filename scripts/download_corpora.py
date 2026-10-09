"""采样六种语言各 ~10MB,锁 seed 存进 corpora/。

主线纪律:所有对照必须锁定同一份数据 —— 语料一变,BPB 全废。
本脚本把采样固定下来,后面所有周都读 corpora/*.txt。

用法:
    python scripts/download_corpora.py
    python scripts/download_corpora.py --mb 5        # 想快一点先跑 5MB
    python scripts/download_corpora.py --source wikipedia
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hf_config import setup_hf_endpoint, report as hf_report  # noqa: E402

# zh 中文, en 英文, ar 阿拉伯语, fi 芬兰语, tr 土耳其语, ja 日语
# —— 形态丰富(fi/tr)和非拉丁(zh/ar/ja)是 fertility 分歧最大的地方
LANGS = ["zh", "en", "ar", "fi", "tr", "ja"]

WIKI_CONFIG = {
    "zh": "20231101.zh", "en": "20231101.en", "ar": "20231101.ar",
    "fi": "20231101.fi", "tr": "20231101.tr", "ja": "20231101.ja",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpora")
SEED = 42


def sample_wikipedia(lang: str, target_bytes: int) -> str:
    """流式拉 wikimedia/wikipedia,shuffle 后累计文章正文到 target_bytes。"""
    from datasets import load_dataset

    cfg = WIKI_CONFIG[lang]
    ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    chunks, total = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:  # 跳过太短的残条
            continue
        chunks.append(t)
        total += len(t.encode("utf-8"))
        if total >= target_bytes:
            break
    return "\n\n".join(chunks)


def sample_oscar(lang: str, target_bytes: int) -> str:
    """备用源:OSCAR-2301(需 HF 登录并接受条款)。"""
    from datasets import load_dataset

    ds = load_dataset("oscar-corpus/OSCAR-2301", lang,
                      split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    chunks, total = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:
            continue
        chunks.append(t)
        total += len(t.encode("utf-8"))
        if total >= target_bytes:
            break
    return "\n\n".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=10, help="每种语言的目标 MB 数")
    ap.add_argument("--source", choices=["wikipedia", "oscar"], default="wikipedia")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--mirror", default="__unset__",
                    help="HF 端点。默认走 hf-mirror.com;传 official 用官方;或给自定义 URL")
    args = ap.parse_args()

    endpoint = None if args.mirror == "official" else args.mirror
    setup_hf_endpoint(endpoint)
    hf_report()

    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    target = args.mb * 1024 * 1024
    sampler = sample_wikipedia if args.source == "wikipedia" else sample_oscar

    for lang in args.langs:
        out_path = os.path.join(OUT_DIR, f"{lang}.txt")
        if os.path.exists(out_path):
            print(f"[skip] {lang}: 已存在 {out_path}")
            continue
        print(f"[..]  采样 {lang} 目标 {args.mb}MB (source={args.source}) ...")
        text = sampler(lang, target)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        got = os.path.getsize(out_path) / 1024 / 1024
        print(f"[ok]  {lang}: {got:.1f}MB -> {out_path}")

    print("\n采样完成。seed 已锁定,复现时删掉 corpora/ 再跑即可得到同一份数据。")


if __name__ == "__main__":
    main()
