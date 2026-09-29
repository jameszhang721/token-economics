"""第 3 周重头戏:token 正式变成钱。

同一份语料,用三种 tokenizer 分别算:
  - fertility 均值 / p90(尾部成本)
  - 每 1000 字符的 token 数
  - 折算 "美元 / 百万字符"(按可配置的 API 单价)

产出对照表 -> results/week3_cost_table.csv,并打印到终端。
你会看到某些语言比英文贵 3–5 倍。原样记下,这是后面所有论证的弹药。

用法:
    python scripts/week3_cost_table.py
    python scripts/week3_cost_table.py --price-per-1m 0.15
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import fertility  # noqa: E402

CORPORA = os.path.join(os.path.dirname(__file__), "..", "corpora")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
LANGS = ["zh", "en", "ar", "fi", "tr", "ja"]

# 三种代表性 tokenizer。cl100k 用 tiktoken;另两个用 HF。
TOKENIZERS = {
    "cl100k (GPT-4)": ("tiktoken", "cl100k_base"),
    "Llama-3 (128k)": ("hf", "meta-llama/Meta-Llama-3-8B"),
    "Qwen2.5 (151k)": ("hf", "Qwen/Qwen2.5-0.5B"),
}


class TiktokenWrap:
    """让 tiktoken 的接口和 HF tokenizer 的 .encode(add_special_tokens=) 对齐。"""
    def __init__(self, name):
        import tiktoken
        self.enc = tiktoken.get_encoding(name)

    def encode(self, s, add_special_tokens=False):
        return self.enc.encode(s, disallowed_special=())


def load_tokenizer(kind, name):
    if kind == "tiktoken":
        return TiktokenWrap(name)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def load_corpus(lang: str, sample_kb: int) -> str:
    path = os.path.join(CORPORA, f"{lang}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在 —— 先跑 download_corpora.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    budget = sample_kb * 1024
    if len(text.encode("utf-8")) > budget:
        text = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-kb", type=int, default=1024)
    ap.add_argument("--price-per-1m", type=float, default=0.15,
                    help="每百万 token 的美元单价(用于成本折算)")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    # 预加载 tokenizer(有的可能需要 HF 登录/授权,失败就跳过并提示)
    toks = {}
    for label, (kind, name) in TOKENIZERS.items():
        try:
            toks[label] = load_tokenizer(kind, name)
            print(f"[ok]  loaded {label}")
        except Exception as e:
            print(f"[skip] {label}: {e}")

    rows = []
    for lang in args.langs:
        text = load_corpus(lang, args.sample_kb)
        n_chars = len(text)
        for label, tok in toks.items():
            n_tokens = len(tok.encode(text, add_special_tokens=False))
            fert = fertility(tok, text)
            tokens_per_1k_char = n_tokens / n_chars * 1000
            cost_per_1m_char = n_tokens / n_chars * 1_000_000 / 1_000_000 * args.price_per_1m
            rows.append({
                "lang": lang,
                "tokenizer": label,
                "tok_per_1k_char": round(tokens_per_1k_char, 2),
                "fert_mean": round(fert["mean"], 3),
                "fert_p90": round(fert["p90"], 2),
                "fert_max": fert["max"],
                "usd_per_1m_char": round(cost_per_1m_char, 4),
            })

    df = pd.DataFrame(rows)

    # 相对英文的倍数(用 tok_per_1k_char,同一 tokenizer 内比较)
    df["vs_en"] = 0.0
    for label in df["tokenizer"].unique():
        mask = df["tokenizer"] == label
        en = df[mask & (df["lang"] == "en")]["tok_per_1k_char"]
        if len(en):
            base = en.values[0]
            df.loc[mask, "vs_en"] = (df.loc[mask, "tok_per_1k_char"] / base).round(2)

    out_csv = os.path.join(RESULTS, "week3_cost_table.csv")
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 120)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"\n已写入 {out_csv}")
    print("重点看 vs_en 列:形态丰富(fi/tr)和非拉丁(zh/ar/ja)通常 >1,")
    print("有的达 3–5 倍。fert_p90/max 揭示尾部成本 —— 均值会把它抹平。")


if __name__ == "__main__":
    main()
