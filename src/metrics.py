"""三把尺子 —— 第 0 周造好,后面每周复用。

跑自检:  python -m src.metrics --selftest
"""
import math
from collections import Counter


def entropy(corpus: str, ngram_order: int = 1) -> float:
    """语料的经验 n-gram 条件熵,单位 bits/char。这是 BPB 的理论下界参照。

    ngram_order=1 : 无条件字符熵 H(X)
    ngram_order=n : 条件熵 H(X_n | X_{n-1..1}) = H(n-gram) - H((n-1)-gram context)
    """
    if len(corpus) < ngram_order:
        raise ValueError("corpus shorter than ngram_order")

    if ngram_order == 1:
        counts = Counter(corpus)
        total = sum(counts.values())
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def H(counter: Counter) -> float:
        t = sum(counter.values())
        return -sum((c / t) * math.log2(c / t) for c in counter.values())

    ngrams = Counter(corpus[i:i + ngram_order]
                     for i in range(len(corpus) - ngram_order + 1))
    context = Counter(corpus[i:i + ngram_order - 1]
                      for i in range(len(corpus) - ngram_order + 2))
    return H(ngrams) - H(context)


def bpb(model, tokenizer, text: str, device: str | None = None,
        max_length: int = 2048, stride: int | None = None) -> float:
    """模型在 text 上的 bits per byte。

    BPB = 总交叉熵(nats) / (ln2 × UTF-8 字节数)

    关键:分母是**字节数**而不是 token 数 —— 这才让不同 tokenizer 可比。
    长文本用滑动窗口分块累计 NLL,避免超出模型上下文。
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if stride is None:
        stride = max_length  # 不重叠;要更精确可设 max_length // 2

    ids = tokenizer.encode(text)
    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        raise ValueError("empty text")

    input_ids = torch.tensor([ids], device=device)
    total_nll_nats = 0.0
    prev_end = 0
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        trg_len = end - prev_end  # 本窗口真正计入 loss 的 token 数
        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        target[:, :-trg_len] = -100  # 只对新 token 计 loss(重叠部分忽略)
        with torch.no_grad():
            out = model(chunk, labels=target)
            # HF loss 是 mean over 有效 token;有效 token 数 = trg_len - 1(shift)
            n_valid = max(trg_len - 1, 1)
            total_nll_nats += out.loss.item() * n_valid
        prev_end = end
        if end == input_ids.size(1):
            break

    return total_nll_nats / (math.log(2) * n_bytes)


def fertility(tokenizer, text: str) -> dict:
    """一个词被切成几个 token。均值没用,要看尾部(p90/p99/max)—— 长词才是成本黑洞。"""
    import numpy as np

    words = text.split()
    if not words:
        raise ValueError("no words in text")
    counts = [len(tokenizer.encode(w, add_special_tokens=False)) for w in words]
    a = np.array(counts, dtype=float)
    return {
        "n_words": len(words),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def _selftest():
    """不依赖 GPU / 模型,只验证熵与 fertility 的数学是否自洽。"""
    print("== entropy sanity ==")
    # 均匀 4 符号 → 正好 2 bits/char
    uniform = "ABCD" * 1000
    h = entropy(uniform, 1)
    print(f"  uniform 4-symbol H = {h:.4f} bits/char (expect 2.0000)")
    assert abs(h - 2.0) < 1e-6, "均匀分布熵应为 log2(4)=2"

    # 单一符号 → 0 bits
    h0 = entropy("A" * 1000, 1)
    print(f"  single-symbol   H = {h0:.4f} bits/char (expect 0.0000)")
    assert abs(h0) < 1e-9

    # 条件熵 ≤ 无条件熵(信息不增)
    text = "the quick brown fox jumps over the lazy dog " * 200
    h1, h2 = entropy(text, 1), entropy(text, 2)
    print(f"  H1 = {h1:.4f}, H2|1 = {h2:.4f}  (expect H2|1 <= H1)")
    assert h2 <= h1 + 1e-9, "条件熵不应超过无条件熵"

    print("== fertility sanity ==")

    class _CharTok:
        """假 tokenizer:每字符一个 token,fertility 应等于词长。"""
        def encode(self, s, add_special_tokens=False):
            return list(s)

    f = fertility(_CharTok(), "a bb ccc dddd")
    print(f"  char-tokenizer fertility mean={f['mean']:.2f} max={f['max']} (expect max=4)")
    assert f["max"] == 4 and abs(f["mean"] - 2.5) < 1e-9

    print("\nAll self-tests passed. 尺子可用。")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("用法: python -m src.metrics --selftest")
