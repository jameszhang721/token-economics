"""用字符二元组和余弦相似度演示小规模、精确的向量检索（不需要网络）。"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

ENCODER = "char-bigram-v1"
SAMPLE_DOCS = [
    {"id": "token", "text": "Tokenizer 把文本切分为 token，再映射到整数 ID。", "metadata": {"topic": "token"}},
    {"id": "embedding", "text": "模型内部的 embedding 层按 token ID 查找权重矩阵中的一行。", "metadata": {"topic": "llm"}},
    {"id": "vector", "text": "向量数据库保存文档向量和原文，可按相似度检索相关段落。", "metadata": {"topic": "retrieval"}},
    {"id": "rag", "text": "RAG 从向量索引检索原文，把原文放入提示词，再由语言模型编码。", "metadata": {"topic": "rag"}},
]


def bigrams(text):
    compact = "".join(text.lower().split())
    return [compact[i:i + 2] for i in range(len(compact) - 1)]


def vectorize(text, vocabulary):
    counts = Counter(bigrams(text))
    return [counts.get(term, 0) for term in vocabulary]


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("向量维度不一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class VectorStore:
    def __init__(self, vocabulary, records):
        if (not isinstance(vocabulary, list) or not vocabulary or
                any(not isinstance(term, str) or len(term) != 2 for term in vocabulary) or
                vocabulary != sorted(set(vocabulary))):
            raise ValueError("词表必须是非空、有序且不重复的字符二元组列表")
        if not isinstance(records, list) or not records:
            raise ValueError("索引不能没有文档")
        seen = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {"id", "text", "metadata", "vector"}:
                raise ValueError("索引记录字段无效")
            doc_id, text, metadata, vector = (record[key] for key in
                                              ("id", "text", "metadata", "vector"))
            if (not isinstance(doc_id, str) or not doc_id or doc_id in seen or
                    not isinstance(text, str) or not text or
                    not isinstance(metadata, dict) or
                    any(not isinstance(k, str) or not isinstance(v, str)
                        for k, v in metadata.items())):
                raise ValueError("文档 ID、原文或元数据无效")
            if (not isinstance(vector, list) or len(vector) != len(vocabulary) or
                    any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value < 0 for value in vector) or
                    not any(vector) or vector != vectorize(text, vocabulary)):
                raise ValueError("文档向量的维度、数值或原文编码不一致")
            seen.add(doc_id)
        if vocabulary != sorted({term for record in records for term in bigrams(record["text"])}):
            raise ValueError("词表与文档原文不一致")
        self.vocabulary = vocabulary
        self.records = records

    @classmethod
    def from_documents(cls, documents):
        documents = list(documents)
        vocabulary = sorted({term for doc in documents for term in bigrams(doc["text"])})
        records = [{**doc, "vector": vectorize(doc["text"], vocabulary)} for doc in documents]
        return cls(vocabulary, records)

    def search(self, query, top_k=3):
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须是正整数")
        query_vector = vectorize(query, self.vocabulary)
        if not any(query_vector):
            return []  # 查询在当前词表中没有匹配项，不凭空给出“相关”结果。
        scored = [(cosine(query_vector, record["vector"]), record)
                  for record in self.records]
        scored = [(score, record) for score, record in scored if score > 0]
        return sorted(scored, key=lambda item: (-item[0], item[1]["id"]))[:top_k]

    def save(self, path):
        payload = {"version": 1, "encoder": ENCODER,
                   "vocabulary": self.vocabulary, "records": self.records}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 不覆盖已有数据；JSON 仅供教学，非并发/生产存储格式。
        with path.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or set(payload) !=
                {"version", "encoder", "vocabulary", "records"} or
                type(payload["version"]) is not int or payload["version"] != 1 or
                payload["encoder"] != ENCODER):
            raise ValueError("索引版本或编码方法不匹配；不能混用不同的向量空间")
        return cls(payload["vocabulary"], payload["records"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="向量数据库怎么检索文档？", help="检索问题")
    parser.add_argument("--top-k", type=int, default=3, help="最多返回的文档数")
    parser.add_argument("--save", type=Path, help="将示例索引保存为 JSON；不会覆盖已有文件")
    parser.add_argument("--load", type=Path, help="从本地 JSON 读取索引，不重建示例文档")
    args = parser.parse_args()
    if args.load and args.save:
        parser.error("--load 与 --save 不能同时使用")
    try:
        store = VectorStore.load(args.load) if args.load else VectorStore.from_documents(SAMPLE_DOCS)
        results = store.search(args.query, args.top_k)
        if args.save:
            store.save(args.save)
            print("已保存索引: {}".format(args.save))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print("编码方法: {}（字符二元组词频，不是语义 embedding）".format(ENCODER))
    print("向量维度: {}；文档数: {}；查询: {!r}".format(
        len(store.vocabulary), len(store.records), args.query))
    if not results:
        print("没有共享的字符二元组；未找到可用匹配。")
    for rank, (score, record) in enumerate(results, 1):
        print("{}. 相似度={:.4f} ID={} 元数据={} 原文={}".format(
            rank, score, record["id"], record["metadata"], record["text"]))


if __name__ == "__main__":
    main()
