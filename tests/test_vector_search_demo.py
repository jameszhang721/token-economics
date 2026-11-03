"""离线向量检索教学示例的测试。"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.vector_search_demo import (ENCODER, SAMPLE_DOCS, VectorStore,
                                        cosine, vectorize)

ROOT = Path(__file__).resolve().parents[1]
CLI_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


class VectorSearchTests(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore.from_documents(SAMPLE_DOCS)

    def test_cosine_and_mismatched_dimensions(self):
        self.assertAlmostEqual(cosine([1, 0], [2, 0]), 1.0)
        self.assertEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([], []), 0.0)
        with self.assertRaisesRegex(ValueError, "维度不一致"):
            cosine([1], [1, 2])

    def test_ranking_top_k_and_stable_ties(self):
        results = self.store.search("向量数据库怎么检索文档？", top_k=2)
        self.assertEqual(results[0][1]["id"], "vector")
        self.assertEqual(len(results), 2)
        tied = VectorStore.from_documents([
            {"id": "b", "text": "相似度", "metadata": {}},
            {"id": "a", "text": "相似度", "metadata": {}}])
        self.assertEqual([record["id"] for _, record in tied.search("相似度")],
                         ["a", "b"])

    def test_empty_unseen_and_invalid_queries(self):
        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("量子力学"), [])
        self.assertEqual(vectorize("量子力学", self.store.vocabulary),
                         [0] * len(self.store.vocabulary))
        with self.assertRaisesRegex(ValueError, "query 必须是字符串"):
            self.store.search(None)
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.store.search("向量", top_k=top_k)

    def test_save_load_preserves_search_results_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.store.save(path)
            loaded = VectorStore.load(path)
            self.assertEqual(loaded.search("token ID 如何进入模型？"),
                             self.store.search("token ID 如何进入模型？"))
            with self.assertRaises(FileExistsError):
                self.store.save(path)

    def test_save_creates_parent_directory_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results" / "vector_index.json"
            self.store.save(path)
            self.assertTrue(path.is_file())
            self.assertEqual(VectorStore.load(path).search("向量数据库"),
                             self.store.search("向量数据库"))

    def test_rejects_incompatible_or_corrupt_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.store.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            alterations = [
                ("encoder", "different-model"),
                ("version", True),
                ("version", 2),
                ("vocabulary", ["错误", "向量"]),
            ]
            for key, value in alterations:
                with self.subTest(key=key, value=value):
                    edited = dict(payload)
                    edited[key] = value
                    path.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        VectorStore.load(path)
            bad = dict(payload)
            bad["records"] = [{**payload["records"][0], "vector": [1]}]
            path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "维度"):
                VectorStore.load(path)
            bad["records"] = [{**payload["records"][0],
                               "vector": [0] * len(payload["vocabulary"])}]
            path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "向量"):
                VectorStore.load(path)
            bad = dict(payload)
            bad["vocabulary"] = payload["vocabulary"][:-1]
            bad["records"] = [{**record, "vector": record["vector"][:-1]}
                              for record in payload["records"]]
            path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                VectorStore.load(path)

    def test_rejects_duplicate_ids_zero_and_nonfinite_vectors(self):
        vocabulary = ["向量"]
        for records in (
            [{"id": "a", "text": "向量", "metadata": {}, "vector": [0]}],
            [{"id": "a", "text": "向量", "metadata": {}, "vector": [math.nan]}],
            [{"id": "a", "text": "向量", "metadata": {}, "vector": [True]}],
            [{"id": "a", "text": "向量", "metadata": {}, "vector": [1]},
             {"id": "a", "text": "向量", "metadata": {}, "vector": [1]}],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                VectorStore(vocabulary, records)

    def test_cli_and_no_new_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "vector_search_demo.py")],
            cwd=str(ROOT), env=CLI_ENV, capture_output=True, text=True,
            encoding="utf-8", check=True)
        self.assertIn("ID=vector", result.stdout)
        self.assertIn(ENCODER, result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            for args in (["--save", str(path)], ["--load", str(path)]):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "vector_search_demo.py"), *args],
                    cwd=str(ROOT), env=CLI_ENV, capture_output=True, text=True,
            encoding="utf-8", check=True)
                self.assertIn("ID=vector", result.stdout)


if __name__ == "__main__":
    unittest.main()
