"""离线检查模型快照选择、输入限制与示例阶段的一致性。"""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import inspect_token_pipeline as pipeline


class SourceTests(unittest.TestCase):
    def test_online_source_is_pinned_to_commit(self):
        api = Mock()
        api.model_info.return_value.sha = "a" * 40
        hub = SimpleNamespace(HfApi=Mock(return_value=api))
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.dict(
                os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}):
            source, kwargs = pipeline.resolve_model_source("org/model", revision="v1")
        self.assertEqual((source, kwargs), ("org/model", {"revision": "a" * 40}))
        hub.HfApi.assert_called_once_with(endpoint="https://hf-mirror.com")
        api.model_info.assert_called_once_with("org/model", revision="v1")

    def test_offline_source_uses_cached_snapshot_path(self):
        hub = SimpleNamespace(snapshot_download=Mock(return_value="/cache/snapshots/abc"))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            source, kwargs = pipeline.resolve_model_source(
                "org/model", revision="abc", local_only=True)
        self.assertEqual((source, kwargs), ("/cache/snapshots/abc", {}))
        hub.snapshot_download.assert_called_once_with(
            "org/model", revision="abc", local_files_only=True)

    def test_missing_offline_cache_reports_error(self):
        hub = SimpleNamespace(snapshot_download=Mock(side_effect=FileNotFoundError()))
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(ValueError, "找不到.*缓存"):
                pipeline.resolve_model_source("org/model", local_only=True)

    def test_local_directory_reused_and_revision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(pipeline.resolve_model_source(directory), (directory, {}))
            with self.assertRaisesRegex(ValueError, "本地模型目录"):
                pipeline.resolve_model_source(directory, revision="v1")


class LimitTests(unittest.TestCase):
    def test_default_rejects_oversized_input(self):
        with self.assertRaisesRegex(ValueError, "输入有 5 个 token"):
            pipeline.check_token_limit([1, 2, 3, 4, 5], 3, 256, False)

    def test_truncation_respects_config_and_requested_budget(self):
        self.assertEqual(pipeline.check_token_limit([1, 2, 3, 4], 3, 256, True),
                         [1, 2, 3])
        self.assertEqual(pipeline.check_token_limit([1, 2, 3, 4], 10, 2, True),
                         [1, 2])

    def test_missing_or_invalid_limit_fails_closed(self):
        for limit in (None, 0, -1, "unknown"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                pipeline.check_token_limit([1], limit, 256, True)
        with self.assertRaisesRegex(ValueError, "必须大于 0"):
            pipeline.check_token_limit([1], 10, 0, False)


class DocumentTests(unittest.TestCase):
    def test_environment_commands_are_portable_and_consistent(self):
        root = Path(__file__).resolve().parents[1]
        environment = (root / "environment.yml").read_text(encoding="utf-8")
        name = re.search(r"^name: (\S+)$", environment, re.MULTILINE).group(1)
        for path in (root / "README.md", root / "docs" / "text_to_transformer.md"):
            text = path.read_text(encoding="utf-8")
            commands = re.findall(r"(?:^|&& )conda activate ([a-zA-Z0-9_-]+)",
                                  text, re.MULTILINE)
            self.assertEqual(commands, [name] * len(commands))
            self.assertGreaterEqual(len(commands), 1)
            self.assertNotIn("D:\\zhangjiabao1\\", text)

    def test_python_blocks_execute_in_order_with_local_model(self):
        import torch

        document = (Path(__file__).resolve().parents[1] / "docs" /
                    "text_to_transformer.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)\n```", document, re.DOTALL)
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.convert_ids_to_tokens.return_value = ["a", "b"]
        tokenizer.decode.return_value = "你好，token！"

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(
                    hidden_states=[self.embedding(input_ids)],
                    logits=torch.zeros(1, input_ids.shape[1], 8))

        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            AutoConfig=SimpleNamespace(from_pretrained=Mock(
                return_value=SimpleNamespace(max_position_embeddings=32))),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(
                return_value=TinyModel())))
        namespace = {}
        with patch.dict(sys.modules, {"transformers": fake_transformers}), patch.object(
                pipeline, "resolve_model_source", return_value=("local-snapshot", {})), patch.object(
                torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            for block in blocks:
                exec(compile(block, "docs/text_to_transformer.md", "exec"), namespace)
        self.assertEqual(namespace["input_ids"].tolist(), [[1, 2]])
        self.assertEqual(tuple(namespace["contextual"].shape), (1, 2, 4))
        for loader in (fake_transformers.AutoTokenizer, fake_transformers.AutoConfig,
                       fake_transformers.AutoModelForCausalLM):
            self.assertEqual(loader.from_pretrained.call_args.args, ("local-snapshot",))


class ModelStageTests(unittest.TestCase):
    def run_stage(self, truncate, source=None):
        import torch

        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [str(i) for i in ids]
        tokenizer.decode.side_effect = lambda ids: ",".join(map(str, ids))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.seen_ids = None

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, **kwargs):
                self.seen_ids = input_ids.tolist()
                return SimpleNamespace(
                    hidden_states=[self.embedding(input_ids)],
                    logits=torch.zeros(1, input_ids.shape[1], 8))

        model = TinyModel()
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            AutoConfig=SimpleNamespace(from_pretrained=Mock(
                return_value=SimpleNamespace(max_position_embeddings=3))),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        hub = SimpleNamespace(
            HfApi=Mock(return_value=SimpleNamespace(
                model_info=Mock(return_value=SimpleNamespace(sha="b" * 40)))),
            snapshot_download=Mock(return_value="/cache/snapshots/abc"))
        output, errors = io.StringIO(), io.StringIO()
        argv = ["inspect_token_pipeline.py", "--stage", "model", "--text", "sample"]
        if source is not None:
            argv.extend(["--model", source, "--local-only"])
        if truncate:
            argv.append("--truncate")
        with patch.dict(sys.modules, {"transformers": fake_transformers,
                                      "huggingface_hub": hub}), patch.object(
                pipeline, "setup_hf_endpoint"), patch.object(pipeline, "hf_report"), patch.object(
                torch.cuda, "is_available", return_value=False), patch.object(
                sys, "argv", argv), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            if truncate:
                pipeline.main()
            else:
                with self.assertRaises(SystemExit):
                    pipeline.main()
        return fake_transformers, model, output.getvalue(), errors.getvalue()

    def test_offline_cached_snapshot_loaders_use_one_path(self):
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(
                snapshot_download=Mock(return_value="/cache/snapshots/abc"))}):
            loaders, model, output, error = self.run_stage(True, "org/model")
        self.assertEqual(error, "")
        self.assertEqual(model.seen_ids, [[1, 2, 3]])
        for loader in (loaders.AutoTokenizer, loaders.AutoConfig,
                       loaders.AutoModelForCausalLM):
            args, kwargs = loader.from_pretrained.call_args
            self.assertEqual(args, ("/cache/snapshots/abc",))
            self.assertTrue(kwargs["local_files_only"])
            self.assertNotIn("revision", kwargs)

    def test_local_model_loaders_use_one_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            loaders, model, output, error = self.run_stage(True, directory)
        self.assertEqual(error, "")
        self.assertEqual(model.seen_ids, [[1, 2, 3]])
        for loader in (loaders.AutoTokenizer, loaders.AutoConfig,
                       loaders.AutoModelForCausalLM):
            args, kwargs = loader.from_pretrained.call_args
            self.assertEqual(args, (directory,))
            self.assertTrue(kwargs["local_files_only"])
            self.assertNotIn("revision", kwargs)

    def test_oversize_rejected_before_model_load(self):
        loaders, model, output, error = self.run_stage(False)
        self.assertIn("输入编码得到 5 个 token", output)
        self.assertIn("输入有 5 个 token", error)
        loaders.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIsNone(model.seen_ids)

    def test_truncated_ids_match_display_and_forward_at_same_revision(self):
        loaders, model, output, error = self.run_stage(True)
        self.assertEqual(error, "")
        self.assertIn("③ token IDs: [1, 2, 3]", output)
        self.assertIn("整体解码: '1,2,3'", output)
        self.assertIn("仅对应展示的 token 前缀", output)
        self.assertEqual(model.seen_ids, [[1, 2, 3]])
        for loader in (loaders.AutoTokenizer, loaders.AutoConfig,
                       loaders.AutoModelForCausalLM):
            args, kwargs = loader.from_pretrained.call_args
            self.assertEqual(args, (pipeline.DEFAULT_MODEL,))
            self.assertEqual(kwargs["revision"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
