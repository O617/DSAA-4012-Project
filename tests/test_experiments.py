import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mlsys360.experiments import run_experiment
from mlsys360.results import JsonlStore


class FakeResult:
    generated_token_ids = [[2, 2]]

    def as_dict(self):
        return {
            "ttft_seconds": 0.1,
            "tpot_seconds": 0.2,
            "decode_seconds": 0.2,
            "end_to_end_seconds": 0.3,
            "tps": 5.0,
            "end_to_end_tps": 6.0,
            "token_timestamps": [0.1, 0.3],
            "token_intervals": [0.2],
            "generated_token_ids": self.generated_token_ids,
            "peak_rss_bytes": 100,
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
        }


class ExperimentTests(unittest.TestCase):
    def test_fully_resumed_variant_does_not_reload_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "experiment": "resume_test",
                "model": {"model_id": "fake", "revision": "abc"},
                "runtime": {"device": "cpu", "dtype": "float32", "use_cache": True},
                "workload": {
                    "context_lengths": [8],
                    "batch_sizes": [1],
                    "output_tokens": 2,
                    "warmups": 0,
                    "repetitions": 1,
                    "seed": 1,
                },
                "output": {"path": str(Path(directory) / "raw.jsonl"), "resume": True},
            }
            bundle = SimpleNamespace(
                model=object(),
                tokenizer=object(),
                device="cpu",
                metadata={},
                quantization={},
            )

            def make_prompt(_tokenizer, context_length, batch_size, _device):
                ids = torch.zeros((batch_size, context_length), dtype=torch.long)
                return SimpleNamespace(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    prompt_hash="x",
                )

            with (
                patch("mlsys360.experiments.load_model", return_value=bundle) as loader,
                patch("mlsys360.experiments.exact_length_prompt", side_effect=make_prompt),
                patch("mlsys360.experiments.fixed_work_decode", return_value=FakeResult()),
            ):
                first = run_experiment(config)
                second = run_experiment(config)

            self.assertEqual(loader.call_count, 1)
            self.assertEqual(first["completed"], 1)
            self.assertEqual(second["skipped"], 1)

    def test_oom_only_skips_batches_at_or_above_failed_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.jsonl"
            config = {
                "experiment": "test",
                "model": {"model_id": "fake", "revision": "abc"},
                "runtime": {
                    "device": "cpu",
                    "dtype": "float32",
                    "attention": "sdpa",
                    "use_cache": True,
                    "compile": False,
                },
                "workload": {
                    "points": [
                        {"context_length": 8, "batch_size": 4},
                        {"context_length": 8, "batch_size": 1},
                        {"context_length": 8, "batch_size": 8},
                    ],
                    "output_tokens": 2,
                    "warmups": 0,
                    "repetitions": 1,
                    "seed": 1,
                },
                "output": {"path": str(output), "resume": True},
            }
            bundle = SimpleNamespace(
                model=object(),
                tokenizer=object(),
                device="cpu",
                metadata={},
                quantization={},
            )
            calls = []

            def make_prompt(_tokenizer, context_length, batch_size, _device):
                ids = torch.zeros((batch_size, context_length), dtype=torch.long)
                return SimpleNamespace(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    prompt_hash="x",
                )

            def decode(_model, input_ids, *_args):
                batch_size = int(input_ids.shape[0])
                calls.append(batch_size)
                if batch_size == 4:
                    raise torch.cuda.OutOfMemoryError("synthetic OOM")
                return FakeResult()

            with (
                patch("mlsys360.experiments.load_model", return_value=bundle),
                patch("mlsys360.experiments.exact_length_prompt", side_effect=make_prompt),
                patch("mlsys360.experiments.fixed_work_decode", side_effect=decode),
            ):
                summary = run_experiment(config)
                run_experiment(config)

            statuses = [row["status"] for row in JsonlStore(output).read()]
            self.assertEqual(calls, [4, 1, 4])
            self.assertIn("ok", statuses)
            self.assertIn("oom", statuses)
            self.assertIn("skipped_after_oom", statuses)
            self.assertEqual(summary["completed"], 1)
            manifest = json.loads(
                output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["invocations"]), 2)


if __name__ == "__main__":
    unittest.main()
