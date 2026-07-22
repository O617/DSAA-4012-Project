import unittest

from mlsys360.results import aggregate_rows, batching_knee, result_identity


def row(batch, repetition, tps, ttft, tpot):
    return {
        "status": "ok",
        "experiment": "test",
        "hardware_id": "machine",
        "device": "cpu",
        "dtype": "float32",
        "attention": "sdpa",
        "use_cache": True,
        "quantization": "none",
        "context_length": 128,
        "batch_size": batch,
        "output_tokens": 4,
        "repetition": repetition,
        "tps": tps,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "end_to_end_seconds": ttft + 3 * tpot,
        "peak_rss_bytes": 100,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
    }


class ResultTests(unittest.TestCase):
    def test_aggregation_and_knee(self):
        rows = [
            row(1, 0, 10, 1.0, 0.10),
            row(1, 1, 10, 1.1, 0.11),
            row(2, 0, 10.5, 1.3, 0.12),
            row(2, 1, 10.5, 1.4, 0.13),
        ]
        aggregates = aggregate_rows(rows)
        self.assertEqual(len(aggregates), 2)
        knee = batching_knee(aggregates)
        self.assertEqual(knee["batch_size"], 2)
        self.assertAlmostEqual(knee["marginal_tps_gain"], 0.05)

    def test_resume_identity_includes_material_runtime_settings(self):
        first = row(1, 0, 10, 1.0, 0.1)
        first.update(
            model_id="model",
            model_revision="a",
            compile=False,
            seed=1,
            num_threads=4,
            num_interop_threads=1,
            cpu_affinity="0-3",
        )
        second = dict(first, model_revision="b")
        third = dict(first, compile=True)
        fourth = dict(first, seed=2)
        fifth = dict(first, num_threads=8)
        self.assertNotEqual(result_identity(first), result_identity(second))
        self.assertNotEqual(result_identity(first), result_identity(third))
        self.assertNotEqual(result_identity(first), result_identity(fourth))
        self.assertNotEqual(result_identity(first), result_identity(fifth))


if __name__ == "__main__":
    unittest.main()
