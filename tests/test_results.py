import unittest

from mlsys360.results import aggregate_rows, batching_knee


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


if __name__ == "__main__":
    unittest.main()

