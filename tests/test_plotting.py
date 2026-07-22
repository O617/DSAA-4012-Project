import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlsys360-matplotlib-tests")

PLOTTING_AVAILABLE = importlib.util.find_spec("seaborn") is not None
main = (
    run_path(str(Path(__file__).parents[1] / "scripts" / "make_plots.py"))["main"]
    if PLOTTING_AVAILABLE
    else None
)


class PlottingTests(unittest.TestCase):
    @unittest.skipUnless(PLOTTING_AVAILABLE, "seaborn is required for plotting")
    def test_variants_produce_separate_frontiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            output = root / "figures"
            rows = []
            for attention, multiplier in (("eager", 1.0), ("sdpa", 0.8)):
                for batch_size in (1, 2):
                    rows.append(
                        {
                            "status": "ok",
                            "hardware_id": "machine",
                            "model_id": "fake/model",
                            "model_revision": "abc",
                            "device": "cpu",
                            "dtype": "float32",
                            "attention": attention,
                            "use_cache": True,
                            "compile": False,
                            "quantization": "none",
                            "context_length": 8,
                            "batch_size": batch_size,
                            "ttft_seconds": multiplier,
                            "tpot_seconds": multiplier / batch_size,
                            "tps": batch_size / multiplier,
                        }
                    )
            pd.DataFrame(rows).to_json(raw, orient="records", lines=True)

            with patch(
                "sys.argv",
                ["make_plots.py", str(raw), "--output-dir", str(output)],
            ):
                self.assertIsNotNone(main)
                self.assertEqual(main(), 0)

            self.assertEqual(len(list(output.glob("tps_heatmap_*.png"))), 2)
            self.assertEqual(len(list(output.glob("tpot_throughput_frontier_*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
