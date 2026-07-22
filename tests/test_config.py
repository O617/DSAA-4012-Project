import tempfile
import unittest
from pathlib import Path

try:
    import yaml  # noqa: F401
except ImportError:
    yaml = None

from mlsys360.config import apply_overrides, deep_merge, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_deep_merge_preserves_nested_values(self):
        result = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"x": 3}})
        self.assertEqual(result, {"a": {"x": 3, "y": 2}})

    @unittest.skipIf(yaml is None, "PyYAML is installed with the project dependencies")
    def test_relative_inheritance_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "model: {model_id: test}\nruntime: {device: cpu}\n"
                "workload: {context_lengths: [8], batch_sizes: [1], output_tokens: 4}\n"
                "output: {path: x.jsonl}\n",
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                "extends: base.yaml\nruntime: {dtype: float32}\n", encoding="utf-8"
            )
            config = load_config(root / "child.yaml")
            config = apply_overrides(config, ["workload.output_tokens=8"])
            validate_config(config)
            self.assertEqual(config["runtime"], {"device": "cpu", "dtype": "float32"})
            self.assertEqual(config["workload"]["output_tokens"], 8)

    @unittest.skipIf(yaml is None, "PyYAML is installed with the project dependencies")
    def test_circular_inheritance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Circular configuration inheritance"):
                load_config(root / "a.yaml")

    def test_invalid_workload_is_rejected(self):
        config = {
            "model": {"model_id": "test"},
            "runtime": {"device": "cpu"},
            "workload": {
                "context_lengths": [0],
                "batch_sizes": [1],
                "output_tokens": 4,
            },
            "output": {"path": "x.jsonl"},
        }
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
