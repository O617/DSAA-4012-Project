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
                "workload: {output_tokens: 4}\noutput: {path: x.jsonl}\n",
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


if __name__ == "__main__":
    unittest.main()
