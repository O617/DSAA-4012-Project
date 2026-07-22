import hashlib
import tempfile
import unittest
from pathlib import Path

from mlsys360.model import local_model_artifact_hashes


class ModelProvenanceTests(unittest.TestCase):
    def test_local_model_hashes_include_inference_artifacts_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(b"config")
            (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            (root / "trainer_state.json").write_bytes(b"training metadata")

            artifacts = local_model_artifact_hashes(directory)

        self.assertEqual(
            [artifact["path"] for artifact in artifacts],
            ["config.json", "model-00001-of-00001.safetensors"],
        )
        self.assertEqual(artifacts[0]["sha256"], hashlib.sha256(b"config").hexdigest())
        self.assertEqual(artifacts[1]["size_bytes"], len(b"weights"))


if __name__ == "__main__":
    unittest.main()
