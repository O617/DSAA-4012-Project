import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlsys360.quality import evaluate_perplexity, json_safe, perplexity_windows


class QualityTests(unittest.TestCase):
    def test_overlapping_windows_score_every_causal_target_once(self):
        windows = perplexity_windows(sequence_length=3000, max_length=2048, stride=512)
        self.assertEqual(sum(window[3] for window in windows), 2999)
        self.assertEqual(windows[0], (0, 2048, 2048, 2047))
        self.assertEqual(windows[1], (512, 2560, 512, 512))

    def test_stride_cannot_leave_context_gaps(self):
        with self.assertRaisesRegex(ValueError, "stride must be between"):
            perplexity_windows(sequence_length=100, max_length=32, stride=64)

    def test_json_safe_uses_stable_callable_name(self):
        value = json_safe({"function": self.test_json_safe_uses_stable_callable_name})
        self.assertEqual(
            value["function"],
            "test_quality.QualityTests.test_json_safe_uses_stable_callable_name",
        )

    @patch("mlsys360.quality.load_model")
    def test_local_text_records_content_provenance(self, load_model):
        import tempfile

        import torch

        model = MagicMock()
        model.config.max_position_embeddings = 16
        model.return_value.loss = torch.tensor(1.0)
        tokenizer = MagicMock()
        tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3, 4]])
        load_model.return_value = MagicMock(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            metadata={},
            quantization={},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("first\n\nsecond\n", encoding="utf-8")
            result = evaluate_perplexity({}, "none", max_samples=2, stride=4, text_file=path)
        self.assertEqual(result["evaluated_tokens"], 3)
        self.assertEqual(result["source"]["type"], "local_text_file")
        self.assertEqual(result["source"]["line_count"], 3)
        self.assertEqual(
            result["source"]["sha256"],
            "457098d8c79229c8199e82791bf8d67aa0a37adbd1ab5e3e145a13f9832d5a23",
        )


if __name__ == "__main__":
    unittest.main()
