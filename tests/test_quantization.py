import unittest

import torch

from mlsys360.quantization import quantize_model


class ToyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


class QuantizationTests(unittest.TestCase):
    def test_dynamic_cpu_quantization_preserves_lm_head(self):
        model, metadata = quantize_model(ToyLanguageModel(), "dynamic_w8a8", "cpu")
        self.assertIn("quantized", type(model.projection).__module__)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertEqual(metadata["excluded_modules"], ["lm_head"])


if __name__ == "__main__":
    unittest.main()
