import unittest

import torch

from mlsys360.quantization import _torchao_quantized_module_info, quantize_model


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

    def test_torchao_weight_tensor_detection_does_not_rely_on_module_type(self):
        class FakeTorchAOWeight:
            pass

        FakeTorchAOWeight.__module__ = "torchao.dtypes.fake"

        class FakeModel:
            def named_modules(self):
                return [("projection", type("Linear", (), {"weight": FakeTorchAOWeight()})())]

        self.assertEqual(
            _torchao_quantized_module_info(FakeModel()),
            [
                {
                    "name": "projection",
                    "weight_type": "torchao.dtypes.fake.FakeTorchAOWeight",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
