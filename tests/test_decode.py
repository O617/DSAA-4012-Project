import unittest
from types import SimpleNamespace

import torch

from mlsys360.decode import fixed_work_decode


class FakeModel:
    def __init__(self):
        self.input_lengths = []
        self.mask_lengths = []

    def __call__(self, input_ids, attention_mask, use_cache, past_key_values=None):
        self.input_lengths.append(input_ids.shape[1])
        self.mask_lengths.append(attention_mask.shape[1])
        logits = torch.zeros((*input_ids.shape, 4))
        logits[:, -1, 2] = 1
        return SimpleNamespace(logits=logits, past_key_values=("cache",))


class DecodeTests(unittest.TestCase):
    def test_cached_decode_uses_one_token_after_prefill(self):
        model = FakeModel()
        input_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        result = fixed_work_decode(model, input_ids, attention_mask, 3, "cpu", True)

        self.assertEqual(model.input_lengths, [3, 1, 1])
        self.assertEqual(model.mask_lengths, [3, 4, 5])
        self.assertEqual(result.generated_token_ids, [[2, 2, 2], [2, 2, 2]])
        self.assertEqual(len(result.token_intervals), 2)
        self.assertGreater(result.memory["peak_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
