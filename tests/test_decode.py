import unittest
from types import SimpleNamespace

import torch

from mlsys360.decode import fixed_work_decode


class FakeModel:
    def __init__(self):
        self.input_lengths = []
        self.mask_lengths = []
        self.logits_to_keep = []

    def __call__(
        self, input_ids, attention_mask, use_cache, past_key_values=None, logits_to_keep=None
    ):
        self.input_lengths.append(input_ids.shape[1])
        self.mask_lengths.append(attention_mask.shape[1])
        self.logits_to_keep.append(logits_to_keep)
        logits = torch.zeros((input_ids.shape[0], logits_to_keep, 4))
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
        self.assertEqual(model.logits_to_keep, [1, 1, 1])
        self.assertEqual(result.generated_token_ids, [[2, 2, 2], [2, 2, 2]])
        self.assertEqual(result.prefill_logits_shape, [2, 1, 4])
        self.assertEqual(result.max_logits_positions, 1)
        self.assertEqual(len(result.token_intervals), 2)
        self.assertGreater(result.memory["peak_rss_bytes"], 0)

    def test_cache_off_keeps_full_history_but_only_last_logits(self):
        model = FakeModel()
        input_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        result = fixed_work_decode(model, input_ids, attention_mask, 3, "cpu", False)

        self.assertEqual(model.input_lengths, [3, 4, 5])
        self.assertEqual(model.mask_lengths, [3, 4, 5])
        self.assertEqual(model.logits_to_keep, [1, 1, 1])
        self.assertEqual(result.prefill_logits_shape, [2, 1, 4])

    def test_model_without_last_logit_support_fails_closed(self):
        class UnsupportedModel:
            def __call__(self, input_ids, attention_mask, use_cache):
                raise AssertionError("must not run without an explicit last-logit adapter")

        input_ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(RuntimeError, "model-specific last-logit adapter"):
            fixed_work_decode(
                UnsupportedModel(), input_ids, torch.ones_like(input_ids), 2, "cpu", True
            )


if __name__ == "__main__":
    unittest.main()
