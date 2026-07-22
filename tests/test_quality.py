import unittest

from mlsys360.quality import perplexity_windows


class QualityTests(unittest.TestCase):
    def test_overlapping_windows_score_every_causal_target_once(self):
        windows = perplexity_windows(sequence_length=3000, max_length=2048, stride=512)
        self.assertEqual(sum(window[3] for window in windows), 2999)
        self.assertEqual(windows[0], (0, 2048, 2048, 2047))
        self.assertEqual(windows[1], (512, 2560, 512, 512))

    def test_stride_cannot_leave_context_gaps(self):
        with self.assertRaisesRegex(ValueError, "stride must be between"):
            perplexity_windows(sequence_length=100, max_length=32, stride=64)


if __name__ == "__main__":
    unittest.main()
