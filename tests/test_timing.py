import math
import unittest

from mlsys360.timing import percentile, summarize


class TimingTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        self.assertAlmostEqual(percentile([0, 10], 0.95), 9.5)

    def test_empty_summary(self):
        result = summarize([])
        self.assertEqual(result["count"], 0)
        self.assertTrue(math.isnan(result["mean"]))


if __name__ == "__main__":
    unittest.main()

