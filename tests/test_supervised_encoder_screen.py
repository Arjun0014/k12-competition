import unittest

import numpy as np

from trace_ace.supervised_encoder_screen import (
    mastery_hypothesis,
    stratified_screen_indices,
)


class SupervisedEncoderScreenTests(unittest.TestCase):
    def test_mastery_hypothesis_retains_objective(self):
        rendered = mastery_hypothesis("  explain equivalent fractions  ")
        self.assertEqual(
            rendered,
            "The student demonstrates mastery of this learning objective: "
            "explain equivalent fractions.",
        )

    def test_stratified_sample_is_deterministic_and_aligned(self):
        candidate = np.arange(100, dtype=np.int64)
        target = np.asarray([0] * 40 + [1] * 60, dtype=np.int8)
        first = stratified_screen_indices(candidate, target, train_rows=20, seed=7)
        second = stratified_screen_indices(candidate, target, train_rows=20, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(int(target[first].sum()), 12)
        self.assertTrue(np.all(np.diff(first) > 0))


if __name__ == "__main__":
    unittest.main()
