import unittest

import numpy as np

from trace_ace.qwen_pilot import PILOT_ROWS, select_pilot_indices


class QwenPilotTests(unittest.TestCase):
    def test_selection_is_reproducible_and_stratified(self):
        rows = 10000
        folds = np.arange(rows) % 5
        target = (np.arange(rows) % 3 != 0).astype(np.int8)
        first = select_pilot_indices(folds, target)
        second = select_pilot_indices(folds, target)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), PILOT_ROWS)
        self.assertEqual(len(np.unique(first)), PILOT_ROWS)
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertEqual(set(folds[first]), set(range(5)))
        self.assertEqual(set(target[first]), {0, 1})


if __name__ == "__main__":
    unittest.main()
