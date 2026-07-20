import unittest

import numpy as np

from trace_ace.metrics import binary_metrics, expected_calibration_error


class MetricsTests(unittest.TestCase):
    def test_better_predictions_have_lower_log_loss(self):
        target = np.array([0, 0, 1, 1])
        good = binary_metrics(target, np.array([0.1, 0.2, 0.8, 0.9]))
        weak = binary_metrics(target, np.full(4, 0.5))
        self.assertLess(good["log_loss"], weak["log_loss"])
        self.assertGreater(good["roc_auc"], weak["roc_auc"])

    def test_ece_is_zero_for_matching_single_bin_rate(self):
        target = np.array([0, 1, 0, 1])
        probability = np.full(4, 0.5)
        self.assertAlmostEqual(expected_calibration_error(target, probability), 0.0)


if __name__ == "__main__":
    unittest.main()
