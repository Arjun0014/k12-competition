import unittest

import numpy as np
import pandas as pd

from trace_ace.baselines import global_prior_oof, objective_prior_oof


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "fold": [0, 0, 1, 1, 2, 2],
                "session_id": ["a", "a", "b", "b", "c", "c"],
                "learning_objective_id": ["x", "y", "x", "y", "x", "z"],
                "target": [1, 0, 1, 1, 0, 0],
            }
        )

    def test_global_prior_uses_only_other_folds(self):
        prediction = global_prior_oof(self.frame)
        expected_fold_zero = self.frame.loc[self.frame["fold"].ne(0), "target"].mean()
        np.testing.assert_allclose(prediction[:2], expected_fold_zero)

    def test_objective_prior_outputs_probabilities(self):
        prediction = objective_prior_oof(self.frame, alpha=2.0)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(((prediction >= 0.0) & (prediction <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
