import unittest

import numpy as np
import pandas as pd

from trace_ace.semantic_knn_validation import KnnConfig, semantic_knn_oof


class SemanticKnnValidationTests(unittest.TestCase):
    def test_predictions_are_finite_and_validation_labels_are_not_used(self) -> None:
        frame = pd.DataFrame(
            {
                "session_id": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "learning_objective_id": ["a", "a", "b", "b", "c", "c"],
                "target": [0, 0, 1, 1, 0, 1],
            }
        )
        folds = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        context = np.array(
            [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.7, 0.7], [0.6, 0.8]],
            dtype=np.float32,
        )
        objective = np.array(
            [[1, 0], [1, 0], [0, 1], [0, 1], [0.7, 0.7], [0.7, 0.7]],
            dtype=np.float32,
        )
        config = KnnConfig("test", 2, 3, 0.5, 8.0, 1.0)
        first = semantic_knn_oof(frame, folds, context, objective, config, 3)
        changed = frame.copy()
        changed.loc[folds == 0, "target"] = 1
        second = semantic_knn_oof(changed, folds, context, objective, config, 3)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first > 0) & (first < 1)).all())
        np.testing.assert_allclose(first[folds == 0], second[folds == 0])


if __name__ == "__main__":
    unittest.main()
