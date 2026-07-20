import unittest

import numpy as np
import pandas as pd

from trace_ace.robust_validation import build_semantic_family_folds, regime_scorecard


class RobustValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        rows = []
        embeddings = []
        for objective in range(18):
            center = rng.normal(size=8)
            center /= np.linalg.norm(center)
            for repetition in range(6):
                rows.append(
                    {
                        "response_id": f"r{objective:02d}_{repetition}",
                        "session_id": f"s{objective:02d}_{repetition}",
                        "learning_objective_id": f"o{objective:02d}",
                        "learning_objective": f"objective {objective}",
                        "target": (objective + repetition) % 2,
                    }
                )
                embeddings.append(center)
        self.frame = pd.DataFrame(rows)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def test_semantic_folds_are_complete_reproducible_and_family_disjoint(self):
        first = build_semantic_family_folds(
            self.frame,
            self.embeddings,
            n_clusters=9,
            n_splits=3,
            cluster_seed=7,
            fold_seed=11,
        )
        second = build_semantic_family_folds(
            self.frame,
            self.embeddings,
            n_clusters=9,
            n_splits=3,
            cluster_seed=7,
            fold_seed=11,
        )
        folds, assignments, summaries = first
        np.testing.assert_array_equal(folds, second[0])
        self.assertEqual(set(folds), {0, 1, 2})
        self.assertEqual(len(assignments), 18)
        self.assertEqual(len(summaries), 3)
        family = assignments.set_index("learning_objective_id")["semantic_family"]
        row_family = self.frame["learning_objective_id"].map(family).to_numpy()
        for fold in range(3):
            validation_families = set(row_family[folds == fold])
            training_families = set(row_family[folds != fold])
            self.assertFalse(validation_families.intersection(training_families))

    def test_regime_scorecard_includes_overall_fold_and_regime_rows(self):
        target = np.array([0, 1, 0, 1, 0, 1])
        prediction = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        folds = np.array([0, 0, 1, 1, 2, 2])
        regimes = pd.DataFrame(
            {
                "response_id": [f"r{i}" for i in range(6)],
                "length": ["short", "short", "long", "long", "long", "short"],
            }
        )
        result = regime_scorecard("test", target, prediction, folds, regimes)
        self.assertTrue(
            ((result["regime"] == "overall") & (result["value"] == "all")).any()
        )
        self.assertEqual(
            set(result.loc[result["regime"] == "fold", "value"]), {"0", "1", "2"}
        )
        self.assertEqual(
            set(result.loc[result["regime"] == "length", "value"]), {"long", "short"}
        )


if __name__ == "__main__":
    unittest.main()
