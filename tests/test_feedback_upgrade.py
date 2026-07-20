from __future__ import annotations

import numpy as np
import unittest
from scipy import sparse

from trace_ace.feedback_upgrade import (
    FEEDBACK_DENSE_WEIGHT,
    FEEDBACK_WORD_FEATURES,
    feedback_word_hasher,
    fit_feedback_model,
)
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES


class FeedbackUpgradeTests(unittest.TestCase):
    def test_feedback_upgrade_feature_contract(self) -> None:
        texts = [
            "[OBJECTIVE] fractions [ANSWER] one half [FEEDBACK] correct",
            "[OBJECTIVE] fractions [ANSWER] one third [FEEDBACK] try again",
            "[OBJECTIVE] area [ANSWER] multiply sides [FEEDBACK] exactly",
            "[OBJECTIVE] area [ANSWER] add sides [FEEDBACK] not quite",
            "[OBJECTIVE] decimals [ANSWER] point five [FEEDBACK] well done",
            "[OBJECTIVE] decimals [ANSWER] five point zero [FEEDBACK] check that",
        ]
        target = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
        word_matrix = feedback_word_hasher().transform(texts).tocsr()
        ordered = np.zeros((len(texts), len(ORDERED_FEATURE_NAMES)), dtype=np.float64)
        ordered[:, 0] = np.arange(len(texts), dtype=np.float64)

        model, scaler, training_shape = fit_feedback_model(word_matrix, ordered, target)

        self.assertEqual(word_matrix.shape, (len(texts), FEEDBACK_WORD_FEATURES))
        self.assertEqual(
            training_shape,
            (len(texts), FEEDBACK_WORD_FEATURES + len(ORDERED_FEATURE_NAMES)),
        )
        self.assertEqual(scaler.n_features_in_, len(ORDERED_FEATURE_NAMES))
        matrix = sparse.hstack(
            [
                word_matrix,
                sparse.csr_matrix(
                    scaler.transform(ordered).astype(np.float32)
                    * np.float32(FEEDBACK_DENSE_WEIGHT)
                ),
            ],
            format="csr",
        )
        probabilities = model.predict_proba(matrix)[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.all((probabilities > 0.0) & (probabilities < 1.0)))


if __name__ == "__main__":
    unittest.main()
