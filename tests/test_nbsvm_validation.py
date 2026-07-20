import unittest

import numpy as np
from scipy import sparse

from trace_ace.nbsvm_validation import nb_log_count_ratio, role_prefixed_feedback_text


class NbSvmValidationTests(unittest.TestCase):
    def test_log_count_ratio_has_expected_direction(self):
        matrix = sparse.csr_matrix(
            np.asarray(
                [
                    [4.0, 0.0, 1.0],
                    [3.0, 0.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 3.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        ratio = nb_log_count_ratio(matrix, np.asarray([1, 1, 0, 0]))
        self.assertGreater(ratio[0], 0.0)
        self.assertLess(ratio[1], 0.0)
        self.assertAlmostEqual(float(ratio[2]), 0.0, places=6)
        self.assertTrue(np.isfinite(ratio).all())

    def test_role_prefixes_apply_to_every_token(self):
        rendered = role_prefixed_feedback_text(
            "[OBJECTIVE] Explain equivalent fractions.\n"
            "[ANSWER] I think they are correct.\n"
            "[FEEDBACK] Correct, because both fractions match."
        )
        tokens = set(rendered.split())
        self.assertIn("learning_objective_equivalent", tokens)
        self.assertIn("student_speech_correct", tokens)
        self.assertIn("tutor_feedback_correct", tokens)
        self.assertIn("tutor_feedback_because", tokens)
        self.assertNotIn("correct", tokens)


if __name__ == "__main__":
    unittest.main()
