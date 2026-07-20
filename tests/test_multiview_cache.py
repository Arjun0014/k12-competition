import unittest

import numpy as np

from trace_ace.multiview_cache import (
    _feedback_windows,
    _parse_role_lines,
    _select_role_evidence,
    _session_trajectory,
)
from trace_ace.multiview_validation import _normalized_mean, build_variant


class MultiviewCacheTests(unittest.TestCase):
    def setUp(self):
        self.lines = _parse_role_lines(
            "\n".join(
                [
                    "[TUTOR] What is one half plus one quarter?",
                    "[STUDENT] I think it is two sixths.",
                    "[TUTOR] Not quite, use a common denominator.",
                    "[STUDENT] One half is two quarters, so it is three quarters.",
                    "[TUTOR] Correct, well done.",
                    "[BACKGROUND] page changed",
                    "[STUDENT] I understand the denominator now.",
                ]
            )
        )

    def test_role_evidence_keeps_objective_and_relevant_student_reasoning(self):
        text = _select_role_evidence(
            self.lines,
            "student",
            "Adding fractions using a common denominator",
        )
        self.assertIn("[OBJECTIVE]", text)
        self.assertIn("three quarters", text)
        self.assertNotIn("[TUTOR]", text)

    def test_feedback_windows_preserve_ordered_answer_feedback_pairs(self):
        text = _feedback_windows(
            self.lines,
            "Adding fractions using a common denominator",
        )
        self.assertIn("[ANSWER]", text)
        self.assertIn("[FEEDBACK]", text)
        self.assertIn("Correct", text)

    def test_trajectory_uses_dialogue_and_ignores_background(self):
        text = _session_trajectory(self.lines)
        self.assertTrue(text.startswith("[SESSION_TRAJECTORY]"))
        self.assertIn("[STUDENT]", text)
        self.assertNotIn("page changed", text)

    def test_normalized_mean_and_variants_are_finite_and_aligned(self):
        base = np.eye(4, dtype=np.float32)
        mean = _normalized_mean(base, np.roll(base, 1, axis=1))
        self.assertEqual(mean.shape, base.shape)
        np.testing.assert_allclose(np.linalg.norm(mean, axis=1), 1.0, atol=1e-6)
        blocks = {
            "context": base,
            "objective": base,
            "student": base,
            "tutor": base,
            "feedback": base,
            "trajectory": base,
        }
        self.assertEqual(build_variant("student_interaction", blocks).shape, (4, 16))
        self.assertEqual(
            build_variant("context_feedback_interaction", blocks).shape, (4, 28)
        )
        self.assertEqual(build_variant("multiview_interaction", blocks).shape, (4, 48))


if __name__ == "__main__":
    unittest.main()
