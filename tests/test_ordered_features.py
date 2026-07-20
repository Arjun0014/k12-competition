import unittest

from trace_ace.multiview_cache import _parse_role_lines
from trace_ace.feedback_hash_validation import feedback_event_text
from trace_ace.ordered_features import (
    ORDERED_FEATURE_NAMES,
    extract_ordered_features,
)


class OrderedFeatureTests(unittest.TestCase):
    def test_correction_then_improved_answer_and_affirmation_is_repair(self):
        lines = _parse_role_lines(
            "\n".join(
                [
                    "[TUTOR] Add one half and one quarter.",
                    "[STUDENT] I am not sure, maybe two sixths.",
                    "[TUTOR] Not quite, check the common denominator.",
                    "[STUDENT] One half is two quarters, so the answer is three quarters.",
                    "[TUTOR] Correct, well done.",
                ]
            )
        )
        features = extract_ordered_features(
            lines, "Represent one half as two quarters"
        )
        self.assertEqual(list(features), ORDERED_FEATURE_NAMES)
        self.assertEqual(features["ordered_repair_opportunities"], 1.0)
        self.assertEqual(features["ordered_repair_overlap_improved"], 1.0)
        self.assertEqual(features["ordered_repair_later_affirm"], 1.0)
        self.assertGreater(features["ordered_student_objective_delta"], 0.0)

    def test_background_is_ignored_and_values_are_finite(self):
        lines = _parse_role_lines(
            "[BACKGROUND] noise\n[TUTOR] Try again.\n[STUDENT] I cannot answer."
        )
        features = extract_ordered_features(lines, "multiplication")
        self.assertEqual(features["ordered_dialogue_turns"], 2.0)
        self.assertEqual(features["ordered_tutor_turns"], 1.0)
        self.assertTrue(all(isinstance(value, float) for value in features.values()))

    def test_feedback_event_text_encodes_pair_interactions(self):
        view = "\n".join(
            [
                "[OBJECTIVE] Represent one half as two quarters",
                "[ANSWER] I think one half is two quarters because they are equal",
                "[FEEDBACK] Correct, well done.",
            ]
        )
        events = feedback_event_text(view)
        self.assertIn("pair_reasoning_affirm", events)
        self.assertIn("pair_obj_high_affirm", events)
        self.assertIn("pair_position_early", events)


if __name__ == "__main__":
    unittest.main()
