import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.multiview_cache import _feedback_windows, _parse_role_lines
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES, extract_ordered_features


def load_submission_module():
    path = Path(__file__).resolve().parents[1] / "submission_src" / "main.py"
    spec = importlib.util.spec_from_file_location("trace_ace_submission_main", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SubmissionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_submission_module()
        cls.asset = {
            "global_prior": 0.6,
            "probability_clip": 1e-6,
            "objective_priors": {"known": 0.8},
        }

    def test_unknown_objective_uses_global_fallback(self):
        features = pd.DataFrame(
            {
                "response_id": ["a", "b"],
                "learning_objective_id": ["known", "unknown"],
            }
        )
        prediction = self.module.predict(features, self.asset)
        np.testing.assert_allclose(prediction, [0.8, 0.6])

    def test_prediction_is_batch_invariant(self):
        batch = pd.DataFrame(
            {
                "response_id": ["a", "b", "c"],
                "learning_objective_id": ["known", "unknown", "known"],
            }
        )
        batch_prediction = self.module.predict(batch, self.asset)
        single_predictions = np.array(
            [self.module.predict(batch.iloc[[index]], self.asset)[0] for index in range(len(batch))]
        )
        np.testing.assert_allclose(batch_prediction, single_predictions)

    def test_v03_feedback_preprocessing_matches_training_code(self):
        lines = _parse_role_lines(
            "\n".join(
                [
                    "[TUTOR] What is one half plus one quarter?",
                    "[STUDENT] I think it may be two sixths.",
                    "[TUTOR] Not quite, remember the common denominator.",
                    "[STUDENT] One half is two quarters, so it is three quarters.",
                    "[TUTOR] Correct, well done.",
                    "[BACKGROUND] page changed",
                ]
            )
        )
        objective = "Adding fractions using a common denominator"
        self.assertEqual(
            self.module._feedback_windows(lines, objective),
            _feedback_windows(lines, objective),
        )
        expected = extract_ordered_features(lines, objective)
        np.testing.assert_allclose(
            self.module._ordered_feature_values(lines, objective),
            np.asarray([expected[name] for name in ORDERED_FEATURE_NAMES]),
            rtol=0.0,
            atol=0.0,
        )

    def test_duplicate_response_id_is_rejected(self):
        features = pd.DataFrame(
            {
                "response_id": ["a", "a"],
                "learning_objective_id": ["known", "unknown"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Duplicate response_id"):
            self.module.predict(features, self.asset)

    def test_transcript_text_is_stably_ordered_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.csv"
            pd.DataFrame(
                {
                    "session_id": ["s", "s"],
                    "utterance_id": [1, 0],
                    "role": ["student", "tutor"],
                    "content": ["Answer", "Question"],
                    "timestamp": ["00:00:02", "00:00:01"],
                }
            ).to_csv(path, index=False)
            self.assertEqual(
                self.module.transcript_text(path),
                "[TUTOR] Question\n[STUDENT] Answer",
            )


if __name__ == "__main__":
    unittest.main()
