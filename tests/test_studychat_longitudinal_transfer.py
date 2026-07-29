from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.studychat_longitudinal_transfer import (
    BEHAVIOR_FEATURES,
    _privacy_hash,
    dialogue_features,
    transfer_distribution_audit,
)


def test_dialogue_features_separate_productive_and_direct_requests() -> None:
    features = dialogue_features(
        [
            "Can you explain why this method works? I think it is because the values are sorted.",
            "Can you check whether my reasoning is correct?",
            "Just give me the answer and write my report.",
        ],
        [
            "This works because each step preserves the ordering.",
            "What should the next step be?",
            "Here is an example with code: def solve(x): return x",
        ],
        multi_turn_fraction=2 / 3,
    )
    assert set(features) == set(BEHAVIOR_FEATURES)
    assert features["student_explanation_request_rate"] > 0
    assert features["student_verification_rate"] > 0
    assert features["student_direct_answer_rate"] > 0
    assert features["student_writing_request_rate"] > 0
    assert features["student_self_explanation_rate"] > 0
    assert features["tutor_explanation_rate"] > 0


def test_privacy_hash_is_deterministic_and_semester_scoped() -> None:
    first = _privacy_hash("f24", "user-1")
    assert first == _privacy_hash("f24", "user-1")
    assert first != _privacy_hash("s25", "user-1")
    assert "user-1" not in first


def test_transfer_distribution_audit_detects_overlap() -> None:
    external = pd.DataFrame(
        {
            name: np.linspace(0.0, 1.0, 20) + index
            for index, name in enumerate(BEHAVIOR_FEATURES)
        }
    )
    competition = external.copy()
    report = transfer_distribution_audit(external, competition)
    assert report["nonconstant_external_features"] == len(BEHAVIOR_FEATURES)
    assert report["nonconstant_competition_features"] == len(BEHAVIOR_FEATURES)
    assert report["feature_overlap_fraction"] == 1.0
