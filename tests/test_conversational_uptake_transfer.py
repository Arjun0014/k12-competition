from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.conversational_uptake_transfer import (
    FOLDS,
    _assign_group_folds,
    _fold_audit,
    _ordinal_prediction,
    hashed_pair_matrix,
    repetition_features,
    sparse_pair_records,
)


def test_directional_pair_features_change_when_roles_reverse() -> None:
    student = ["I added thirty to seventy"]
    teacher = ["Why did you add thirty?"]
    forward = sparse_pair_records(student, teacher)[0]
    reverse = sparse_pair_records(teacher, student)[0]
    assert forward != reverse
    assert forward["s:u:added"] == 1.0
    assert forward["t:b:why_did"] == 1.0
    assert forward["shared:thirty"] == 1.0
    assert forward["cross:added>why"] == 1.0


def test_pair_and_repetition_matrices_are_finite_and_aligned() -> None:
    student = ["I think it is twelve.", "Could it be four?"]
    teacher = ["Why is it twelve?", "Yes, four is correct."]
    repetition = repetition_features(student, teacher)
    hashed = hashed_pair_matrix(student, teacher)
    assert repetition.shape == (2, 10)
    assert hashed.shape[0] == 2
    assert np.isfinite(repetition).all()
    assert np.isfinite(hashed.data).all()
    assert hashed.data.min() >= 0.0
    assert hashed.nnz > 0


def test_group_folds_are_deterministic_and_disjoint() -> None:
    rows = []
    for group in range(20):
        for exchange in range(2):
            rows.append(
                {
                    "obs_id": f"obs-{group:02d}",
                    "uptake_zscore": float(group + exchange),
                }
            )
    frame = pd.DataFrame(rows)
    first = _assign_group_folds(frame)
    second = _assign_group_folds(frame)
    np.testing.assert_array_equal(first, second)
    frame["fold"] = first
    audit = _fold_audit(frame)
    assert len(audit) == FOLDS
    assert set(first.tolist()) == set(range(FOLDS))
    assert all(row["group_overlap"] == 0 for row in audit)


def test_ordinal_prediction_respects_frozen_thresholds() -> None:
    prediction = np.array([-1.0, -0.1, 0.2, 0.8])
    actual = _ordinal_prediction(prediction, (-0.2, 0.5))
    np.testing.assert_array_equal(actual, np.array([0, 1, 1, 2]))
