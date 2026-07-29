from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.productive_numeric_elaboration_validation import (
    FEATURE_NAMES,
    align_numeric_features,
    build_prediction_rows,
    fit_numeric_oof,
)


def test_align_numeric_features_reuses_session_values() -> None:
    component = pd.DataFrame({"session_id": ["s2", "s1", "s2"]})
    features = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "n_student_words": [200.0, 300.0],
            "numeric_turns_per_word": [0.10, 0.20],
            "digit_chars_per_word": [0.30, 0.40],
        }
    )
    values = align_numeric_features(component, features)
    assert values.shape == (3, len(FEATURE_NAMES))
    assert np.allclose(values[0], values[2])
    assert values[1, 0] == 200.0


def test_fit_numeric_oof_is_complete_and_uses_quiet_fallback() -> None:
    rows = 200
    component = pd.DataFrame(
        {
            "session_id": [f"s{index}" for index in range(rows)],
            "fold": np.arange(rows) % 5,
            "target": np.arange(rows) % 2,
        }
    )
    values = np.column_stack(
        [
            np.linspace(50.0, 500.0, rows),
            np.linspace(0.01, 0.10, rows),
            np.linspace(0.10, 0.50, rows),
        ]
    )
    prediction, summaries = fit_numeric_oof(component, values)
    assert np.isfinite(prediction).all()
    assert len(summaries) == 5
    assert sum(row["quiet_validation_rows"] for row in summaries) > 0


def test_build_prediction_rows_uses_only_preregistered_blends() -> None:
    component = pd.DataFrame(
        {
            "environment": ["V_seen"],
            "response_id": ["r1"],
            "session_id": ["s1"],
            "learning_objective_id": ["o1"],
            "semantic_family": ["f1"],
            "fold": [0],
            "evaluation_eligible": [True],
            "target": [1],
            "pred_full": [0.60],
            "pred_role": [0.70],
            "pred_bge_base": [0.80],
        }
    )
    rows = build_prediction_rows(component, np.asarray([0.50]), [0.10, 0.30])
    assert list(rows["numeric_weight"]) == [0.10, 0.30]
    assert np.allclose(rows["pred_v05_raw"], 0.725)
