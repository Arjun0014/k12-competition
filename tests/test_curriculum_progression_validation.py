from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.curriculum_progression_validation import (
    FEATURE_COLUMNS,
    _objective_equal_weights,
    align_curriculum_features,
    build_prediction_rows,
)


def _feature_row(objective_id: str, offset: float) -> dict[str, object]:
    return {
        "learning_objective_id": objective_id,
        **{
            name: float(index) / len(FEATURE_COLUMNS) + offset
            for index, name in enumerate(FEATURE_COLUMNS)
        },
    }


def test_align_curriculum_features_reuses_objective_values() -> None:
    component = pd.DataFrame(
        {"learning_objective_id": ["o2", "o1", "o2"]}
    )
    features = pd.DataFrame(
        [_feature_row("o1", 0.0), _feature_row("o2", 1.0)]
    )
    values = align_curriculum_features(component, features)
    assert values.shape == (3, len(FEATURE_COLUMNS))
    assert np.allclose(values[0], values[2])
    assert not np.allclose(values[0], values[1])


def test_objective_equal_weights_equalize_total_objective_mass() -> None:
    objective_ids = pd.Series(["a", "a", "a", "b"])
    weights = _objective_equal_weights(objective_ids)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[:3].sum(), weights[3])


def test_build_prediction_rows_uses_only_given_frozen_blends() -> None:
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
    rows = build_prediction_rows(
        component,
        np.asarray([0.50]),
        [0.10, 0.30],
    )
    assert list(rows["curriculum_weight"]) == [0.10, 0.30]
    assert np.allclose(rows["pred_v05_raw"], 0.725)
    assert np.allclose(rows["prediction"], [0.7025, 0.6575])
