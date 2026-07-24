from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trace_ace.timing_dynamics_validation import (
    BLEND_WEIGHTS,
    TIMING_FEATURE_NAMES,
    _gap_bin_entropy,
    align_timing_features,
    build_prediction_rows,
    session_timing_feature_row,
)


def _session() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1"] * 8,
            "utterance_id": list(range(8)),
            "timestamp": [
                "00:00:00",
                "00:00:02",
                "00:00:03",
                "00:00:03",
                "00:00:09",
                "00:00:12",
                "00:00:25",
                "00:00:31",
            ],
            "role": [
                "tutor",
                "student",
                "tutor",
                "student",
                "tutor",
                "student",
                "background",
                "tutor",
            ],
        }
    )


def test_session_timing_features_are_fixed_finite_and_order_invariant() -> None:
    first = session_timing_feature_row(_session())
    second = session_timing_feature_row(
        _session().sample(frac=1.0, random_state=9).reset_index(drop=True)
    )
    assert list(first) == ["session_id", *TIMING_FEATURE_NAMES]
    assert first == second
    assert first["timing_zero_gap_share"] == pytest.approx(1 / 7)
    assert first["timing_tutor_student_zero_share"] == pytest.approx(1 / 3)
    assert np.isfinite(
        np.asarray([first[name] for name in TIMING_FEATURE_NAMES], dtype=np.float64)
    ).all()


def test_non_monotonic_timestamps_are_rejected() -> None:
    frame = _session()
    frame.loc[4, "timestamp"] = "00:00:01"
    with pytest.raises(ValueError, match="Non-monotonic"):
        session_timing_feature_row(frame)


def test_gap_entropy_is_bounded_and_zero_for_one_bin() -> None:
    assert _gap_bin_entropy(np.array([0, 0, 0])) == pytest.approx(0.0)
    entropy = _gap_bin_entropy(np.array([0, 1, 2, 3, 6, 11, 31]))
    assert entropy == pytest.approx(1.0)


def test_alignment_preserves_component_row_order() -> None:
    component = pd.DataFrame({"session_id": ["s2", "s1", "s2"]})
    timing = pd.DataFrame(
        [
            {"session_id": "s1", **dict.fromkeys(TIMING_FEATURE_NAMES, 1.0)},
            {"session_id": "s2", **dict.fromkeys(TIMING_FEATURE_NAMES, 2.0)},
        ]
    )
    values = align_timing_features(component, timing)
    assert values.shape == (3, len(TIMING_FEATURE_NAMES))
    assert values[:, 0].tolist() == [2.0, 1.0, 2.0]


def test_prediction_registry_contains_only_frozen_weights() -> None:
    component = pd.DataFrame(
        {
            "environment": ["V_seen", "V_seen"],
            "response_id": ["r1", "r2"],
            "session_id": ["s1", "s2"],
            "learning_objective_id": ["o1", "o2"],
            "semantic_family": [0, 1],
            "fold": [0, 1],
            "evaluation_eligible": [True, True],
            "target": [0, 1],
            "pred_full": [0.2, 0.8],
            "pred_role": [0.3, 0.7],
            "pred_bge_small": [0.4, 0.6],
            "pred_bge_base": [0.5, 0.5],
        }
    )
    timing = np.array([0.1, 0.9])
    result = build_prediction_rows(component, timing, BLEND_WEIGHTS)
    assert sorted(result["timing_weight"].unique()) == list(BLEND_WEIGHTS)
    expected_bge = np.array([0.375, 0.625])
    np.testing.assert_allclose(
        result.loc[result["timing_weight"].eq(0.1), "pred_bge_replace"],
        expected_bge,
    )
    np.testing.assert_allclose(
        result.loc[result["timing_weight"].eq(0.1), "prediction"],
        0.9 * expected_bge + 0.1 * timing,
    )
