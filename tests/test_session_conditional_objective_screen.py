from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.session_conditional_objective_screen import (
    BLEND_WEIGHTS,
    build_prediction_rows,
    build_session_pairs,
    fit_conditional_head,
)


def test_pairs_are_cartesian_within_session_and_equal_weight_sessions() -> None:
    frame = pd.DataFrame(
        {
            "response_id": ["r3", "r1", "r2", "r5", "r4", "r6"],
            "session_id": ["s1", "s1", "s1", "s2", "s2", "s3"],
        }
    )
    labels = np.array([1, 0, 1, 0, 1, 1], dtype=np.int8)
    mask = np.ones(len(frame), dtype=bool)
    positive, negative, weights, audit = build_session_pairs(
        frame, mask, labels
    )
    assert audit["mixed_outcome_sessions"] == 2
    assert audit["positive_negative_pairs"] == 3
    assert np.array_equal(
        frame.iloc[positive]["session_id"].to_numpy(),
        frame.iloc[negative]["session_id"].to_numpy(),
    )
    assert set(labels[positive]) == {1}
    assert set(labels[negative]) == {0}
    session_weight = pd.DataFrame(
        {
            "session_id": frame.iloc[positive]["session_id"].to_numpy(),
            "symmetric_weight": 2.0 * weights,
        }
    ).groupby("session_id")["symmetric_weight"].sum()
    np.testing.assert_allclose(session_weight.to_numpy(), np.ones(2))


def test_conditional_fit_is_deterministic_and_ranks_pairs() -> None:
    features = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    positive = np.array([2, 3], dtype=np.int64)
    negative = np.array([0, 1], dtype=np.int64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    first, first_summary = fit_conditional_head(
        features,
        positive,
        negative,
        weights,
        features,
        training_prior=0.5,
    )
    second, second_summary = fit_conditional_head(
        features,
        positive,
        negative,
        weights,
        features,
        training_prior=0.5,
    )
    np.testing.assert_array_equal(first, second)
    assert first_summary == second_summary
    assert first[2:].min() > first[:2].max()


def test_prediction_registry_contains_only_frozen_weights_and_raw_v05() -> None:
    component = pd.DataFrame(
        {
            "environment": ["V_seen", "V_seen"],
            "response_id": ["r1", "r2"],
            "session_id": ["s1", "s2"],
            "learning_objective_id": ["o1", "o2"],
            "semantic_family": ["f1", "f2"],
            "fold": [0, 1],
            "evaluation_eligible": [True, True],
            "target": [0, 1],
            "pred_full": [0.2, 0.8],
            "pred_role": [0.3, 0.7],
            "pred_bge_base": [0.5, 0.5],
        }
    )
    conditional = np.array([0.1, 0.9])
    result = build_prediction_rows(component, conditional, BLEND_WEIGHTS)
    assert sorted(result["conditional_weight"].unique()) == list(BLEND_WEIGHTS)
    raw_v05 = np.array([0.375, 0.625])
    selected = result.loc[result["conditional_weight"].eq(0.1)]
    np.testing.assert_allclose(selected["pred_v05_raw"], raw_v05)
    np.testing.assert_allclose(
        selected["prediction"], 0.9 * raw_v05 + 0.1 * conditional
    )
