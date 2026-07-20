from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from trace_ace.source_robust_validation import (
    D100_FEATURE_WIDTH,
    D100_NAME,
    D100_SPEC,
    AxisGroups,
    _current_rss_bytes,
    _fold_checkpoint_paths,
    d100_selection_gate,
    merge_rare_axis_groups,
    robust_objective_and_gradient,
    validate_d100_runtime,
)


def _axis(name: str, values: list[int]) -> AxisGroups:
    codes = np.asarray(values, dtype=np.int32)
    return AxisGroups(
        name=name,
        codes=codes,
        labels=tuple(f"{name}:{index}" for index in range(int(codes.max()) + 1)),
        counts=np.bincount(codes).astype(np.int64),
    )


def test_d100_frozen_feature_width_is_exact() -> None:
    features = D100_SPEC["features"]
    assert features["total_width"] == D100_FEATURE_WIDTH
    assert 1536 + 131072 + 35 + 51 == D100_FEATURE_WIDTH
    assert D100_SPEC["candidate_formula"] == "0.50*v02 + 0.50*robust_head"


def test_current_process_rss_is_observable() -> None:
    assert _current_rss_bytes() > 0


def test_d100_runtime_matches_competition_aligned_environment() -> None:
    validate_d100_runtime()


def test_fold_checkpoint_paths_preserve_every_fold_number(tmp_path) -> None:
    observed = {
        _fold_checkpoint_paths(tmp_path, "V_seen", fold) for fold in range(5)
    }
    assert len(observed) == 5
    for fold in range(5):
        parquet, metadata = _fold_checkpoint_paths(tmp_path, "V_seen", fold)
        assert parquet.name == f"V_seen.d100.fold_{fold}.parquet"
        assert metadata.name == f"V_seen.d100.fold_{fold}.json"


def test_merge_rare_groups_counts_distinct_sessions() -> None:
    axis = ["large"] * 5 + ["small"] * 4
    sessions = ["a", "a", "b", "c", "d", "x", "x", "y", "y"]
    result = merge_rare_axis_groups(
        axis,
        sessions,
        axis_name="family",
        minimum_sessions=3,
    )
    assert result.labels == ("family:large", "family:rare")
    assert result.counts.tolist() == [5, 4]


def test_robust_objective_gradient_matches_finite_difference() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.2],
                [0.5, 0.5, 0.1],
                [-0.3, 0.2, 1.0],
                [0.7, -0.4, 0.0],
                [0.1, 0.8, -0.5],
            ],
            dtype=np.float64,
        )
    )
    target = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.float64)
    family = _axis("family", [0, 0, 0, 1, 1, 1])
    style = _axis("style", [0, 0, 1, 1, 2, 2])
    parameters = np.asarray([0.2, -0.1, 0.05, -0.3], dtype=np.float64)
    value, gradient = robust_objective_and_gradient(
        parameters, matrix, target, family, style
    )
    numeric = np.empty_like(parameters)
    epsilon = 1e-6
    for index in range(len(parameters)):
        left = parameters.copy()
        right = parameters.copy()
        left[index] -= epsilon
        right[index] += epsilon
        left_value, _ = robust_objective_and_gradient(
            left, matrix, target, family, style
        )
        right_value, _ = robust_objective_and_gradient(
            right, matrix, target, family, style
        )
        numeric[index] = (right_value - left_value) / (2.0 * epsilon)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numeric, rtol=2e-5, atol=2e-7)


def test_d100_selection_gate_requires_every_frozen_clause() -> None:
    metrics = pd.DataFrame(
        {
            "environment": ["V_seen", "V_objective", "V_style"],
            "candidate": [D100_NAME] * 3,
            "calibration": ["raw"] * 3,
            "delta_log_loss_vs_v02": [-0.0030, -0.0027, -0.0025],
            "delta_roc_auc_vs_v02": [0.001, 0.002, 0.001],
            "delta_ece_10_vs_v02": [-0.001, 0.0, -0.001],
            "delta_brier_score_vs_v02": [-0.001, -0.001, 0.0],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "environment": ["ALL_MACRO", "ALL_MACRO"],
            "resampler": ["session", "objective_family"],
            "probability_gain_positive": [0.95, 0.96],
        }
    )
    result = d100_selection_gate(metrics, bootstrap)
    assert result["passes_selection_gate"] is True
    metrics.loc[0, "delta_log_loss_vs_v02"] = -0.001
    metrics.loc[1, "delta_log_loss_vs_v02"] = -0.001
    metrics.loc[2, "delta_log_loss_vs_v02"] = -0.001
    result = d100_selection_gate(metrics, bootstrap)
    assert result["passes_selection_gate"] is False
    assert result["clauses"]["mean_log_loss_gain_at_least_0_0025"] is False
