from __future__ import annotations

import numpy as np
import pytest

from trace_ace.bge_base_multiview_screen import (
    BENCHMARK_ROWS_PER_VIEW,
    EMBEDDING_DIMENSION,
    PILOT_ROWS,
    interaction_features,
    normalized_mean,
    select_length_quantiles,
)


def test_normalized_mean_is_unit_length() -> None:
    left = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
    right = np.zeros_like(left)
    left[0, 0] = 1
    right[0, 1] = 1
    left[1, 2] = 1
    right[1, 2] = 1
    values = normalized_mean(left, right)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)
    assert values[0, 0] == pytest.approx(1 / np.sqrt(2))
    assert values[1, 2] == pytest.approx(1.0)


def test_normalized_mean_rejects_cancelled_rows() -> None:
    left = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
    right = np.zeros_like(left)
    left[0, 0] = 1
    right[0, 0] = -1
    with pytest.raises(ValueError, match="undefined"):
        normalized_mean(left, right)


def test_interaction_features_match_frozen_scaling() -> None:
    context = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
    objective = np.zeros_like(context)
    context[0, 0] = 1
    objective[0, 0] = 1
    context[1, 1] = 1
    objective[1, 2] = 1
    features, similarity = interaction_features(context, objective)
    assert features.shape == (2, EMBEDDING_DIMENSION * 4)
    assert similarity[:, 0].tolist() == pytest.approx([1.0, 0.0])
    assert features[0, EMBEDDING_DIMENSION * 2] == pytest.approx(6.0)
    assert features[1, EMBEDDING_DIMENSION * 3 + 1] == pytest.approx(0.7)
    assert features[1, EMBEDDING_DIMENSION * 3 + 2] == pytest.approx(0.7)


def test_length_quantiles_are_fixed_unique_and_span_order() -> None:
    lengths = np.arange(PILOT_ROWS, dtype=np.int32)[::-1]
    selected = select_length_quantiles(lengths)
    assert selected.shape == (BENCHMARK_ROWS_PER_VIEW,)
    assert len(np.unique(selected)) == BENCHMARK_ROWS_PER_VIEW
    assert lengths[selected[0]] == 0
    assert lengths[selected[-1]] == PILOT_ROWS - 1
