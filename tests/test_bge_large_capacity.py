import numpy as np
import pytest

from trace_ace.bge_large_capacity import (
    BENCHMARK_ROWS,
    EMBEDDING_DIMENSION,
    PILOT_ROWS,
    interaction_features,
    select_benchmark_indices,
)


def test_benchmark_indices_span_length_quantiles() -> None:
    lengths = np.arange(PILOT_ROWS, dtype=np.int32)[::-1]
    selected = select_benchmark_indices(lengths)
    assert selected.shape == (BENCHMARK_ROWS,)
    assert len(np.unique(selected)) == BENCHMARK_ROWS
    selected_lengths = np.sort(lengths[selected])
    assert selected_lengths[0] == 0
    assert selected_lengths[-1] == PILOT_ROWS - 1


def test_benchmark_indices_require_frozen_pilot_size() -> None:
    with pytest.raises(ValueError, match="must cover the pilot"):
        select_benchmark_indices(np.arange(100, dtype=np.int32))


def test_interaction_matches_frozen_scaling() -> None:
    context = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
    objective = np.zeros_like(context)
    context[0, 0] = 1.0
    objective[0, 0] = 1.0
    context[1, 1] = 1.0
    objective[1, 2] = 1.0
    features, similarity = interaction_features(context, objective)
    assert features.shape == (2, EMBEDDING_DIMENSION * 4)
    assert features.dtype == np.float32
    assert similarity.shape == (2, 1)
    np.testing.assert_allclose(similarity[:, 0], [1.0, 0.0])
    assert features[0, EMBEDDING_DIMENSION * 2] == pytest.approx(6.0)
    assert features[1, EMBEDDING_DIMENSION * 3 + 1] == pytest.approx(0.7)
    assert features[1, EMBEDDING_DIMENSION * 3 + 2] == pytest.approx(0.7)


def test_interaction_rejects_wrong_dimension() -> None:
    values = np.zeros((2, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="embedding shape"):
        interaction_features(values, values)
