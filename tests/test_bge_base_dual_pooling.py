from __future__ import annotations

import numpy as np

from trace_ace.bge_base_dual_pooling import (
    EMBEDDING_DIMENSION,
    _load_pilot,
    dual_interaction_features,
    masked_mean_pool,
)


def test_masked_mean_pool_excludes_padding_and_normalizes() -> None:
    hidden = np.zeros((2, 3, EMBEDDING_DIMENSION), dtype=np.float32)
    hidden[0, 0, 0] = 1.0
    hidden[0, 1, 1] = 1.0
    hidden[0, 2, 2] = 100.0
    hidden[1, 0, 3] = 2.0
    hidden[1, 1, 3] = 2.0
    hidden[1, 2, 3] = 2.0
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    pooled = masked_mean_pool(hidden, mask)
    assert pooled.shape == (2, EMBEDDING_DIMENSION)
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)
    assert pooled[0, 2] == 0.0
    assert np.isclose(pooled[0, 0], pooled[0, 1])
    assert pooled[1, 3] == 1.0


def test_dual_features_have_frozen_width_and_finite_similarity() -> None:
    rng = np.random.default_rng(20260724)
    arrays = []
    for _ in range(4):
        values = rng.normal(size=(3, EMBEDDING_DIMENSION)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        arrays.append(values)
    features, similarity = dual_interaction_features(*arrays)
    assert features.shape == (3, 8 * EMBEDDING_DIMENSION)
    assert similarity.shape == (3, 1)
    assert np.isfinite(features).all()
    assert np.isfinite(similarity).all()


def test_real_pilot_contract() -> None:
    frame, texts, indices, folds = _load_pilot(".")
    assert frame.shape[0] == 4_096
    assert texts.shape == (4_096, 3)
    assert indices.shape == (4_096,)
    assert set(folds.tolist()) == set(range(5))
    assert set(frame["target"].tolist()) == {0, 1}
