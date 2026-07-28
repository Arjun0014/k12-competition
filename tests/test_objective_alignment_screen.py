from __future__ import annotations

import numpy as np

from trace_ace.objective_alignment_screen import (
    EMBEDDING_DIMENSION,
    aligned_features,
    fit_procrustes,
    transform_aligned,
)


def test_procrustes_is_orthogonal_and_recovers_rotation() -> None:
    rng = np.random.default_rng(7)
    context = rng.normal(size=(900, EMBEDDING_DIMENSION)).astype(np.float32)
    signs = np.where(np.arange(EMBEDDING_DIMENSION) % 2 == 0, 1.0, -1.0)
    objective = (context * signs).astype(np.float32)
    context_mean, objective_mean, rotation = fit_procrustes(context, objective)
    mapped, centered = transform_aligned(
        context, objective, context_mean, objective_mean, rotation
    )
    assert np.allclose(rotation.T @ rotation, np.eye(EMBEDDING_DIMENSION), atol=2e-10)
    assert float(np.mean(np.sum(mapped * centered, axis=1))) > 0.999


def test_aligned_features_have_frozen_shape_and_similarity() -> None:
    rng = np.random.default_rng(11)
    left = rng.normal(size=(3, EMBEDDING_DIMENSION)).astype(np.float32)
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    features, similarity = aligned_features(left, left)
    assert features.shape == (3, 4 * EMBEDDING_DIMENSION)
    assert similarity.shape == (3, 1)
    assert np.allclose(similarity, 1.0, atol=1e-6)
