from __future__ import annotations

import math

import numpy as np
import torch

from trace_ace.residual_head_screen import (
    ResidualOutcomeHead,
    _load_features,
    train_residual_head,
)


def test_residual_head_initializes_to_training_prior() -> None:
    prior = 0.7
    model = ResidualOutcomeHead(input_width=12, prior=prior, seed=20260725)
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.randn(5, 12))).numpy()
    assert np.allclose(probability, prior, atol=1e-7)
    assert math.isclose(
        float(model.bias.detach()), math.log(prior / (1 - prior)), abs_tol=2e-8
    )


def test_residual_training_is_deterministic() -> None:
    rng = np.random.default_rng(20260725)
    x_train = rng.normal(size=(64, 20)).astype(np.float32)
    y_train = np.tile(np.array([0, 1], dtype=np.int8), 32)
    x_validation = rng.normal(size=(12, 20)).astype(np.float32)
    first, first_summary = train_residual_head(
        x_train, y_train, x_validation, seed=17, epochs=2
    )
    second, second_summary = train_residual_head(
        x_train, y_train, x_validation, seed=17, epochs=2
    )
    assert np.array_equal(first, second)
    assert first_summary == second_summary


def test_real_pilot_feature_contract() -> None:
    frame, indices, folds, semantic, dense = _load_features(".")
    assert frame.shape[0] == 4_096
    assert indices.shape == (4_096,)
    assert semantic.shape == (4_096, 3_072)
    assert dense.shape == (4_096, 35)
    assert set(folds.tolist()) == set(range(5))
