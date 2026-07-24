from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.objective_pairwise_ranker import (
    build_objective_pairs,
    fit_pairwise_ranker,
)


def test_pairs_stay_within_objective_and_cycle_smaller_class() -> None:
    frame = pd.DataFrame(
        {
            "response_id": ["r4", "r1", "r3", "r2", "r5", "r6"],
            "learning_objective_id": ["a", "a", "a", "a", "b", "b"],
        }
    )
    labels = np.array([1, 0, 1, 1, 0, 0], dtype=np.int8)
    mask = np.ones(len(frame), dtype=bool)
    positive, negative, audit = build_objective_pairs(frame, mask, labels)
    assert len(positive) == len(negative) == 3
    assert audit["mixed_label_objectives"] == 1
    assert set(frame.iloc[positive]["learning_objective_id"]) == {"a"}
    assert set(frame.iloc[negative]["learning_objective_id"]) == {"a"}
    assert set(labels[positive]) == {1}
    assert set(labels[negative]) == {0}


def test_pairwise_fit_is_deterministic_and_ranks_training_pairs() -> None:
    x_train = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int8)
    positive = np.array([2, 3], dtype=np.int64)
    negative = np.array([0, 1], dtype=np.int64)
    first, first_summary = fit_pairwise_ranker(
        x_train, y_train, x_train, positive, negative
    )
    second, second_summary = fit_pairwise_ranker(
        x_train, y_train, x_train, positive, negative
    )
    assert np.array_equal(first, second)
    assert first_summary == second_summary
    assert first[2:].min() > first[:2].max()
