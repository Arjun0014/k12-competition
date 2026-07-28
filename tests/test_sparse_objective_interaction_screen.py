from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.sparse_objective_interaction_screen import (
    HASH_FEATURES,
    _unique_tokens,
    build_matrix,
    interaction_features,
)


def test_unique_tokens_preserves_first_occurrence_and_cap() -> None:
    assert _unique_tokens("Two TWO plus three four", 3) == ["two", "plus", "three"]


def test_interaction_features_separate_roles_and_cross_objective() -> None:
    values = interaction_features(
        "Add two numbers",
        "[STUDENT] I add two.\n[TUTOR] Why add?\n[BACKGROUND] ignored add",
    )
    assert values["o=add"] == 1.0
    assert values["s=two"] == 1.0
    assert values["t=why"] == 1.0
    assert values["os=add|two"] == 1.0
    assert values["ot=add|why"] == 1.0
    assert "s=ignored" not in values


def test_build_matrix_is_aligned_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "response_id": ["r1", "r2"],
            "learning_objective": ["Add numbers", "Subtract numbers"],
        }
    )
    contexts = pd.DataFrame(
        {
            "response_id": ["r1", "r2"],
            "objective_context": [
                "[STUDENT] add one\n[TUTOR] correct",
                "[STUDENT] take away\n[TUTOR] try",
            ],
        }
    )
    matrix = build_matrix(frame, contexts)
    assert matrix.shape == (2, HASH_FEATURES)
    assert matrix.nnz > 0
    assert np.isfinite(matrix.data).all()
