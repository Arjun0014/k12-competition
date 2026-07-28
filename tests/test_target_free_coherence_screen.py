from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.target_free_coherence_screen import (
    AGGREGATE_NAMES,
    _aggregate_scores,
    _ssl_fold,
    extract_pairs,
    pair_features,
)


def test_ssl_fold_is_stable() -> None:
    assert _ssl_fold("session-a") == _ssl_fold("session-a")
    assert 0 <= _ssl_fold("session-a") < 5


def test_pair_features_include_cross_and_numeric_features() -> None:
    values = pair_features("Why is ten plus two twelve?", "Because 10 plus 2 is 12.")
    assert values["shared=plus"] == 1.0
    assert values["x=why|because"] == 1.0
    assert values["n:tutor_question"] == 1.0
    assert values["n:jaccard"] > 0


def test_extract_pairs_uses_next_student_as_negative() -> None:
    frame = pd.DataFrame(
        {"response_id": ["r1"], "session_id": ["s1"], "target": [0]}
    )
    contexts = pd.DataFrame(
        {
            "response_id": ["r1"],
            "session_id": ["s1"],
            "objective_context": [
                "[OBJECTIVE] Add.\n"
                "[TUTOR] First?\n[STUDENT] One.\n"
                "[TUTOR] Second?\n[STUDENT] Two."
            ],
        }
    )
    pairs, positive, negative = extract_pairs(frame, contexts)
    assert len(pairs) == 2
    assert "s=one" in positive[0]
    assert "s=two" in negative[0]
    assert pairs["ssl_fold"].nunique() == 1


def test_aggregate_scores_shape_and_delta() -> None:
    pairs = pd.DataFrame(
        {
            "row_index": [0, 0, 0, 0],
            "sequence": [0, 1, 2, 3],
        }
    )
    values = _aggregate_scores(pairs, np.array([0.1, 0.2, 0.7, 0.8]), 2)
    assert values.shape == (2, len(AGGREGATE_NAMES))
    assert np.isclose(values[0, 0], 4)
    assert values[0, 9] > 0
    assert np.all(values[1] == 0)
