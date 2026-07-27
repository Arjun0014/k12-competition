from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.session_bagging_screen import (
    _load_features,
    assign_session_slices,
    session_slice,
)


def test_session_slice_is_stable_and_bounded() -> None:
    values = [session_slice(value) for value in ("a", "b", "c", "a")]
    assert values[0] == values[3]
    assert all(0 <= value < 5 for value in values)


def test_assign_session_slices_never_splits_session() -> None:
    frame = pd.DataFrame({"session_id": ["a", "a", "b", "c", "c"]})
    first = assign_session_slices(frame)
    second = assign_session_slices(frame)
    assert np.array_equal(first, second)
    assert first[0] == first[1]
    assert first[3] == first[4]


def test_real_pilot_feature_contract() -> None:
    frame, indices, folds, semantic, dense = _load_features(".")
    assert frame.shape[0] == 4_096
    assert indices.shape == (4_096,)
    assert semantic.shape == (4_096, 3_072)
    assert dense.shape == (4_096, 35)
    assert set(folds.tolist()) == set(range(5))
