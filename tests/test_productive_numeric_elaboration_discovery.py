from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.productive_numeric_elaboration_discovery import (
    FEATURE_NAMES,
    _assert_no_outcomes,
    _fold,
    _published_summary,
    count_student_turn,
)


def test_count_student_turn_matches_published_regex_contract() -> None:
    assert count_student_turn("2,000 plus 725 would be 2,725.") == (8, 1, 11)
    assert count_student_turn("Twenty-four isn't 24.") == (4, 1, 2)
    assert count_student_turn(None) == (0, 0, 0)


def test_fold_is_stable_and_bounded() -> None:
    assert _fold("session-a") == _fold("session-a")
    assert 0 <= _fold("session-a") < 5


def test_outcome_guard_rejects_target_column() -> None:
    frame = pd.DataFrame({"session_id": ["s1"], "target": [1]})
    try:
        _assert_no_outcomes(frame, "unit test")
    except ValueError as error:
        assert "outcomes" in str(error)
    else:
        raise AssertionError("Outcome guard did not reject target.")


def test_published_summary_counts_quiet_rows_without_outcomes() -> None:
    sessions = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "n_student_words": [200.0, np.nan],
            "numeric_turns_per_word": [0.05, np.nan],
            "digit_chars_per_word": [0.20, np.nan],
        }
    )
    responses = pd.DataFrame(
        {
            "response_id": ["r1", "r2", "r3"],
            "session_id": ["s1", "s2", "s2"],
            "n_student_words": [200.0, np.nan, np.nan],
            "numeric_turns_per_word": [0.05, np.nan, np.nan],
            "digit_chars_per_word": [0.20, np.nan, np.nan],
        }
    )
    summary, match = _published_summary(responses, sessions)
    assert not match
    assert summary["quiet_sessions"] == 1
    assert summary["quiet_responses"] == 2
    assert list(sessions.columns[1:]) == list(FEATURE_NAMES)
