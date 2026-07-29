from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.speaker_role_denoising import (
    _session_fold,
    _session_selected,
    _top_label_ece,
    enrich_session_text,
)


def test_enrichment_uses_neighbors_but_not_current_role_token() -> None:
    frame = pd.DataFrame(
        {
            "utterance_id": [0, 1, 2],
            "role": ["tutor", "student", "tutor"],
            "content": ["Can you explain?", "I think it is four.", "Correct."],
        }
    )
    texts = enrich_session_text(frame)
    assert "prev_role_boundary" in texts[0]
    assert "next_role_student" in texts[0]
    assert "prev_role_tutor" in texts[1]
    assert "next_role_tutor" in texts[1]
    assert "current_role" not in " ".join(texts)


def test_session_sampling_and_folds_are_deterministic() -> None:
    selected = [_session_selected(f"session-{index}") for index in range(100)]
    folds = [_session_fold(f"session-{index}") for index in range(100)]
    assert selected == [
        _session_selected(f"session-{index}") for index in range(100)
    ]
    assert folds == [_session_fold(f"session-{index}") for index in range(100)]
    assert 1 <= sum(selected) <= 10
    assert set(folds) == set(range(5))


def test_top_label_ece_is_zero_for_perfect_confidence() -> None:
    target = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probability = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert _top_label_ece(target, probability) == 0.0
