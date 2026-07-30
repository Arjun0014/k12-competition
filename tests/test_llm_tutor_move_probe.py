from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.llm_tutor_move_probe import (
    LABELS,
    MAX_LENGTH,
    SCREEN_ROWS_PER_CLASS,
    _metric_summary,
    select_screen_rows,
)


def test_screen_selection_is_balanced_stable_and_label_complete() -> None:
    rows = []
    for label in LABELS:
        for index in range(SCREEN_ROWS_PER_CLASS + 7):
            rows.append(
                {
                    "split": "test",
                    "move": label,
                    "qid": f"{label}-{index // 3}",
                    "source_row": index,
                    "turn_index": index % 3,
                }
            )
    frame = pd.DataFrame(rows)
    first = select_screen_rows(frame)
    second = select_screen_rows(frame.sample(frac=1, random_state=5))
    assert first.equals(second)
    assert first["move"].value_counts().to_dict() == {
        label: SCREEN_ROWS_PER_CLASS for label in LABELS
    }


def test_metric_summary_uses_fixed_label_order() -> None:
    target = list(LABELS)
    probabilities = np.eye(len(LABELS), dtype=np.float64)
    metrics = _metric_summary(target, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class_f1"] == {label: 1.0 for label in LABELS}


def test_frozen_prompt_budget_is_declared() -> None:
    assert MAX_LENGTH == 256
