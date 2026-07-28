from __future__ import annotations

import pandas as pd

from trace_ace.simulator_quality_transfer import (
    TASK_LINE,
    _audit_folds,
    fold_for_worker,
    quality_dialogue_text,
)


def test_quality_dialogue_excludes_rating_and_second_problem() -> None:
    row = {
        "user_queries": ["first student", "second sentinel"],
        "ai_responses": ["first tutor", "second tutor sentinel"],
        "problem_1_turns": 1,
        "overall_rating": 10,
        "turk_final_answer": "forbidden",
    }
    text = quality_dialogue_text(row)
    assert text == "\n".join(
        ["Student: first student", "Tutor: first tutor", TASK_LINE]
    )
    assert "sentinel" not in text
    assert "forbidden" not in text
    assert "10" not in text


def test_quality_worker_fold_is_stable() -> None:
    assert fold_for_worker("worker-a") == fold_for_worker("worker-a")
    assert 0 <= fold_for_worker("worker-b") < 5


def test_quality_fold_audit_purges_validation_problems() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        rows.append(
            {
                "worker_id": f"worker-{fold}",
                "problem_id": fold,
                "fold": fold,
                "target": fold / 4,
            }
        )
    audit = _audit_folds(pd.DataFrame(rows))
    assert len(audit) == 5
    assert all(row["worker_overlap"] == 0 for row in audit)
    assert all(row["problem_overlap"] == 0 for row in audit)
