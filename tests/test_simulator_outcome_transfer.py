from __future__ import annotations

import pandas as pd

from trace_ace.simulator_outcome_transfer import (
    TASK_LINE,
    _audit_folds,
    canonical_dialogue_text,
    fold_for_worker,
)


def test_canonical_dialogue_uses_only_first_problem_chat() -> None:
    row = {
        "user_queries": ["first student", "second-problem sentinel"],
        "ai_responses": ["first tutor", "second tutor sentinel"],
        "problem_1_turns": 1,
        "turk_final_answer": "forbidden-final-answer",
        "turk_solution": "forbidden-solution",
        "solve_or_not": "yes",
        "overall_rating": 10,
    }
    text = canonical_dialogue_text(row)
    assert text == "\n".join(
        ["Student: first student", "Tutor: first tutor", TASK_LINE]
    )
    assert "sentinel" not in text
    assert "forbidden" not in text


def test_canonical_dialogue_falls_back_to_all_available_pairs() -> None:
    row = {
        "user_queries": [" a   b ", "c"],
        "ai_responses": [" d\n e ", "f"],
        "problem_1_turns": -1,
    }
    text = canonical_dialogue_text(row)
    assert "Student: a b" in text
    assert "Tutor: d e" in text
    assert text.endswith(TASK_LINE)


def test_worker_fold_is_stable() -> None:
    assert fold_for_worker("worker-a") == fold_for_worker("worker-a")
    assert 0 <= fold_for_worker("worker-b") < 5


def test_fold_audit_purges_validation_problems() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for target in (0, 1):
            rows.append(
                {
                    "worker_id": f"worker-{fold}-{target}",
                    "problem_id": 100 * fold + target,
                    "fold": fold,
                    "target": target,
                }
            )
    frame = pd.DataFrame(rows)
    audit = _audit_folds(frame)
    assert len(audit) == 5
    assert all(row["worker_overlap"] == 0 for row in audit)
    assert all(row["problem_overlap"] == 0 for row in audit)
