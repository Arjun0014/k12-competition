from __future__ import annotations

import pandas as pd

from trace_ace.cima_student_state_transfer import (
    TASK_LINE,
    _audit_protocols,
    _boolean,
    canonical_dialogue_text,
    concept_key,
    external_gate,
    group_fold,
    protocol_splits,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for target in (0, 1):
            rows.append(
                {
                    "example_id": f"row-{fold}-{target}",
                    "exercise_id": f"exercise-{fold}",
                    "concept_id": f"concept-{fold}",
                    "exercise_fold": fold,
                    "concept_fold": fold,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def test_canonical_dialogue_text_preserves_roles_and_is_target_free() -> None:
    text = canonical_dialogue_text(
        [
            "Please answer in Italian.",
            "What is table?",
            "Table is tavolo.",
            "Is it tavolo blu?",
        ]
    )
    assert text.splitlines()[-1] == TASK_LINE
    assert text.splitlines()[0].startswith("Tutor:")
    assert text.splitlines()[-2].startswith("Student:")
    assert "studentactions" not in text.casefold()


def test_boolean_parser_is_literal() -> None:
    assert _boolean("True") is True
    assert _boolean(False) is False


def test_concept_and_group_folds_are_stable() -> None:
    row = {"engPrep": "is under", "engObj": "table", "engColor": "blue"}
    assert concept_key(row) == '["is under","table","blue"]'
    assert group_fold("exercise_disjoint", "image-1") == group_fold(
        "exercise_disjoint", "image-1"
    )
    assert 0 <= group_fold("concept_disjoint", concept_key(row)) < 5


def test_protocols_hold_out_groups_and_both_classes() -> None:
    frame = _frame()
    for protocol, column in (
        ("exercise_disjoint", "exercise_id"),
        ("concept_disjoint", "concept_id"),
    ):
        for _, train, valid in protocol_splits(frame, protocol):
            assert set(frame.loc[train, column]).isdisjoint(
                set(frame.loc[valid, column])
            )
            assert set(frame.loc[valid, "target"]) == {0, 1}
    audit = _audit_protocols(frame)
    assert len(audit["exercise_disjoint"]) == 5
    assert len(audit["concept_disjoint"]) == 5


def test_external_gate_is_literal() -> None:
    metrics = {
        "roc_auc": 0.75,
        "macro_f1": 0.70,
        "log_loss": 0.60,
        "brier_score": 0.20,
        "ece_10": 0.08,
    }
    prior = {"log_loss": 0.68, "brier_score": 0.24}
    folds = [{"log_loss_gain": 0.01} for _ in range(5)]
    bootstrap = {"support": 0.96}
    assert all(external_gate(metrics, prior, folds, bootstrap).values())
    metrics["ece_10"] = 0.11
    assert external_gate(metrics, prior, folds, bootstrap)["ece_10"] is False
