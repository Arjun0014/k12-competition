from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.bridge_remediation_transfer import (
    EMBEDDING_DIMENSION,
    TASK_LINE,
    _audit_protocols,
    canonical_candidate_text,
    external_gate,
    group_fold,
    paired_training_matrix,
    protocol_splits,
    session_root,
)


def _turn(role: str, text: str) -> dict[str, str]:
    return {"user": role, "text": text}


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        rows.append(
            {
                "pair_id": f"pair-{fold}",
                "session_root": f"session-{fold}",
                "lesson_topic": f"lesson-{fold}",
                "session_fold": fold,
                "lesson_fold": fold,
            }
        )
    return pd.DataFrame(rows)


def test_canonical_candidate_text_uses_only_history_and_candidate() -> None:
    history = [
        _turn("tutor", "Try the area formula."),
        _turn("student", "I added the sides."),
        _turn("tutor", "Which operation gives area?"),
        _turn("student", "Maybe multiply?"),
    ]
    candidate = [_turn("tutor", "Yes. Multiply length by width.")]
    text = canonical_candidate_text(history, candidate)
    assert text.splitlines()[-1] == TASK_LINE
    assert "Tutor: Yes. Multiply length by width." in text
    assert "strategy" not in text.casefold()
    assert "intention" not in text.casefold()


def test_canonical_candidate_text_preserves_released_empty_turn_marker() -> None:
    history = [
        _turn("tutor", "Try the area formula."),
        _turn("student", ""),
        _turn("tutor", "Which operation gives area?"),
        _turn("student", "Multiply."),
    ]
    candidate = [_turn("tutor", ""), _turn("tutor", "Correct.")]
    text = canonical_candidate_text(history, candidate)
    assert "Student:" in text.splitlines()
    assert text.splitlines().count("Tutor:") == 1
    assert "Tutor: Correct." in text


def test_session_root_and_group_folds_are_stable() -> None:
    assert session_root("4110653_19") == "4110653"
    assert group_fold("session_disjoint", "4110653") == group_fold(
        "session_disjoint", "4110653"
    )
    assert 0 <= group_fold("lesson_disjoint", "3.6D") < 5


def test_protocol_splits_hold_out_the_declared_group() -> None:
    frame = _frame()
    for protocol, column in (
        ("session_disjoint", "session_root"),
        ("lesson_disjoint", "lesson_topic"),
    ):
        for _, train, valid in protocol_splits(frame, protocol):
            assert set(frame.loc[train, column]).isdisjoint(
                set(frame.loc[valid, column])
            )
    audit = _audit_protocols(frame)
    assert len(audit["session_disjoint"]) == 5
    assert len(audit["lesson_disjoint"]) == 5


def test_paired_training_matrix_is_antisymmetric_and_balanced() -> None:
    differences = np.ones((3, EMBEDDING_DIMENSION), dtype=np.float32)
    features, target = paired_training_matrix(differences)
    assert features.shape == (6, EMBEDDING_DIMENSION)
    assert np.array_equal(features[:3], -features[3:])
    assert target.tolist() == [1, 1, 1, 0, 0, 0]


def test_external_gate_is_literal() -> None:
    metrics = {
        "accuracy": 0.65,
        "roc_auc": 0.70,
        "log_loss": 0.64,
        "brier_score": 0.23,
        "ece_10": 0.08,
    }
    folds = [{"log_loss_gain": 0.01} for _ in range(5)]
    bootstrap = {"support": 0.96}
    assert all(external_gate(metrics, folds, bootstrap).values())
    metrics["accuracy"] = 0.59
    assert external_gate(metrics, folds, bootstrap)["accuracy"] is False
