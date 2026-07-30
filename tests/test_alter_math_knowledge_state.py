from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.alter_math_knowledge_state import (
    COMPETITION_FEATURES,
    LABELS,
    _canonicalize_source,
    _classifier,
    _current_utterance,
    _student_turns,
    _vectorizer,
    aggregate_probabilities,
    source_fold,
)


def test_current_utterance_excludes_document() -> None:
    role, text = _current_utterance(
        "u17: I would subtract five. *document* u17: hidden future text"
    )
    assert role == "u17"
    assert text == "I would subtract five."
    assert "hidden" not in text


def test_canonicalization_keeps_students_and_purges_cross_session_duplicates(
    monkeypatch,
) -> None:
    rows = pd.DataFrame(
        {
            "id": ["a", "a", "b", "c", "d"],
            "id2": ["1", "2", "1", "1", "1"],
            "content": [
                "u1: unique reasoning *document* transcript",
                "e1: tutor only *document* transcript",
                "u2: repeated answer *document* transcript",
                "u3: repeated answer *document* transcript",
                "u4: another unique attempt *document* transcript",
            ],
            "computational skill": [1, 0, 0, 1, 0],
            "conceptual knowledge": [0, 0, 1, 0, 1],
            "strategic knowledge": [0, 0, 0, 0, 1],
        }
    )
    monkeypatch.setattr(
        "trace_ace.alter_math_knowledge_state.EXPECTED_ROWS",
        2,
    )
    monkeypatch.setattr(
        "trace_ace.alter_math_knowledge_state.EXPECTED_SESSIONS",
        2,
    )
    monkeypatch.setattr(
        "trace_ace.alter_math_knowledge_state.EXPECTED_POSITIVES",
        {
            "computational skill": 1,
            "conceptual knowledge": 1,
            "strategic knowledge": 1,
        },
    )
    expected_folds = {}
    retained = rows.iloc[[0, 4]]
    for fold in range(5):
        mask = retained["id"].map(source_fold).eq(fold)
        expected_folds[fold] = {
            "rows": int(mask.sum()),
            "sessions": int(retained.loc[mask, "id"].nunique()),
            "positives": tuple(
                int(retained.loc[mask, label].sum()) for label in LABELS
            ),
        }
    monkeypatch.setattr(
        "trace_ace.alter_math_knowledge_state.EXPECTED_FOLDS",
        expected_folds,
    )
    frame, audit = _canonicalize_source(rows, verify_frozen_hash=False)
    assert list(frame["text"]) == ["unique reasoning", "another unique attempt"]
    assert audit["removed_cross_session_duplicate_rows"] == 2
    assert audit["success_column_read"] is False


def test_vectorizer_and_classifier_produce_finite_probabilities() -> None:
    texts = [
        "I multiply both sides by five",
        "I think the slope means the rate",
        "first I isolate x and then substitute",
        "thanks",
    ] * 8
    labels = np.asarray([1, 0, 1, 0] * 8)
    matrix = _vectorizer().fit_transform(texts)
    probabilities = _classifier().fit(matrix, labels).predict_proba(matrix)[:, 1]
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] > 20
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_student_turn_parser_is_role_specific() -> None:
    context = "\n".join(
        [
            "[OBJECTIVE] Solve a linear equation.",
            "[STUDENT] I subtract two.",
            "[TUTOR] What happens next?",
            "[STUDENT] Then divide by three.",
            "[BACKGROUND] Ignore this.",
        ]
    )
    assert _student_turns(context) == [
        "I subtract two.",
        "Then divide by three.",
    ]


def test_probability_aggregation_has_frozen_order_and_zero_turn_fallback() -> None:
    probabilities = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [0.9, 0.8, 0.7],
        ]
    )
    values = aggregate_probabilities(probabilities, [0.02, 0.05, 0.01])
    assert tuple(values) == COMPETITION_FEATURES
    assert np.isclose(values["alter_computational_skill_early_mean"], 0.2)
    assert np.isclose(values["alter_computational_skill_late_mean"], 0.9)
    empty = aggregate_probabilities(
        np.empty((0, len(LABELS))),
        [0.02, 0.05, 0.01],
    )
    assert empty["alter_computational_skill_mean"] == 0.02
    assert empty["alter_computational_skill_late_minus_early"] == 0.0
