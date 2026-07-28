from __future__ import annotations

import pandas as pd

from trace_ace.fairytale_answer_support_transfer import (
    TASK_LINE,
    canonical_text,
    external_gate,
    negative_source_indices,
)


def test_canonical_text_has_only_frozen_fields() -> None:
    text = canonical_text(" story  context ", " why? ", " because ")
    assert text == (
        "Context: story context\n"
        "Question: why?\n"
        "Student response: because\n"
        f"{TASK_LINE}"
    )
    assert "explicit" not in text
    assert "local" not in text


def test_negative_answer_stays_within_story_and_differs() -> None:
    frame = pd.DataFrame(
        {
            "story_name": ["a", "a", "a", "b", "b"],
            "answer1": ["one", "two", "three", "left", "right"],
        }
    )
    mapping = negative_source_indices(frame, "train")
    for source, negative in mapping.items():
        assert frame.at[source, "story_name"] == frame.at[negative, "story_name"]
        assert frame.at[source, "answer1"] != frame.at[negative, "answer1"]


def test_negative_selection_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "story_name": ["a", "a", "a"],
            "answer1": ["one", "two", "three"],
        }
    )
    assert negative_source_indices(frame, "valid") == negative_source_indices(
        frame, "valid"
    )


def test_external_gate_is_literal() -> None:
    metrics = {
        "roc_auc": 0.71,
        "macro_f1": 0.66,
        "log_loss": 0.65,
        "brier_score": 0.23,
        "ece_10": 0.09,
    }
    prior = {"log_loss": 0.69, "brier_score": 0.25}
    bootstrap = {"support": 0.96}
    assert all(external_gate(metrics, prior, bootstrap).values())
    metrics["ece_10"] = 0.11
    assert external_gate(metrics, prior, bootstrap)["ece_10"] is False
