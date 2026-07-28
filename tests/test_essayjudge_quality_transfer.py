from __future__ import annotations

import pandas as pd

from trace_ace.essayjudge_quality_transfer import (
    TASK_LINE,
    _audit_folds,
    canonical_text,
    external_gate,
    fold_for_component,
    prompt_component_map,
)


def test_canonical_text_contains_only_prompt_response_and_task() -> None:
    text = canonical_text("  explain   this ", " a  clear response ")
    assert text == (
        "Prompt: explain this\n"
        "Student response: a clear response\n"
        f"{TASK_LINE}"
    )
    assert "ground_truth" not in text


def test_duplicate_essay_links_prompt_components() -> None:
    frame = pd.DataFrame(
        {
            "Question": ["q1", "q2", "q3", "q3"],
            "Essay": ["same", "same", "other", "unique"],
        }
    )
    mapping = prompt_component_map(frame, expected_components=2)
    assert mapping["q1"] == mapping["q2"]
    assert mapping["q1"] != mapping["q3"]


def test_fold_hash_is_deterministic() -> None:
    assert fold_for_component("prompt") == fold_for_component("prompt")
    assert 0 <= fold_for_component("prompt") < 5


def test_fold_audit_accepts_group_locked_rows() -> None:
    rows = []
    for fold in range(5):
        component = f"component-{fold}"
        rows.append(
            {
                "fold": fold,
                "component_id": component,
                "essay_sha256": f"essay-{fold}",
            }
        )
    audits = _audit_folds(pd.DataFrame(rows))
    assert [row["validation_rows"] for row in audits] == [1, 1, 1, 1, 1]


def test_external_gate_is_literal() -> None:
    passing_metrics = {
        "pearson": 0.36,
        "spearman": 0.36,
        "rmse": 0.099,
    }
    passed = external_gate(
        passing_metrics, 0.006, 0.006, [0.001] * 5, 0.96, 0.96
    )
    assert all(passed.values())
    failing = external_gate(
        passing_metrics, 0.006, 0.006, [0.001] * 4 + [0.0], 0.96, 0.96
    )
    assert failing["positive_fold_gains"] is False
