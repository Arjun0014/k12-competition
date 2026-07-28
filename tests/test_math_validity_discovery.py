from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.math_validity_discovery import (
    ASSIGNMENT_COLUMNS,
    CONTEXT_COLUMNS,
    MODELING_IDENTITY_COLUMNS,
    evaluate_relations,
    extract_context_relations,
    extract_relations,
    feedback_label,
    synthetic_gate,
)


def test_synthetic_gate_passes_every_clause() -> None:
    result = synthetic_gate()
    assert result["passes"]
    assert all(result["clauses"].values())


def test_exact_arithmetic_and_fraction_relations() -> None:
    examples = {
        "50 plus 6 equals 56": True,
        "30 plus 46 equals 70": False,
        "0.4 + 0.6 = 1": True,
        "1/2 + 1/4 = 3/4": True,
        "4/12 is greater than 3/12": True,
        "3/12 > 4/12": False,
        "9 divided by 3 is 3": True,
        "7 take away 2 equals 5": True,
    }
    for text, expected in examples.items():
        relations = extract_relations(text)
        assert len(relations) == 1
        assert relations[0]["valid"] is expected


def test_unsafe_and_ambiguous_relations_are_rejected() -> None:
    examples = (
        "9 divided by 0 is 4",
        "The date is 2026-07-28.",
        "50% plus 25% equals 75%.",
        "1 1/2 plus 1 equals 2 1/2.",
        "1000001 plus 1 equals 1000002.",
        "2 plus 3 plus 4 equals 9.",
        "I have 2 pencils and 3 books.",
    )
    assert all(not extract_relations(value) for value in examples)


def test_multiple_independent_relations_are_retained() -> None:
    relations = extract_relations("2 + 2 = 4, and 3 times 3 is 9.")
    assert len(relations) == 2
    assert all(record["valid"] for record in relations)


def test_feedback_rules_are_unambiguous() -> None:
    assert feedback_label("Exactly, that is correct.") == "positive"
    assert feedback_label("Not quite. Try again.") == "negative"
    assert feedback_label("Correct, but that is not quite the answer.") == "unknown"
    assert feedback_label("Tell me how you worked it out.") == "unknown"


def test_immediate_tutor_turn_is_required() -> None:
    context = "\n".join(
        [
            "[STUDENT] 2 + 3 = 5.",
            "[TUTOR] Correct.",
            "[STUDENT] 4 + 4 = 9.",
            "[BACKGROUND] Noise.",
            "[TUTOR] Not quite.",
        ]
    )
    records = extract_context_relations("response", context, 3)
    assert len(records) == 2
    assert records[0]["feedback"] == "positive"
    assert records[1]["feedback"] == "unknown"
    assert records[0]["style_cell"] == 3


def test_target_column_is_never_requested() -> None:
    assert "target" not in MODELING_IDENTITY_COLUMNS
    assert "target" not in CONTEXT_COLUMNS
    assert "target" not in ASSIGNMENT_COLUMNS


def test_evaluation_metrics_and_style_counts() -> None:
    rows = []
    for index in range(40):
        valid = index % 2 == 0
        rows.append(
            {
                "response_id": f"r{index}",
                "style_cell": index % 4,
                "valid": valid,
                "feedback": "positive" if valid else "negative",
            }
        )
    frame = pd.DataFrame(rows)
    result = evaluate_relations(frame, total_responses=100)
    assert np.isclose(result["balanced_accuracy"], 1.0)
    assert np.isclose(result["matthews_correlation"], 1.0)
    assert np.isclose(result["positive_feedback_rate_gap"], 1.0)
    assert result["style_cells_with_at_least_10_links"] == 4
    assert not result["passes_discovery_gate"]

