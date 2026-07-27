import pandas as pd

from trace_ace.dialogue_kt_transfer import (
    _correct_final_turn,
    canonical_turn_rows,
    external_gate,
)


def test_canonical_turn_input_excludes_current_student_response() -> None:
    frame = pd.DataFrame(
        [
            {
                "index": 1,
                "qid": "q1",
                "self-correctness": "Yes",
                "dialogue": repr(
                    [
                        {
                            "turn": 1,
                            "teacher": "What is two plus two?",
                            "student": "SECRET_CURRENT_ANSWER",
                        }
                    ]
                ),
                "annotation": repr(
                    {
                        "turn 1": {
                            "correct": True,
                            "kcs": ["Add whole numbers"],
                        }
                    }
                ),
                "meta_data": repr({}),
            }
        ]
    )
    rows, invalid = canonical_turn_rows(
        frame, split="mathdial_train", source="mathdial"
    )
    assert invalid == 0
    assert len(rows) == 1
    assert "What is two plus two?" in rows[0]["history"]
    assert "SECRET_CURRENT_ANSWER" not in rows[0]["history"]
    assert rows[0]["objective"] == "Add whole numbers"


def test_final_turn_correction_excludes_revealed_answer() -> None:
    annotation = {"turn 1": {"correct": True, "kcs": ["Addition"]}}
    dialogue = [{"turn": 1, "teacher": "Question", "student": "Answer"}]
    row = pd.Series(
        {
            "self-correctness": "Yes, but I had to reveal the answer",
            "meta_data": {},
        }
    )
    corrected = _correct_final_turn(
        annotation, dialogue, row, source="mathdial"
    )
    assert corrected["turn 1"]["correct"] is None


def test_external_gate_requires_every_clause() -> None:
    math_metrics = {
        "roc_auc": 0.71,
        "macro_f1": 0.61,
        "log_loss": 0.65,
        "ece_10": 0.09,
        "brier_score": 0.235,
    }
    prior = {"log_loss": 0.68, "brier_score": 0.245}
    bootstrap = {"support_positive_log_loss_gain": 0.96}
    clauses = external_gate("mathdial_test", math_metrics, prior, bootstrap)
    assert all(clauses.values())
    math_metrics["ece_10"] = 0.11
    clauses = external_gate("mathdial_test", math_metrics, prior, bootstrap)
    assert not clauses["ece_10_at_most_0_10"]


def test_comta_gate_is_evaluation_only_and_strict() -> None:
    metrics = {
        "roc_auc": 0.61,
        "macro_f1": 0.56,
        "log_loss": 0.67,
        "ece_10": 0.14,
        "brier_score": 0.23,
    }
    prior = {"log_loss": 0.69, "brier_score": 0.24}
    bootstrap = {"support_positive_log_loss_gain": 0.91}
    assert all(
        external_gate("comta_eval_only", metrics, prior, bootstrap).values()
    )
    bootstrap["support_positive_log_loss_gain"] = 0.89
    clauses = external_gate("comta_eval_only", metrics, prior, bootstrap)
    assert not clauses["dialogue_bootstrap_support_at_least_0_90"]
