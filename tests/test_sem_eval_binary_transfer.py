import numpy as np
import pandas as pd

from trace_ace.sem_eval_binary_transfer import (
    _gate_for_split,
    binary_mastery_labels,
)


def test_binary_mastery_labels_only_fully_correct_positive() -> None:
    frame = pd.DataFrame(
        {
            "sra_label": [
                "correct",
                "partially_correct_incomplete",
                "contradictory",
                "irrelevant",
                "non_domain",
            ]
        }
    )
    np.testing.assert_array_equal(binary_mastery_labels(frame), [1, 0, 0, 0, 0])


def test_external_gate_requires_every_strict_clause() -> None:
    metrics = {
        "roc_auc": 0.73,
        "macro_f1": 0.66,
        "ece_10": 0.05,
        "log_loss": 0.60,
        "brier_score": 0.20,
    }
    prior = {"log_loss": 0.63, "brier_score": 0.22}
    bootstrap = {"support_positive_log_loss_gain": 0.96}
    clauses = _gate_for_split(
        "test-unseen-questions", metrics, prior, bootstrap
    )
    assert all(clauses.values())
    metrics["macro_f1"] = 0.64
    clauses = _gate_for_split(
        "test-unseen-questions", metrics, prior, bootstrap
    )
    assert not clauses["macro_f1_at_least_0_65"]
