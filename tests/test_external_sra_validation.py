import numpy as np
import pandas as pd

from trace_ace.external_sra_validation import _gate_decisions, _softmax


def test_softmax_is_stable_and_normalized() -> None:
    probability = _softmax(np.asarray([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert probability.argmax(axis=1).tolist() == [1]


def test_gate_requires_top5_magnitude() -> None:
    metrics = pd.DataFrame(
        {
            "delta_log_loss_vs_v02": [-0.0012] * 4,
            "delta_roc_auc_vs_v02": [0.001] * 4,
            "delta_brier_score_vs_v02": [-0.001] * 4,
            "delta_ece_10_vs_v02": [-0.001] * 4,
        }
    )
    bootstrap = pd.DataFrame(
        {
            "environment": ["ALL_MACRO", "ALL_MACRO"],
            "resampler": ["session", "objective_family"],
            "probability_gain_positive": [0.99, 0.99],
        }
    )
    result = _gate_decisions(metrics, bootstrap)
    assert result["passes_backup_gate"] is True
    assert result["passes_top5_gate"] is False
