import numpy as np
import pandas as pd

from trace_ace.agreement_gate_validation import (
    component_agreement_prediction,
    pilot_continuation,
)


def test_component_agreement_gate_uses_combined_only_on_matching_directions():
    v02 = np.array([0.50, 0.50, 0.50, 0.50])
    small = np.array([0.40, 0.40, 0.60, 0.60])
    base = np.array([0.60, 0.60, 0.40, 0.40])
    nb = np.array([0.70, 0.30, 0.30, 0.70])
    combined = np.array([0.65, 0.45, 0.35, 0.55])

    gated, agrees = component_agreement_prediction(v02, small, base, nb, combined)

    assert agrees.tolist() == [True, False, True, False]
    np.testing.assert_allclose(gated, [0.65, 0.50, 0.35, 0.50])


def test_pilot_continuation_path_a_requires_gain_and_auc_retention():
    metrics = pd.DataFrame(
        [
            {
                "baseline_log_loss": 0.5900,
                "baseline_roc_auc": 0.6200,
                "combined_log_loss": 0.5880,
                "combined_roc_auc": 0.6300,
                "delta_log_loss_vs_v02": -0.0024,
                "delta_roc_auc_vs_v02": 0.0085,
                "delta_log_loss_vs_combined": -0.0004,
            }
        ]
    )
    folds = pd.DataFrame(
        {
            "combined_delta_log_loss_vs_v02": [-0.0030, 0.0015],
            "delta_log_loss_vs_v02": [-0.0020, 0.0010],
        }
    )

    decision = pilot_continuation(metrics, folds)

    assert decision["passes_path_a"] is True
    assert decision["continue_confirmation"] is True
