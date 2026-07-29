import numpy as np
import pandas as pd

from trace_ace.mathdial_contrastive_competition import (
    BLEND_WEIGHTS,
    PILOT_ROWS,
    _candidate_features,
    _metric_tables,
    _prediction_rows,
    load_target_free_inputs,
    verify_external_gate,
)


def test_e820_external_gate_and_target_free_lineage() -> None:
    report = verify_external_gate(".")
    frame, indices = load_target_free_inputs(".")
    assert report["passes_target_free_gate"] is True
    assert len(frame) == PILOT_ROWS
    assert len(indices) == PILOT_ROWS
    assert not frame["response_id"].duplicated().any()
    assert frame["query"].str.startswith(
        "Represent this K-12 math learning target"
    ).all()


def test_e820_candidate_feature_geometry() -> None:
    context = np.zeros((PILOT_ROWS, 768), dtype=np.float32)
    objective = np.zeros_like(context)
    context[:, 0] = 1.0
    objective[:, 1] = 1.0
    features, similarity = _candidate_features(context, objective)
    assert features.shape == (PILOT_ROWS, 4 * 768)
    assert similarity.shape == (PILOT_ROWS, 1)
    assert np.all(similarity == 0.0)
    assert np.all(features[:, 0] == np.float32(0.7))
    assert np.all(features[:, 768 + 1] == np.float32(0.7))


def test_e820_prediction_and_equal_fold_metrics() -> None:
    rows = 20
    component = pd.DataFrame(
        {
            "environment": ["V_seen"] * rows,
            "response_id": [f"r{i}" for i in range(rows)],
            "session_id": [f"s{i}" for i in range(rows)],
            "learning_objective_id": [f"o{i % 4}" for i in range(rows)],
            "semantic_family": [f"f{i % 3}" for i in range(rows)],
            "fold": np.repeat(np.arange(5), 4),
            "evaluation_eligible": [True] * rows,
            "target": np.tile([0, 1], rows // 2),
            "pred_full": [0.5] * rows,
            "pred_role": [0.5] * rows,
            "pred_bge_base": [0.5] * rows,
        }
    )
    candidate = np.where(component["target"].to_numpy() == 1, 0.8, 0.2)
    predictions = _prediction_rows(
        component,
        candidate,
        np.full(rows, 0.5),
        BLEND_WEIGHTS,
    )
    folds, environments = _metric_tables(predictions)
    assert len(predictions) == rows * len(BLEND_WEIGHTS)
    assert len(folds) == 5 * len(BLEND_WEIGHTS)
    assert len(environments) == len(BLEND_WEIGHTS)
    assert (
        environments["delta_log_loss_vs_v05_raw"].to_numpy() < 0
    ).all()
