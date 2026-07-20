from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probability, boundaries[1:-1]), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(y_true[mask].mean() - probability[mask].mean()))
    return error


def binary_metrics(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    clip: float = 1e-6,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), clip, 1.0 - clip)
    if y.shape != p.shape:
        raise ValueError(f"Target and probability shapes differ: {y.shape} vs {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("Predictions contain non-finite values.")
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10": expected_calibration_error(y, p, n_bins=10),
        "prediction_mean": float(p.mean()),
    }


def fold_metric_rows(
    frame: pd.DataFrame,
    prediction_column: str,
    clip: float = 1e-6,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled = binary_metrics(frame["target"], frame[prediction_column], clip=clip)
    rows.append({"fold": "pooled", "rows": len(frame), **pooled})
    for fold, fold_frame in frame.groupby("fold", sort=True):
        metrics = binary_metrics(fold_frame["target"], fold_frame[prediction_column], clip=clip)
        rows.append({"fold": int(fold), "rows": len(fold_frame), **metrics})
    return rows
