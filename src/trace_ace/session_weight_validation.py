from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from trace_ace.hard_validation import (
    _fold_masks,
    _response_transcript_matrix,
    prepare_hash_matrices,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
WEIGHT_POWERS = (0.5, 1.0)


def session_weighted_full_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    matrix,
    weight_power: float,
    n_splits: int,
) -> np.ndarray:
    if weight_power <= 0:
        raise ValueError("Session weight power must be positive.")
    target = frame["target"].to_numpy(dtype=np.int8)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        train_sessions = frame.loc[train_mask, "session_id"]
        counts = train_sessions.value_counts()
        sample_weight = train_sessions.map(counts).to_numpy(dtype=np.float64)
        sample_weight = np.power(sample_weight, -weight_power)
        sample_weight /= sample_weight.mean()
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=3e-5,
            max_iter=100,
            tol=1e-4,
            shuffle=True,
            random_state=20260716 + fold,
            average=True,
        )
        model.fit(matrix[train_mask], target[train_mask], sample_weight=sample_weight)
        prediction[validation_mask] = model.predict_proba(matrix[validation_mask])[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete session-weighted predictions.")
    return prediction


def _load_baseline(
    experiments_dir: Path,
    baseline_run_id: str,
    protocols: set[str] | None,
) -> pd.DataFrame:
    baseline = pd.read_parquet(
        experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    )
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline predictions match the requested protocols.")
    return baseline.reset_index(drop=True)


def run_session_weight_validation(
    project_root: str | Path,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
    weight_powers: Iterable[float] = WEIGHT_POWERS,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    session_matrix, _ = prepare_hash_matrices(frame, paths.cache_dir)
    response_matrix = _response_transcript_matrix(
        frame, paths.cache_dir, session_matrix
    )
    baseline = _load_baseline(paths.experiments_dir, baseline_run_id, protocols)
    target = frame["target"].to_numpy(dtype=np.float64)
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    selected_powers = list(weight_powers)
    for power in selected_powers:
        for protocol_name in protocol_names:
            protocol_base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
            if list(protocol_base["response_id"]) != list(frame["response_id"]):
                raise ValueError(f"Baseline row order mismatch for {protocol_name}.")
            folds = protocol_base["fold"].to_numpy(dtype=np.int8)
            print(
                f"Session-weighted full model power={power:g} on {protocol_name}",
                flush=True,
            )
            weighted_prediction = session_weighted_full_oof(
                frame,
                folds,
                response_matrix,
                weight_power=power,
                n_splits=int(folds.max()) + 1,
            )
            pred_v02 = protocol_base["pred_v02"].to_numpy(dtype=np.float64)
            pred_full = protocol_base["pred_full"].to_numpy(dtype=np.float64)
            pred_role = protocol_base["pred_role"].to_numpy(dtype=np.float64)
            pred_semantic = protocol_base["pred_semantic"].to_numpy(dtype=np.float64)
            baseline_metrics = binary_metrics(target, pred_v02)
            candidates = {
                "standalone": weighted_prediction,
                "replace_full_25": (
                    0.25 * weighted_prediction
                    + 0.25 * pred_role
                    + 0.50 * pred_semantic
                ),
                "hybrid_full_12p5": (
                    0.125 * weighted_prediction
                    + 0.125 * pred_full
                    + 0.25 * pred_role
                    + 0.50 * pred_semantic
                ),
                "v02_blend_weighted10": 0.90 * pred_v02 + 0.10 * weighted_prediction,
                "v02_blend_weighted20": 0.80 * pred_v02 + 0.20 * weighted_prediction,
            }
            for ensemble, candidate_prediction in candidates.items():
                metrics = binary_metrics(target, candidate_prediction)
                model_name = f"session_power{power:g}__{ensemble}"
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": model_name,
                        "ensemble": ensemble,
                        "weight_power": power,
                        **metrics,
                        "delta_log_loss_vs_v02": (
                            metrics["log_loss"] - baseline_metrics["log_loss"]
                        ),
                        "delta_roc_auc_vs_v02": (
                            metrics["roc_auc"] - baseline_metrics["roc_auc"]
                        ),
                    }
                )
                for fold in range(int(folds.max()) + 1):
                    mask = folds == fold
                    fold_metrics = binary_metrics(target[mask], candidate_prediction[mask])
                    fold_baseline = binary_metrics(target[mask], pred_v02[mask])
                    fold_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold,
                            "model": model_name,
                            "ensemble": ensemble,
                            "weight_power": power,
                            **fold_metrics,
                            "delta_log_loss_vs_v02": (
                                fold_metrics["log_loss"] - fold_baseline["log_loss"]
                            ),
                            "delta_roc_auc_vs_v02": (
                                fold_metrics["roc_auc"] - fold_baseline["roc_auc"]
                            ),
                        }
                    )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "protocol": protocol_name,
                        "response_id": frame["response_id"],
                        "fold": folds,
                        "target": target,
                        "weight_power": power,
                        "pred_v02": pred_v02,
                        "pred_session_weighted": weighted_prediction,
                    }
                )
            )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_session_weight_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at_utc": timestamp.isoformat(),
                "baseline_run_id": baseline_run_id,
                "protocols": protocol_names,
                "weight_powers": selected_powers,
                "selection_policy": (
                    "Screen on the frozen pilot and confirm only one locked session "
                    "weighting policy on untouched semantic protocols."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate objective-shift-robust session weighting."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--weight-power", action="append", type=float)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_session_weight_validation(
        args.project_root,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
        weight_powers=args.weight_power or WEIGHT_POWERS,
    )
    print(f"Session-weight validation complete: {result['run_id']}")
    print(result["metrics"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
