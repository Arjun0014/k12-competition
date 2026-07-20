from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.multiview_validation import BASELINE_RUN_ID, BLEND_WEIGHTS, PILOT_PROTOCOL
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
from trace_ace.semantic_hard_validation import prepare_semantic_features


TREE_SPECS = {
    "ordered_hgb7": {
        "max_leaf_nodes": 7,
        "l2_regularization": 5.0,
        "min_samples_leaf": 100,
    },
    "ordered_hgb15": {
        "max_leaf_nodes": 15,
        "l2_regularization": 10.0,
        "min_samples_leaf": 100,
    },
}


def hist_gradient_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    features: np.ndarray,
    specification: dict[str, float | int],
) -> np.ndarray:
    target = frame["target"].to_numpy()
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train_mask, validation_mask = _fold_masks(frame, folds, int(fold))
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=250,
            max_leaf_nodes=int(specification["max_leaf_nodes"]),
            min_samples_leaf=int(specification["min_samples_leaf"]),
            l2_regularization=float(specification["l2_regularization"]),
            early_stopping=False,
            random_state=20260717 + int(fold),
        )
        model.fit(features[train_mask], target[train_mask])
        prediction[validation_mask] = model.predict_proba(features[validation_mask])[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete dense-tree OOF predictions.")
    return prediction


def run_dense_tree_validation(
    project_root: str | Path,
    model_names: Iterable[str] = TREE_SPECS,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy()
    semantic_variants, base_dense, base_names = prepare_semantic_features(
        paths.cache_dir, frame
    )
    del semantic_variants
    ordered = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    if list(ordered["response_id"]) != list(frame["response_id"]):
        raise ValueError("Ordered tree rows do not align with modeling data.")
    ordered_values = ordered[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    features = np.column_stack([base_dense, ordered_values])
    if not np.isfinite(features).all():
        raise ValueError("Dense tree features contain non-finite values.")

    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    )
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline protocols match the dense-tree request.")
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_models = list(model_names)
    for model_name in selected_models:
        for protocol_name in protocol_names:
            protocol_base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
            if list(protocol_base["response_id"]) != list(frame["response_id"]):
                raise ValueError(f"Baseline order mismatch for {protocol_name}.")
            folds = protocol_base["fold"].to_numpy(dtype=np.int8)
            prediction = hist_gradient_oof(
                frame, folds, features, TREE_SPECS[model_name]
            )
            baseline_prediction = protocol_base["pred_v02"].to_numpy(dtype=np.float64)
            baseline_metrics = binary_metrics(target, baseline_prediction)
            standalone = binary_metrics(target, prediction)
            metric_rows.append(
                {
                    "protocol": protocol_name,
                    "model": model_name,
                    "prediction_type": "standalone",
                    "new_model_weight": 1.0,
                    **standalone,
                    "delta_log_loss_vs_v02": standalone["log_loss"]
                    - baseline_metrics["log_loss"],
                    "delta_roc_auc_vs_v02": standalone["roc_auc"]
                    - baseline_metrics["roc_auc"],
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "protocol": protocol_name,
                        "response_id": frame["response_id"],
                        "target": target,
                        "fold": folds,
                        "model": model_name,
                        "pred_v02": baseline_prediction,
                        "pred_dense_tree": prediction,
                    }
                )
            )
            for weight in BLEND_WEIGHTS:
                blended = (1.0 - weight) * baseline_prediction + weight * prediction
                metrics = binary_metrics(target, blended)
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": f"{model_name}__blend_w{weight:.2f}",
                        "prediction_type": "blend",
                        "new_model_weight": weight,
                        **metrics,
                        "delta_log_loss_vs_v02": metrics["log_loss"]
                        - baseline_metrics["log_loss"],
                        "delta_roc_auc_vs_v02": metrics["roc_auc"]
                        - baseline_metrics["roc_auc"],
                    }
                )
            print(
                f"{protocol_name} {model_name}: loss={standalone['log_loss']:.6f}, "
                f"auc={standalone['roc_auc']:.6f}",
                flush=True,
            )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_dense_tree_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "baseline_run_id": baseline_run_id,
        "protocols": protocol_names,
        "models": selected_models,
        "tree_specs": {name: TREE_SPECS[name] for name in selected_models},
        "feature_names": [*base_names, *ORDERED_FEATURE_NAMES],
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate nonlinear ordered/dense tutoring features."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model", action="append", choices=tuple(TREE_SPECS))
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_dense_tree_validation(
        args.project_root,
        model_names=args.model or TREE_SPECS,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
    )
    print(f"Dense tree validation complete: {result['run_id']}")
    print(result["metrics"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
