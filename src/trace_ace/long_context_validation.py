from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
DEFAULT_C_VALUES = (0.03, 0.1)


def run_long_context_validation(
    project_root: str | Path,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    target = frame["target"].to_numpy(dtype=np.float64)
    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    )
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline predictions match requested protocols.")
    context = np.load(paths.cache_dir / "bge_small_context_384.npy").astype(np.float32)
    objective = np.load(paths.cache_dir / "bge_small_objective_256.npy").astype(np.float32)
    if context.shape != objective.shape or context.shape != (len(frame), 384):
        raise ValueError("Long-context cache shape mismatch.")
    features = np.column_stack(
        [
            context * np.float32(0.7),
            objective * np.float32(0.7),
            context * objective * np.float32(6.0),
            np.abs(context - objective) * np.float32(0.7),
        ]
    )
    _, dense, dense_names = prepare_semantic_features(paths.cache_dir, frame)
    dense = dense.copy()
    dense[:, -1:] = np.sum(context * objective, axis=1, keepdims=True)

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))
    c_list = list(c_values)
    for c_value in c_list:
        for protocol_name in protocol_names:
            base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
            if list(base["response_id"]) != list(frame["response_id"]):
                raise ValueError(f"Baseline row order mismatch for {protocol_name}.")
            folds = base["fold"].to_numpy(dtype=np.int8)
            print(f"BGE-small long context C={c_value:g} on {protocol_name}", flush=True)
            long_prediction = semantic_logistic_oof(
                frame,
                folds,
                features,
                dense,
                regularization_c=c_value,
                n_splits=int(folds.max()) + 1,
            )
            pred_full = base["pred_full"].to_numpy(dtype=np.float64)
            pred_role = base["pred_role"].to_numpy(dtype=np.float64)
            pred_small = base["pred_semantic"].to_numpy(dtype=np.float64)
            pred_v02 = base["pred_v02"].to_numpy(dtype=np.float64)
            baseline_metrics = binary_metrics(target, pred_v02)
            candidates = {
                "standalone": long_prediction,
                "replace_small_50": 0.25 * pred_full + 0.25 * pred_role + 0.50 * long_prediction,
                "hybrid_small25_long25": (
                    0.25 * pred_full
                    + 0.25 * pred_role
                    + 0.25 * pred_small
                    + 0.25 * long_prediction
                ),
            }
            for ensemble, candidate in candidates.items():
                metrics = binary_metrics(target, candidate)
                model_name = f"bge_small_384_c{c_value:g}__{ensemble}"
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": model_name,
                        "ensemble": ensemble,
                        **metrics,
                        "delta_log_loss_vs_v02": metrics["log_loss"] - baseline_metrics["log_loss"],
                        "delta_roc_auc_vs_v02": metrics["roc_auc"] - baseline_metrics["roc_auc"],
                    }
                )
                for fold in range(int(folds.max()) + 1):
                    mask = folds == fold
                    fm = binary_metrics(target[mask], candidate[mask])
                    fb = binary_metrics(target[mask], pred_v02[mask])
                    fold_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold,
                            "model": model_name,
                            "ensemble": ensemble,
                            **fm,
                            "delta_log_loss_vs_v02": fm["log_loss"] - fb["log_loss"],
                            "delta_roc_auc_vs_v02": fm["roc_auc"] - fb["roc_auc"],
                        }
                    )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "protocol": protocol_name,
                        "response_id": frame["response_id"],
                        "fold": folds,
                        "target": target,
                        "regularization_c": c_value,
                        "pred_v02": pred_v02,
                        "pred_long_context": long_prediction,
                    }
                )
            )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_long_context_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
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
                "model": "BAAI/bge-small-en-v1.5",
                "license": "MIT",
                "max_sequence_length": 384,
                "regularization_c": c_list,
                "dense_features": dense_names,
                "selection_policy": "Pilot first; lock one candidate before confirmation.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate 384-token BGE-small context.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--c", action="append", type=float)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_long_context_validation(
        args.project_root,
        c_values=args.c or DEFAULT_C_VALUES,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
    )
    print(f"Long-context validation complete: {result['run_id']}")
    print(result["metrics"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
