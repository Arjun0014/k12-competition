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
DEFAULT_C_VALUES = (0.01, 0.03, 0.1)


def prepare_bge_base_features(
    cache_dir: Path, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    context = np.load(cache_dir / "bge_base_context_256.npy").astype(np.float32)
    objective = np.load(cache_dir / "bge_base_objective_256.npy").astype(np.float32)
    if context.shape != objective.shape or context.shape[0] != len(frame):
        raise ValueError("BGE-base cache shape does not match modeling rows.")
    if context.shape[1] != 768:
        raise ValueError(f"Expected 768 BGE-base dimensions; found {context.shape[1]}.")
    if not np.isfinite(context).all() or not np.isfinite(objective).all():
        raise ValueError("BGE-base cache contains non-finite values.")
    features = np.column_stack(
        [
            context * np.float32(0.7),
            objective * np.float32(0.7),
            context * objective * np.float32(6.0),
            np.abs(context - objective) * np.float32(0.7),
        ]
    )
    similarity = np.sum(context * objective, axis=1, keepdims=True)
    return features, similarity


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


def run_encoder_upgrade_validation(
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
    baseline = _load_baseline(paths.experiments_dir, baseline_run_id, protocols)
    base_features, base_similarity = prepare_bge_base_features(paths.cache_dir, frame)
    _, dense, dense_names = prepare_semantic_features(paths.cache_dir, frame)
    dense = dense.copy()
    dense[:, -1:] = base_similarity
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    c_list = list(c_values)
    for c_value in c_list:
        for protocol_name in protocol_names:
            protocol_base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
            if list(protocol_base["response_id"]) != list(frame["response_id"]):
                raise ValueError(f"Baseline row order mismatch for {protocol_name}.")
            folds = protocol_base["fold"].to_numpy(dtype=np.int8)
            print(f"BGE-base C={c_value:g} on {protocol_name}", flush=True)
            base_prediction = semantic_logistic_oof(
                frame,
                folds,
                base_features,
                dense,
                regularization_c=c_value,
                n_splits=int(folds.max()) + 1,
            )
            pred_full = protocol_base["pred_full"].to_numpy(dtype=np.float64)
            pred_role = protocol_base["pred_role"].to_numpy(dtype=np.float64)
            pred_small = protocol_base["pred_semantic"].to_numpy(dtype=np.float64)
            pred_v02 = protocol_base["pred_v02"].to_numpy(dtype=np.float64)
            baseline_metrics = binary_metrics(target, pred_v02)
            candidates = {
                "standalone": base_prediction,
                "replace_small_50": 0.25 * pred_full + 0.25 * pred_role + 0.50 * base_prediction,
                "hybrid_small25_base25": (
                    0.25 * pred_full
                    + 0.25 * pred_role
                    + 0.25 * pred_small
                    + 0.25 * base_prediction
                ),
                "base_heavy_60": 0.20 * pred_full + 0.20 * pred_role + 0.60 * base_prediction,
                "v02_blend_base10": 0.90 * pred_v02 + 0.10 * base_prediction,
                "v02_blend_base20": 0.80 * pred_v02 + 0.20 * base_prediction,
                "v02_blend_base30": 0.70 * pred_v02 + 0.30 * base_prediction,
            }
            for ensemble, candidate_prediction in candidates.items():
                metrics = binary_metrics(target, candidate_prediction)
                model_name = f"bge_base_c{c_value:g}__{ensemble}"
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": model_name,
                        "ensemble": ensemble,
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
                        "regularization_c": c_value,
                        "pred_v02": pred_v02,
                        "pred_bge_base": base_prediction,
                    }
                )
            )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_encoder_upgrade_validation")
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
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "baseline_run_id": baseline_run_id,
        "protocols": protocol_names,
        "model": "BAAI/bge-base-en-v1.5",
        "license": "MIT",
        "embedding_dimension": 768,
        "max_sequence_length": 256,
        "regularization_c": c_list,
        "dense_features": dense_names,
        "selection_policy": (
            "Select architecture and C only on the frozen pilot, then lock and "
            "confirm on untouched semantic-family protocols."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate BGE-base semantic upgrade.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--c", action="append", type=float)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_encoder_upgrade_validation(
        args.project_root,
        c_values=args.c or DEFAULT_C_VALUES,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
    )
    print(f"Encoder upgrade validation complete: {result['run_id']}")
    print(result["metrics"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
