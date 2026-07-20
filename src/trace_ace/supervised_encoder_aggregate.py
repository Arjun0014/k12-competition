from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


LOCKED_WEIGHT = 0.30


def aggregate_supervised_screens(
    project_root: str | Path,
    run_ids: Iterable[str],
    locked_weight: float = LOCKED_WEIGHT,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    supplied = list(run_ids)
    if len(supplied) != 5:
        raise ValueError("Exactly five supervised screen runs are required.")
    prediction_frames: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    for run_id in supplied:
        run_dir = paths.experiments_dir / "runs" / run_id
        reports.append(json.loads((run_dir / "report.json").read_text(encoding="utf-8")))
        prediction_frames.append(
            pd.read_parquet(run_dir / "validation_predictions.parquet")
        )
    folds = [int(report["validation_fold"]) for report in reports]
    if sorted(folds) != list(range(5)) or len(set(folds)) != 5:
        raise ValueError(f"Expected validation folds 0-4 exactly once; found {folds}.")
    comparable_settings = {
        (
            report["training_rows"],
            report["max_length"],
            report["epochs"],
            report["unfrozen_layers"],
            report["seed"],
        )
        for report in reports
    }
    if len(comparable_settings) != 1:
        raise ValueError("Supervised screen settings differ across folds.")

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "response_id"]
    )
    if predictions["response_id"].duplicated().any():
        raise ValueError("Aggregated supervised predictions contain duplicate responses.")
    predictions["pred_locked"] = (
        (1.0 - locked_weight) * predictions["pred_v02"]
        + locked_weight * predictions["pred_supervised"]
    )
    target = predictions["target"].to_numpy()
    baseline = predictions["pred_v02"].to_numpy()
    candidate = predictions["pred_locked"].to_numpy()
    baseline_metrics = binary_metrics(target, baseline)
    candidate_metrics = binary_metrics(target, candidate)
    metrics = pd.DataFrame(
        [
            {"model": "v02_baseline", "model_weight": 0.0, **baseline_metrics},
            {
                "model": "supervised_bge_small_locked",
                "model_weight": locked_weight,
                **candidate_metrics,
            },
        ]
    )
    metrics["delta_log_loss_vs_v02"] = metrics["log_loss"] - baseline_metrics["log_loss"]
    metrics["delta_roc_auc_vs_v02"] = metrics["roc_auc"] - baseline_metrics["roc_auc"]

    fold_rows: list[dict[str, object]] = []
    for fold, frame in predictions.groupby("fold", sort=True):
        fold_target = frame["target"].to_numpy()
        fold_baseline = binary_metrics(fold_target, frame["pred_v02"].to_numpy())
        fold_candidate = binary_metrics(fold_target, frame["pred_locked"].to_numpy())
        fold_rows.append(
            {
                "fold": int(fold),
                **fold_candidate,
                "delta_log_loss_vs_v02": fold_candidate["log_loss"]
                - fold_baseline["log_loss"],
                "delta_roc_auc_vs_v02": fold_candidate["roc_auc"]
                - fold_baseline["roc_auc"],
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "delta_log_loss_vs_v02": float(
            candidate_metrics["log_loss"] - baseline_metrics["log_loss"]
        ),
        "delta_roc_auc_vs_v02": float(
            candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"]
        ),
        "fold_loss_wins": int(fold_metrics["delta_log_loss_vs_v02"].lt(0).sum()),
        "fold_auc_wins": int(fold_metrics["delta_roc_auc_vs_v02"].gt(0).sum()),
        "fold_count": len(fold_metrics),
        "promoted": False,
    }

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_supervised_encoder_confirmation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "source_run_ids": supplied,
        "locked_weight": locked_weight,
        "summary": summary,
        "decision": "reject",
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": summary}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate locked supervised screen folds.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--weight", type=float, default=LOCKED_WEIGHT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = aggregate_supervised_screens(
        args.project_root, args.run_id, locked_weight=args.weight
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
