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
from trace_ace.retrieval_hard_validation import RETRIEVAL_DENSE_COLUMNS
from trace_ace.robust_validation import build_regime_frame, regime_scorecard


def audit_candidate(
    project_root: str | Path,
    run_ids: Iterable[str],
    model_name: str,
    prediction_column: str,
    candidate_weight: float,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    run_id_list = list(run_ids)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy()
    retrieval = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", *RETRIEVAL_DENSE_COLUMNS],
    )
    regimes = build_regime_frame(frame, retrieval)

    inputs: list[pd.DataFrame] = []
    for run_id in run_id_list:
        predictions = pd.read_parquet(
            paths.experiments_dir / "runs" / run_id / "oof_predictions.parquet"
        )
        selected = predictions.loc[predictions["model"].eq(model_name)].copy()
        if not selected.empty:
            selected.insert(0, "source_run_id", run_id)
            inputs.append(selected)
    if not inputs:
        raise ValueError("The requested model was not found in the supplied run IDs.")
    selected = pd.concat(inputs, ignore_index=True)
    if selected.duplicated(["protocol", "response_id"]).any():
        raise ValueError("Candidate audit inputs contain duplicate protocol-response rows.")

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    regime_rows: list[pd.DataFrame] = []
    output_rows: list[pd.DataFrame] = []
    for protocol, protocol_frame in selected.groupby("protocol", sort=False):
        if list(protocol_frame["response_id"]) != list(frame["response_id"]):
            raise ValueError(f"Candidate row order mismatch for {protocol}.")
        folds = protocol_frame["fold"].to_numpy(dtype=np.int8)
        baseline = protocol_frame["pred_v02"].to_numpy(dtype=np.float64)
        candidate_component = protocol_frame[prediction_column].to_numpy(dtype=np.float64)
        candidate = (1.0 - candidate_weight) * baseline + candidate_weight * candidate_component
        baseline_metrics = binary_metrics(target, baseline)
        candidate_metrics = binary_metrics(target, candidate)
        metric_rows.append(
            {
                "protocol": protocol,
                **candidate_metrics,
                "baseline_log_loss": baseline_metrics["log_loss"],
                "baseline_roc_auc": baseline_metrics["roc_auc"],
                "delta_log_loss_vs_v02": candidate_metrics["log_loss"]
                - baseline_metrics["log_loss"],
                "delta_roc_auc_vs_v02": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
            }
        )
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            candidate_fold = binary_metrics(target[mask], candidate[mask])
            baseline_fold = binary_metrics(target[mask], baseline[mask])
            fold_rows.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    **candidate_fold,
                    "delta_log_loss_vs_v02": candidate_fold["log_loss"]
                    - baseline_fold["log_loss"],
                    "delta_roc_auc_vs_v02": candidate_fold["roc_auc"]
                    - baseline_fold["roc_auc"],
                }
            )
        candidate_scorecard = regime_scorecard(
            str(protocol), target, candidate, folds, regimes
        )
        baseline_scorecard = regime_scorecard(
            str(protocol), target, baseline, folds, regimes
        )
        merged = candidate_scorecard.merge(
            baseline_scorecard,
            on=["protocol", "regime", "value"],
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        merged["delta_log_loss_vs_v02"] = (
            merged["log_loss"] - merged["log_loss_baseline"]
        )
        merged["delta_roc_auc_vs_v02"] = (
            merged["roc_auc"] - merged["roc_auc_baseline"]
        )
        regime_rows.append(merged)
        output_rows.append(
            pd.DataFrame(
                {
                    "protocol": protocol,
                    "response_id": frame["response_id"],
                    "target": target,
                    "fold": folds,
                    "pred_v02": baseline,
                    "pred_candidate_component": candidate_component,
                    "pred_candidate_blend": candidate,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values("protocol")
    folds = pd.DataFrame(fold_rows).sort_values(["protocol", "fold"])
    scorecard = pd.concat(regime_rows, ignore_index=True)
    legal_regimes = scorecard.loc[
        ~scorecard["regime"].isin(["overall", "fold", "session_objectives"])
        & scorecard["rows"].ge(100)
    ]
    loss_improvements = -metrics["delta_log_loss_vs_v02"]
    summary = {
        "protocol_count": len(metrics),
        "protocol_loss_wins": int(metrics["delta_log_loss_vs_v02"].lt(0).sum()),
        "protocol_auc_wins": int(metrics["delta_roc_auc_vs_v02"].gt(0).sum()),
        "median_log_loss_improvement": float(loss_improvements.median()),
        "mean_log_loss_improvement": float(loss_improvements.mean()),
        "mean_roc_auc_improvement": float(metrics["delta_roc_auc_vs_v02"].mean()),
        "fold_loss_win_fraction": float(folds["delta_log_loss_vs_v02"].lt(0).mean()),
        "fold_auc_win_fraction": float(folds["delta_roc_auc_vs_v02"].gt(0).mean()),
        "worst_fold_log_loss_regression": float(
            folds["delta_log_loss_vs_v02"].max()
        ),
        "worst_legal_regime_log_loss_regression": float(
            legal_regimes["delta_log_loss_vs_v02"].max()
        ),
        "passes_loss_size_gate": bool(loss_improvements.median() >= 0.002),
        "passes_protocol_consistency_gate": bool(
            metrics["delta_log_loss_vs_v02"].lt(0).all()
        ),
        "passes_fold_consistency_gate": bool(
            folds["delta_log_loss_vs_v02"].lt(0).mean() >= 0.70
        ),
        "passes_regime_regression_gate": bool(
            legal_regimes["delta_log_loss_vs_v02"].max() <= 0.001
        ),
    }
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_candidate_audit")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "protocol_metrics.csv", index=False, lineterminator="\n")
    folds.to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    scorecard.to_csv(run_dir / "regime_scorecard.csv", index=False, lineterminator="\n")
    pd.concat(output_rows, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "source_run_ids": run_id_list,
        "model": model_name,
        "prediction_column": prediction_column,
        "candidate_weight": candidate_weight,
        "summary": summary,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": summary}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit a locked candidate across protocols, folds, and legal regimes."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--weight", type=float, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = audit_candidate(
        args.project_root,
        run_ids=args.run_id,
        model_name=args.model,
        prediction_column=args.prediction_column,
        candidate_weight=args.weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
