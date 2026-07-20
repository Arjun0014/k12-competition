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


LOCKED_BGE_C = 0.1
LOCKED_NB_MODEL = "nbsvm_feedback_word_ordered_c1"


def component_agreement_prediction(
    pred_v02: np.ndarray,
    pred_semantic: np.ndarray,
    pred_bge_base: np.ndarray,
    pred_nbsvm: np.ndarray,
    pred_combined: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the preregistered target-free component-direction gate."""

    arrays = [pred_v02, pred_semantic, pred_bge_base, pred_nbsvm, pred_combined]
    if len({np.asarray(values).shape for values in arrays}) != 1:
        raise ValueError("All prediction arrays must have identical shapes.")
    if not all(np.isfinite(np.asarray(values, dtype=np.float64)).all() for values in arrays):
        raise ValueError("Agreement-gate inputs must be finite.")

    bge_direction = np.asarray(pred_bge_base) - np.asarray(pred_semantic)
    nb_direction = np.asarray(pred_nbsvm) - np.asarray(pred_v02)
    agrees = bge_direction * nb_direction >= 0.0
    gated = np.where(agrees, pred_combined, pred_v02).astype(np.float64)
    return np.clip(gated, 1e-6, 1.0 - 1e-6), agrees


def _read_concat(run_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path / "oof_predictions.parquet") for path in run_dirs]
    if not frames:
        raise ValueError("At least one source run is required.")
    return pd.concat(frames, ignore_index=True)


def _unique_protocol_rows(
    frame: pd.DataFrame,
    columns: list[str],
    source: str,
) -> pd.DataFrame:
    result = frame.loc[:, columns].copy()
    if result.duplicated(["protocol", "response_id"]).any():
        raise ValueError(f"{source} contains duplicate protocol-response rows.")
    return result


def assemble_gate_inputs(
    project_root: str | Path,
    robust_run_ids: Iterable[str],
    bge_run_ids: Iterable[str],
    nb_run_ids: Iterable[str],
    combined_run_ids: Iterable[str],
) -> pd.DataFrame:
    paths = discover_project_paths(project_root)
    runs_root = paths.experiments_dir / "runs"

    robust = _read_concat([runs_root / run_id for run_id in robust_run_ids])
    robust = _unique_protocol_rows(
        robust,
        ["protocol", "response_id", "target", "fold", "pred_v02", "pred_semantic"],
        "robust inputs",
    )

    bge = _read_concat([runs_root / run_id for run_id in bge_run_ids])
    bge = bge.loc[np.isclose(bge["regularization_c"], LOCKED_BGE_C)]
    bge = _unique_protocol_rows(
        bge,
        ["protocol", "response_id", "pred_bge_base"],
        "locked BGE inputs",
    )

    nb = _read_concat([runs_root / run_id for run_id in nb_run_ids])
    nb = nb.loc[nb["model"].eq(LOCKED_NB_MODEL)]
    nb = _unique_protocol_rows(
        nb,
        ["protocol", "response_id", "pred_nbsvm"],
        "locked NB-SVM inputs",
    )

    combined = _read_concat([runs_root / run_id for run_id in combined_run_ids])
    if "model" in combined.columns:
        models = combined["model"].dropna().unique()
        if len(models) != 1:
            raise ValueError("Combined inputs must contain exactly one locked model.")
    combined = _unique_protocol_rows(
        combined,
        ["protocol", "response_id", "pred_combined"],
        "combined inputs",
    )

    keys = ["protocol", "response_id"]
    merged = robust.merge(bge, on=keys, how="inner", validate="one_to_one")
    merged = merged.merge(nb, on=keys, how="inner", validate="one_to_one")
    merged = merged.merge(combined, on=keys, how="inner", validate="one_to_one")
    expected = set(robust["protocol"].unique())
    observed = set(merged["protocol"].unique())
    if not observed:
        raise ValueError("No common protocols were found across component runs.")
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"Component inputs are incomplete for protocols: {missing}")
    counts = merged.groupby("protocol")["response_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Component protocols do not contain the same number of responses.")
    return merged.sort_values(["protocol", "response_id"]).reset_index(drop=True)


def evaluate_agreement_gate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for protocol, group in frame.groupby("protocol", sort=True):
        gated, agrees = component_agreement_prediction(
            group["pred_v02"].to_numpy(dtype=np.float64),
            group["pred_semantic"].to_numpy(dtype=np.float64),
            group["pred_bge_base"].to_numpy(dtype=np.float64),
            group["pred_nbsvm"].to_numpy(dtype=np.float64),
            group["pred_combined"].to_numpy(dtype=np.float64),
        )
        target = group["target"].to_numpy(dtype=np.int8)
        baseline = group["pred_v02"].to_numpy(dtype=np.float64)
        combined = group["pred_combined"].to_numpy(dtype=np.float64)
        base_metrics = binary_metrics(target, baseline)
        combined_metrics = binary_metrics(target, combined)
        gated_metrics = binary_metrics(target, gated)
        metric_rows.append(
            {
                "protocol": protocol,
                **gated_metrics,
                "agreement_rate": float(agrees.mean()),
                "baseline_log_loss": base_metrics["log_loss"],
                "baseline_roc_auc": base_metrics["roc_auc"],
                "combined_log_loss": combined_metrics["log_loss"],
                "combined_roc_auc": combined_metrics["roc_auc"],
                "delta_log_loss_vs_v02": gated_metrics["log_loss"] - base_metrics["log_loss"],
                "delta_roc_auc_vs_v02": gated_metrics["roc_auc"] - base_metrics["roc_auc"],
                "delta_log_loss_vs_combined": gated_metrics["log_loss"] - combined_metrics["log_loss"],
                "delta_roc_auc_vs_combined": gated_metrics["roc_auc"] - combined_metrics["roc_auc"],
            }
        )
        for fold in sorted(group["fold"].unique()):
            mask = group["fold"].to_numpy() == fold
            fold_base = binary_metrics(target[mask], baseline[mask])
            fold_combined = binary_metrics(target[mask], combined[mask])
            fold_gated = binary_metrics(target[mask], gated[mask])
            fold_rows.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    **fold_gated,
                    "agreement_rate": float(agrees[mask].mean()),
                    "delta_log_loss_vs_v02": fold_gated["log_loss"] - fold_base["log_loss"],
                    "delta_roc_auc_vs_v02": fold_gated["roc_auc"] - fold_base["roc_auc"],
                    "combined_delta_log_loss_vs_v02": fold_combined["log_loss"] - fold_base["log_loss"],
                    "combined_delta_roc_auc_vs_v02": fold_combined["roc_auc"] - fold_base["roc_auc"],
                }
            )
        output = group.copy()
        output["component_agrees"] = agrees
        output["pred_agreement_gate"] = gated
        output_frames.append(output)

    return (
        pd.concat(output_frames, ignore_index=True),
        pd.DataFrame(metric_rows).sort_values("protocol").reset_index(drop=True),
        pd.DataFrame(fold_rows).sort_values(["protocol", "fold"]).reset_index(drop=True),
    )


def pilot_continuation(metrics: pd.DataFrame, folds: pd.DataFrame) -> dict[str, object]:
    if len(metrics) != 1:
        raise ValueError("The pilot continuation decision requires exactly one protocol.")
    row = metrics.iloc[0]
    combined_loss_gain = -float(row["combined_log_loss"] - row["baseline_log_loss"])
    combined_auc_gain = float(row["combined_roc_auc"] - row["baseline_roc_auc"])
    gated_loss_gain = -float(row["delta_log_loss_vs_v02"])
    gated_auc_gain = float(row["delta_roc_auc_vs_v02"])
    loss_gain_over_combined = -float(row["delta_log_loss_vs_combined"])
    combined_worst = float(folds["combined_delta_log_loss_vs_v02"].max())
    gated_worst = float(folds["delta_log_loss_vs_v02"].max())
    worst_fold_reduction = combined_worst - gated_worst
    retains_auc = gated_auc_gain >= 0.8 * combined_auc_gain
    retains_both = retains_auc and gated_loss_gain >= 0.8 * combined_loss_gain
    path_a = loss_gain_over_combined >= 0.0003 and retains_auc
    path_b = worst_fold_reduction >= 0.001 and retains_both
    return {
        "combined_loss_gain": combined_loss_gain,
        "combined_auc_gain": combined_auc_gain,
        "gated_loss_gain": gated_loss_gain,
        "gated_auc_gain": gated_auc_gain,
        "loss_gain_over_combined": loss_gain_over_combined,
        "combined_worst_fold_log_loss_regression": combined_worst,
        "gated_worst_fold_log_loss_regression": gated_worst,
        "worst_fold_regression_reduction": worst_fold_reduction,
        "retains_80pct_auc_gain": bool(retains_auc),
        "retains_80pct_loss_and_auc_gain": bool(retains_both),
        "passes_path_a": bool(path_a),
        "passes_path_b": bool(path_b),
        "continue_confirmation": bool(path_a or path_b),
    }


def run_agreement_gate_validation(
    project_root: str | Path,
    robust_run_ids: Iterable[str],
    bge_run_ids: Iterable[str],
    nb_run_ids: Iterable[str],
    combined_run_ids: Iterable[str],
    pilot: bool,
) -> dict[str, object]:
    frame = assemble_gate_inputs(
        project_root,
        robust_run_ids=robust_run_ids,
        bge_run_ids=bge_run_ids,
        nb_run_ids=nb_run_ids,
        combined_run_ids=combined_run_ids,
    )
    predictions, metrics, folds = evaluate_agreement_gate(frame)
    decision = pilot_continuation(metrics, folds) if pilot else None

    paths = discover_project_paths(project_root)
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_agreement_gate_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    metrics.to_csv(run_dir / "protocol_metrics.csv", index=False, lineterminator="\n")
    folds.to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "pilot": pilot,
        "locked_bge_c": LOCKED_BGE_C,
        "locked_nb_model": LOCKED_NB_MODEL,
        "robust_run_ids": list(robust_run_ids),
        "bge_run_ids": list(bge_run_ids),
        "nb_run_ids": list(nb_run_ids),
        "combined_run_ids": list(combined_run_ids),
        "pilot_decision": decision,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "pilot_decision": decision}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the locked component-agreement gate.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--robust-run-id", action="append", required=True)
    parser.add_argument("--bge-run-id", action="append", required=True)
    parser.add_argument("--nb-run-id", action="append", required=True)
    parser.add_argument("--combined-run-id", action="append", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_agreement_gate_validation(
        args.project_root,
        robust_run_ids=args.robust_run_id,
        bge_run_ids=args.bge_run_id,
        nb_run_ids=args.nb_run_id,
        combined_run_ids=args.combined_run_id,
        pilot=args.pilot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
