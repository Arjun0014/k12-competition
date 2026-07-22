from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.environment_component_validation import (
    ALLOWED_ENVIRONMENTS,
    DEVELOPMENT_ENVIRONMENTS,
    paired_bootstrap_summary,
    prediction_metric_tables,
)
from trace_ace.io import discover_project_paths
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


PHASE_C_RUN_ID = "20260720T075633Z_environment_component_validation"
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
REGULARIZATION_C = 0.10
PUBLIC_BASELINE_LOG_LOSS = 0.6081
TOP5_TARGET_LOG_LOSS = 0.6038
TOP5_REQUIRED_GAIN = PUBLIC_BASELINE_LOG_LOSS - TOP5_TARGET_LOG_LOSS
BACKUP_REQUIRED_GAIN = 0.0010
MAX_ENVIRONMENT_REGRESSION = 0.0005
BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _cache_paths(project_root: str | Path, transfer_run_id: str) -> tuple[Path, Path, dict]:
    paths = discover_project_paths(project_root)
    transfer_report_path = paths.experiments_dir / "runs" / transfer_run_id / "report.json"
    if not transfer_report_path.exists():
        raise FileNotFoundError(f"Missing E400 transfer report: {transfer_report_path}")
    transfer_report = json.loads(transfer_report_path.read_text(encoding="utf-8"))
    if not bool(transfer_report.get("passes_external_gate")):
        raise RuntimeError("Competition validation is forbidden because the external gate failed.")
    prefix = str(transfer_report["delta_sha256"])[:12]
    pooled_path = paths.cache_dir / f"sra_deberta_session_pooled_256_{prefix}.npy"
    logits_path = paths.cache_dir / f"sra_deberta_session_logits_256_{prefix}.npy"
    metadata_path = paths.cache_dir / f"sra_deberta_session_256_{prefix}.metadata.json"
    if not (pooled_path.exists() and logits_path.exists() and metadata_path.exists()):
        raise FileNotFoundError("The adapted competition cache is incomplete.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("delta_sha256") != transfer_report.get("delta_sha256"):
        raise ValueError("Adapted cache and transfer checkpoint do not match.")
    return pooled_path, logits_path, metadata


def _component_frame(project_root: str | Path, environment: str) -> pd.DataFrame:
    paths = discover_project_paths(project_root)
    component_path = (
        paths.experiments_dir
        / "runs"
        / PHASE_C_RUN_ID
        / f"component_oof_{environment}.parquet"
    )
    if not component_path.exists():
        raise FileNotFoundError(f"Missing frozen component OOF: {component_path}")
    frame = pd.read_parquet(component_path).reset_index(drop=True)
    required = {
        "environment",
        "response_id",
        "session_id",
        "semantic_family",
        "fold",
        "evaluation_eligible",
        "target",
        "pred_full",
        "pred_role",
        "pred_bge_small",
        "pred_bge_base",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Frozen component OOF is missing columns: {missing}")
    if set(frame["environment"].astype(str)) != {environment}:
        raise ValueError(f"Frozen component OOF has the wrong environment: {environment}")
    return frame


def _selection_table(environment_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, group in environment_metrics.loc[
        environment_metrics["environment"].isin(DEVELOPMENT_ENVIRONMENTS)
    ].groupby("candidate", sort=True):
        if set(group["environment"].astype(str)) != set(DEVELOPMENT_ENVIRONMENTS):
            raise ValueError(f"Incomplete E400 development metrics for {candidate}.")
        loss_delta = group["delta_log_loss_vs_v02"].to_numpy(dtype=np.float64)
        row = {
            "candidate": str(candidate),
            "mean_log_loss_gain": float(-loss_delta.mean()),
            "worst_environment_regression": float(max(0.0, loss_delta.max())),
            "improved_environments": int(np.sum(loss_delta < 0.0)),
            "mean_auroc_gain": float(group["delta_roc_auc_vs_v02"].mean()),
            "mean_brier_gain": float(-group["delta_brier_score_vs_v02"].mean()),
            "mean_ece_gain": float(-group["delta_ece_10_vs_v02"].mean()),
        }
        row["passes_development_guard"] = bool(
            row["mean_log_loss_gain"] > 0.0
            and row["worst_environment_regression"] <= MAX_ENVIRONMENT_REGRESSION
            and row["mean_auroc_gain"] >= 0.0
            and row["mean_brier_gain"] >= 0.0
            and row["mean_ece_gain"] >= 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes_development_guard", "mean_log_loss_gain"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _gate_decisions(
    selected_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, object]:
    loss_delta = selected_metrics["delta_log_loss_vs_v02"].to_numpy(dtype=np.float64)
    mean_gain = float(-loss_delta.mean())
    improved = int(np.sum(loss_delta < 0.0))
    worst_regression = float(max(0.0, loss_delta.max()))
    mean_auroc_gain = float(selected_metrics["delta_roc_auc_vs_v02"].mean())
    mean_brier_gain = float(-selected_metrics["delta_brier_score_vs_v02"].mean())
    mean_ece_gain = float(-selected_metrics["delta_ece_10_vs_v02"].mean())
    macro = bootstrap.loc[bootstrap["environment"].eq("ALL_MACRO")].set_index("resampler")
    session_probability = float(macro.loc["session", "probability_gain_positive"])
    family_probability = float(
        macro.loc["objective_family", "probability_gain_positive"]
    )
    common = bool(
        worst_regression <= MAX_ENVIRONMENT_REGRESSION
        and mean_auroc_gain >= 0.0
        and mean_brier_gain >= 0.0
        and mean_ece_gain >= 0.0
    )
    backup = bool(
        common
        and mean_gain >= BACKUP_REQUIRED_GAIN
        and improved >= 3
        and session_probability >= 0.90
        and family_probability >= 0.90
    )
    top5 = bool(
        common
        and mean_gain >= TOP5_REQUIRED_GAIN
        and improved == len(ALLOWED_ENVIRONMENTS)
        and session_probability >= 0.95
        and family_probability >= 0.95
    )
    projected = PUBLIC_BASELINE_LOG_LOSS - mean_gain
    return {
        "mean_log_loss_gain": mean_gain,
        "improved_environments": improved,
        "worst_environment_regression": worst_regression,
        "mean_auroc_gain": mean_auroc_gain,
        "mean_brier_gain": mean_brier_gain,
        "mean_ece_gain": mean_ece_gain,
        "session_probability_gain_positive": session_probability,
        "objective_family_probability_gain_positive": family_probability,
        "passes_backup_gate": backup,
        "passes_top5_gate": top5,
        "projected_public_log_loss": projected,
        "top5_target_log_loss": TOP5_TARGET_LOG_LOSS,
        "distance_to_top5_target": projected - TOP5_TARGET_LOG_LOSS,
        "rank_projection": (
            "credible_top5_candidate" if top5 else "not_yet_top5_evidence"
        ),
    }


def run_external_sra_validation(
    project_root: str | Path,
    transfer_run_id: str,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    pooled_path, logits_path, cache_metadata = _cache_paths(project_root, transfer_run_id)
    pooled = np.load(pooled_path).astype(np.float32)
    logits = np.load(logits_path).astype(np.float32)
    if pooled.shape != (len(frame), 768) or logits.shape != (len(frame), 3):
        raise ValueError("Adapted E400 cache shape does not match modeling rows.")
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    if (norm <= 0.0).any():
        raise ValueError("Adapted E400 pooled embeddings contain zero vectors.")
    features = pooled / norm
    probability = _softmax(logits)
    _, semantic_dense, semantic_dense_names = prepare_semantic_features(
        paths.cache_dir, frame
    )
    adapted_dense = np.column_stack(
        [
            semantic_dense,
            logits,
            probability,
            logits[:, 1] - logits[:, 0],
            probability[:, 1] - probability[:, 0],
        ]
    )
    if not np.isfinite(adapted_dense).all():
        raise RuntimeError("Adapted E400 feature block contains non-finite values.")

    environment_predictions: dict[str, pd.DataFrame] = {}
    development_rows: list[pd.DataFrame] = []
    for environment in ALLOWED_ENVIRONMENTS:
        component = _component_frame(project_root, environment)
        if list(component["response_id"].astype(str)) != list(frame["response_id"].astype(str)):
            raise ValueError(f"E400 response order differs in {environment}.")
        folds = component["fold"].to_numpy(dtype=np.int8)
        adapted_prediction = semantic_logistic_oof(
            frame,
            folds,
            features,
            adapted_dense,
            regularization_c=REGULARIZATION_C,
            n_splits=5,
        )
        baseline = np.clip(
            0.25 * component["pred_full"].to_numpy(dtype=np.float64)
            + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
            + 0.50 * component["pred_bge_small"].to_numpy(dtype=np.float64),
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        bge_replacement = np.clip(
            0.25 * component["pred_full"].to_numpy(dtype=np.float64)
            + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
            + 0.50 * component["pred_bge_base"].to_numpy(dtype=np.float64),
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        rows: list[pd.DataFrame] = []
        for weight in BLEND_WEIGHTS:
            output = component[
                [
                    "environment",
                    "response_id",
                    "session_id",
                    "learning_objective_id",
                    "semantic_family",
                    "fold",
                    "evaluation_eligible",
                    "target",
                ]
            ].copy()
            output["candidate"] = f"e400_sra_blend_{int(round(100 * weight)):02d}"
            output["calibration"] = "raw"
            output["prediction"] = np.clip(
                (1.0 - weight) * bge_replacement + weight * adapted_prediction,
                PROBABILITY_CLIP,
                1.0 - PROBABILITY_CLIP,
            )
            output["pred_v02"] = baseline
            rows.append(output)
        environment_predictions[environment] = pd.concat(rows, ignore_index=True)
        if environment in DEVELOPMENT_ENVIRONMENTS:
            development_rows.append(environment_predictions[environment])

    development = pd.concat(development_rows, ignore_index=True)
    development_fold_metrics, development_environment_metrics = prediction_metric_tables(
        development
    )
    selection = _selection_table(development_environment_metrics)
    passing = selection.loc[selection["passes_development_guard"].astype(bool)]
    selected_candidate = None if passing.empty else str(passing.iloc[0]["candidate"])

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_external_sra_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    development.to_parquet(run_dir / "development_predictions.parquet", index=False)
    development_fold_metrics.to_csv(run_dir / "development_fold_metrics.csv", index=False)
    development_environment_metrics.to_csv(
        run_dir / "development_environment_metrics.csv", index=False
    )
    selection.to_csv(run_dir / "selection.csv", index=False)

    gate = None
    if selected_candidate is not None:
        selected = pd.concat(
            [
                environment_predictions[environment].loc[
                    environment_predictions[environment]["candidate"].eq(selected_candidate)
                ]
                for environment in ALLOWED_ENVIRONMENTS
            ],
            ignore_index=True,
        )
        fold_metrics, environment_metrics = prediction_metric_tables(selected)
        bootstrap = paired_bootstrap_summary(
            selected,
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=20260722,
        )
        selected.to_parquet(run_dir / "selected_predictions.parquet", index=False)
        fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
        environment_metrics.to_csv(run_dir / "environment_metrics.csv", index=False)
        bootstrap.to_csv(run_dir / "bootstrap.csv", index=False)
        gate = _gate_decisions(environment_metrics, bootstrap)

    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate_family": "E400_sem_eval_adapted_deberta_blend",
        "transfer_run_id": transfer_run_id,
        "transfer_delta_sha256": cache_metadata["delta_sha256"],
        "phase_c_component_run_id": PHASE_C_RUN_ID,
        "blend_weights_preregistered": list(BLEND_WEIGHTS),
        "regularization_c": REGULARIZATION_C,
        "adapted_dense_columns": semantic_dense_names
        + [
            "logit_contradiction",
            "logit_entailment",
            "logit_neutral",
            "prob_contradiction",
            "prob_entailment",
            "prob_neutral",
            "logit_entailment_minus_contradiction",
            "prob_entailment_minus_contradiction",
        ],
        "selected_candidate": selected_candidate,
        "development_guard_passed": selected_candidate is not None,
        "gate": gate,
        "backup_zip_required": bool(gate and gate["passes_backup_gate"]),
        "continue_research": not bool(gate and gate["passes_top5_gate"]),
        "V_joint_status": "confirmation_only_after_development_weight_selection",
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the preregistered SemEval-adapted DeBERTa blend."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--transfer-run-id", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_external_sra_validation(args.project_root, args.transfer_run_id)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
