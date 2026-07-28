from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.bge_base_dual_pooling import _load_pilot
from trace_ace.bge_base_multiview_screen import (
    _current_rss_bytes,
    _session_bootstrap,
    assert_runtime,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


PROTOCOL_ID = "E730_centered_objective_procrustes_v1"
PILOT_ROWS = 4_096
EMBEDDING_DIMENSION = 768
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
MAX_RSS_BYTES = 8 * 1024**3


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_procrustes(
    context: np.ndarray, objective: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        context.shape != objective.shape
        or context.ndim != 2
        or context.shape[1] != EMBEDDING_DIMENSION
    ):
        raise ValueError("E730 Procrustes inputs have invalid shape.")
    context_mean = context.mean(axis=0, dtype=np.float64)
    objective_mean = objective.mean(axis=0, dtype=np.float64)
    cross_covariance = (
        (context.astype(np.float64) - context_mean).T
        @ (objective.astype(np.float64) - objective_mean)
    )
    left, _, right = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right
    if (
        rotation.shape != (EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        or not np.isfinite(rotation).all()
        or not np.allclose(
            rotation.T @ rotation,
            np.eye(EMBEDDING_DIMENSION),
            atol=2e-10,
            rtol=0,
        )
    ):
        raise ValueError("E730 rotation failed its orthogonality audit.")
    return context_mean, objective_mean, rotation


def transform_aligned(
    context: np.ndarray,
    objective: np.ndarray,
    context_mean: np.ndarray,
    objective_mean: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mapped = (context.astype(np.float64) - context_mean) @ rotation
    centered_objective = objective.astype(np.float64) - objective_mean
    mapped_norm = np.linalg.norm(mapped, axis=1, keepdims=True)
    objective_norm = np.linalg.norm(centered_objective, axis=1, keepdims=True)
    if np.any(mapped_norm <= 0) or np.any(objective_norm <= 0):
        raise ValueError("E730 centered embedding has zero norm.")
    mapped = (mapped / mapped_norm).astype(np.float32)
    centered_objective = (centered_objective / objective_norm).astype(np.float32)
    return mapped, centered_objective


def aligned_features(
    mapped_context: np.ndarray, centered_objective: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if mapped_context.shape != centered_objective.shape:
        raise ValueError("E730 aligned embeddings do not match.")
    features = np.column_stack(
        [
            mapped_context * np.float32(0.7),
            centered_objective * np.float32(0.7),
            mapped_context * centered_objective * np.float32(6.0),
            np.abs(mapped_context - centered_objective) * np.float32(0.7),
        ]
    ).astype(np.float32)
    similarity = np.sum(
        mapped_context * centered_objective, axis=1, keepdims=True
    ).astype(np.float32)
    return features, similarity


def _fit_fold(
    context: np.ndarray,
    objective: np.ndarray,
    dense: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    fold: int,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    context_mean, objective_mean, rotation = fit_procrustes(
        context[train_mask], objective[train_mask]
    )
    mapped_train, objective_train = transform_aligned(
        context[train_mask],
        objective[train_mask],
        context_mean,
        objective_mean,
        rotation,
    )
    mapped_validation, objective_validation = transform_aligned(
        context[validation_mask],
        objective[validation_mask],
        context_mean,
        objective_mean,
        rotation,
    )
    x_train, similarity_train = aligned_features(mapped_train, objective_train)
    x_validation, similarity_validation = aligned_features(
        mapped_validation, objective_validation
    )
    dense_train_raw = dense[train_mask].copy()
    dense_validation_raw = dense[validation_mask].copy()
    dense_train_raw[:, -1:] = similarity_train
    dense_validation_raw[:, -1:] = similarity_validation
    scaler = StandardScaler()
    dense_train = scaler.fit_transform(dense_train_raw).astype(np.float32)
    dense_validation = scaler.transform(dense_validation_raw).astype(np.float32)
    x_train = np.column_stack(
        [x_train, dense_train * np.float32(0.08)]
    )
    x_validation = np.column_stack(
        [x_validation, dense_validation * np.float32(0.08)]
    )
    model = LogisticRegression(
        C=REGULARIZATION_C,
        solver="lbfgs",
        max_iter=400,
        tol=1e-5,
        random_state=20260728 + fold,
    )
    model.fit(x_train, target[train_mask])
    prediction = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("E730 fold predictions are non-finite.")
    summary = {
        "fold": fold,
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "iterations": int(model.n_iter_[0]),
        "aligned_training_cosine_mean": float(similarity_train.mean()),
        "aligned_validation_cosine_mean": float(similarity_validation.mean()),
        "rotation_identity_frobenius_per_dimension": float(
            np.linalg.norm(rotation - np.eye(EMBEDDING_DIMENSION))
            / EMBEDDING_DIMENSION
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return prediction, summary


def aligned_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    context: np.ndarray,
    objective: np.ndarray,
    dense: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target = frame["target"].to_numpy(dtype=np.int8)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        values, summary = _fit_fold(
            context,
            objective,
            dense,
            target,
            train_mask,
            validation_mask,
            fold=fold,
        )
        prediction[validation_mask] = values
        summaries.append(summary)
        if _current_rss_bytes() >= MAX_RSS_BYTES:
            raise MemoryError("E730 exceeded the 8 GiB RSS gate.")
    if not np.isfinite(prediction).all():
        raise RuntimeError("E730 OOF predictions are incomplete.")
    return prediction, summaries


def _load_inputs(
    project_root: str | Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    paths = discover_project_paths(project_root)
    frame, _, indices, folds = _load_pilot(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    context = np.load(
        paths.cache_dir / "bge_base_context_256.npy", mmap_mode="r"
    )[indices].astype(np.float32)
    objective = np.load(
        paths.cache_dir / "bge_base_objective_256.npy", mmap_mode="r"
    )[indices].astype(np.float32)
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    return (
        frame,
        indices,
        folds,
        context,
        objective,
        dense_full[indices].copy(),
        dense_names,
    )


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    frame, indices, folds, context, objective, dense, _ = _load_inputs(project_root)
    train_mask, validation_mask = _fold_masks(frame, folds, 0)
    synthetic = (indices % 2).astype(np.int8)
    _, summary = _fit_fold(
        context,
        objective,
        dense,
        synthetic,
        train_mask,
        validation_mask,
        fold=0,
    )
    projected = summary["elapsed_seconds"] * 5
    result = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "one_fold": summary,
        "projected_five_fold_seconds": projected,
        "peak_rss_bytes": _current_rss_bytes(),
        "proceed": bool(projected <= 2 * 3600 and _current_rss_bytes() < MAX_RSS_BYTES),
    }
    paths = discover_project_paths(project_root)
    path = paths.cache_dir / "objective_alignment_e730_benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["benchmark_path"] = str(path)
    result["benchmark_sha256"] = _sha256(path)
    return result


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    (
        frame,
        indices,
        folds,
        context,
        objective,
        dense,
        dense_names,
    ) = _load_inputs(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features, base_similarity = prepare_bge_base_features(paths.cache_dir, full)
    baseline_dense = dense.copy()
    baseline_dense[:, -1:] = base_similarity[indices]
    baseline = semantic_logistic_oof(
        frame,
        folds,
        base_features[indices],
        baseline_dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    candidate, fit_summaries = aligned_oof(
        frame, folds, context, objective, dense
    )
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline_metrics = binary_metrics(target, baseline)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for weight in BLEND_WEIGHTS:
        prediction = (1 - weight) * baseline + weight * candidate
        metrics = binary_metrics(target, prediction)
        metric_rows.append(
            {
                "blend_weight": weight,
                **metrics,
                **{
                    f"delta_{name}_vs_bge": metrics[name] - baseline_metrics[name]
                    for name in ("log_loss", "roc_auc", "brier_score", "ece_10")
                },
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "response_id": frame["response_id"],
                    "session_id": frame["session_id"],
                    "fold": folds,
                    "target": target,
                    "blend_weight": weight,
                    "pred_bge_base": baseline,
                    "pred_aligned": candidate,
                    "prediction": prediction,
                }
            )
        )
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["log_loss", "roc_auc", "blend_weight"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = metrics_frame.iloc[0].to_dict()
    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_predictions = all_predictions.loc[
        all_predictions["blend_weight"].eq(selected["blend_weight"])
    ].reset_index(drop=True)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = selected_predictions["fold"].to_numpy() == fold
        base_fold = binary_metrics(target[mask], baseline[mask])
        candidate_fold = binary_metrics(
            target[mask], selected_predictions.loc[mask, "prediction"].to_numpy()
        )
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "baseline_log_loss": base_fold["log_loss"],
                "candidate_log_loss": candidate_fold["log_loss"],
                "delta_log_loss_vs_bge": (
                    candidate_fold["log_loss"] - base_fold["log_loss"]
                ),
            }
        )
    bootstrap = _session_bootstrap(
        frame,
        baseline,
        selected_predictions["prediction"].to_numpy(),
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    max_fold_regression = max(row["delta_log_loss_vs_bge"] for row in fold_rows)
    clauses = {
        "log_loss_gain_at_least_0_0016": (
            selected["delta_log_loss_vs_bge"] <= -0.0016
        ),
        "max_fold_regression_at_most_0_0005": max_fold_regression <= 0.0005,
        "auroc_non_regression": selected["delta_roc_auc_vs_bge"] >= 0,
        "brier_non_regression": selected["delta_brier_score_vs_bge"] <= 0,
        "ece_regression_at_most_0_001": selected["delta_ece_10_vs_bge"] <= 0.001,
        "bootstrap_support_at_least_0_95": (
            bootstrap["support_positive_log_loss_gain"] >= 0.95
        ),
    }
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_objective_alignment")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    selected_predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame([bootstrap]).to_csv(
        run_dir / "bootstrap.csv", index=False, lineterminator="\n"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "runtime": runtime,
        "pilot_rows": PILOT_ROWS,
        "baseline_metrics": baseline_metrics,
        "selected": selected,
        "fold_metrics": fold_rows,
        "fit_summaries": fit_summaries,
        "paired_session_bootstrap": bootstrap,
        "clauses": clauses,
        "passes_screen": bool(all(clauses.values())),
        "validation_elapsed_seconds": time.perf_counter() - started,
        "dense_features": dense_names,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["artifact_hashes"] = {
        name: _sha256(run_dir / name)
        for name in (
            "metrics.csv",
            "fold_metrics.csv",
            "oof_predictions.parquet",
            "bootstrap.csv",
            "report.json",
        )
    }
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run frozen E730 alignment screen.")
    parser.add_argument("stage", choices=("benchmark", "validate"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = (
        benchmark(args.project_root)
        if args.stage == "benchmark"
        else validate(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
