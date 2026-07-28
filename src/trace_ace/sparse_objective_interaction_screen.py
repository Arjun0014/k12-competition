from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
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


PROTOCOL_ID = "E720_sparse_objective_role_interactions_v1"
PILOT_ROWS = 4_096
HASH_FEATURES = 2**19
OBJECTIVE_TOKEN_CAP = 12
ROLE_TOKEN_CAP = 64
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
REGULARIZATION_C = 0.1
BOOTSTRAP_REPLICATES = 5_000
SEED = 20260728
MAX_RSS_BYTES = 8 * 1024**3
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_tokens(text: str, cap: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        if token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) == cap:
            break
    return result


def interaction_features(
    objective: str, objective_context: str
) -> dict[str, float]:
    objective_tokens = _unique_tokens(objective, OBJECTIVE_TOKEN_CAP)
    student_tokens: list[str] = []
    tutor_tokens: list[str] = []
    student_seen: set[str] = set()
    tutor_seen: set[str] = set()
    for line in objective_context.splitlines():
        if line.startswith("[STUDENT]"):
            destination, seen = student_tokens, student_seen
        elif line.startswith("[TUTOR]"):
            destination, seen = tutor_tokens, tutor_seen
        else:
            continue
        if len(destination) >= ROLE_TOKEN_CAP:
            continue
        for token in TOKEN_RE.findall(line.lower()):
            if token not in seen:
                seen.add(token)
                destination.append(token)
            if len(destination) == ROLE_TOKEN_CAP:
                break
    features: dict[str, float] = {}
    for token in objective_tokens:
        features[f"o={token}"] = 1.0
    for token in student_tokens:
        features[f"s={token}"] = 1.0
    for token in tutor_tokens:
        features[f"t={token}"] = 1.0
    for objective_token in objective_tokens:
        for token in student_tokens:
            features[f"os={objective_token}|{token}"] = 1.0
        for token in tutor_tokens:
            features[f"ot={objective_token}|{token}"] = 1.0
    return features


def build_matrix(
    frame: pd.DataFrame, contexts: pd.DataFrame
) -> sparse.csr_matrix:
    if list(frame["response_id"].astype(str)) != list(
        contexts["response_id"].astype(str)
    ):
        raise ValueError("E720 context rows do not align.")
    dictionaries = [
        interaction_features(objective, context)
        for objective, context in zip(
            frame["learning_objective"].astype(str),
            contexts["objective_context"].astype(str),
            strict=True,
        )
    ]
    matrix = FeatureHasher(
        n_features=HASH_FEATURES,
        input_type="dict",
        alternate_sign=True,
    ).transform(dictionaries)
    matrix = matrix.tocsr().astype(np.float32)
    if (
        matrix.shape != (len(frame), HASH_FEATURES)
        or matrix.nnz == 0
        or not np.isfinite(matrix.data).all()
    ):
        raise ValueError("E720 hashed matrix failed its contract.")
    return matrix


def _fit_sparse_fold(
    matrix: sparse.csr_matrix,
    dense: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    scaler = StandardScaler()
    dense_train = scaler.fit_transform(dense[train_mask]).astype(np.float32)
    dense_validation = scaler.transform(dense[validation_mask]).astype(np.float32)
    x_train = sparse.hstack(
        [
            matrix[train_mask],
            sparse.csr_matrix(dense_train * np.float32(0.08)),
        ],
        format="csr",
    )
    x_validation = sparse.hstack(
        [
            matrix[validation_mask],
            sparse.csr_matrix(dense_validation * np.float32(0.08)),
        ],
        format="csr",
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        penalty="l2",
        max_iter=20,
        tol=None,
        average=True,
        random_state=SEED,
        class_weight=None,
    )
    model.fit(x_train, target[train_mask])
    prediction = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("E720 fold predictions are non-finite.")
    return prediction, int(model.n_iter_)


def sparse_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    matrix: sparse.csr_matrix,
    dense: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target = frame["target"].to_numpy(dtype=np.int8)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        values, iterations = _fit_sparse_fold(
            matrix, dense, target, train_mask, validation_mask
        )
        prediction[validation_mask] = values
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "iterations": iterations,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("E720 OOF predictions are incomplete.")
    return prediction, summaries


def _load_inputs(
    project_root: str | Path, rows: int | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    paths = discover_project_paths(project_root)
    frame, _, indices, folds = _load_pilot(project_root)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    ).iloc[indices].reset_index(drop=True)
    if rows is not None:
        frame = frame.iloc[:rows].reset_index(drop=True)
        indices = indices[:rows]
        folds = folds[:rows]
        contexts = contexts.iloc[:rows].reset_index(drop=True)
    return frame, indices, folds, contexts


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    frame, indices, folds, contexts = _load_inputs(project_root, rows=512)
    started = time.perf_counter()
    matrix = build_matrix(frame, contexts)
    paths = discover_project_paths(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    _, dense_full, _ = prepare_semantic_features(paths.cache_dir, full)
    dense = dense_full[indices]
    fold = int(folds[0])
    validation_mask = folds == fold
    train_mask = ~validation_mask
    synthetic = (indices % 2).astype(np.int8)
    _, iterations = _fit_sparse_fold(
        matrix, dense, synthetic, train_mask, validation_mask
    )
    elapsed = time.perf_counter() - started
    projected = elapsed * PILOT_ROWS / len(frame) * 5
    result = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "benchmark_rows": len(frame),
        "matrix_nnz": int(matrix.nnz),
        "matrix_density": float(matrix.nnz / np.prod(matrix.shape)),
        "one_fold_iterations": iterations,
        "elapsed_seconds": elapsed,
        "projected_five_fold_seconds": projected,
        "peak_rss_bytes": _current_rss_bytes(),
        "proceed": bool(projected <= 2 * 3600 and _current_rss_bytes() < MAX_RSS_BYTES),
    }
    path = paths.cache_dir / "sparse_objective_e720_benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["benchmark_path"] = str(path)
    result["benchmark_sha256"] = _sha256(path)
    return result


def build_cache(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    frame, _, _, contexts = _load_inputs(project_root)
    started = time.perf_counter()
    matrix = build_matrix(frame, contexts)
    cache_path = paths.cache_dir / "sparse_objective_e720_pilot.npz"
    metadata_path = paths.cache_dir / "sparse_objective_e720.metadata.json"
    sparse.save_npz(cache_path, matrix, compressed=True)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _current_rss_bytes(),
        "cache_sha256": _sha256(cache_path),
        "response_sha256": hashlib.sha256(
            "\n".join(frame["response_id"].astype(str)).encode()
        ).hexdigest(),
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **metadata,
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    frame, indices, folds, _ = _load_inputs(project_root)
    cache_path = paths.cache_dir / "sparse_objective_e720_pilot.npz"
    metadata_path = paths.cache_dir / "sparse_objective_e720.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["cache_sha256"] != _sha256(cache_path):
        raise ValueError("E720 cache hash changed.")
    matrix = sparse.load_npz(cache_path).tocsr()
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features, base_similarity = prepare_bge_base_features(paths.cache_dir, full)
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    dense = dense_full[indices].copy()
    dense[:, -1:] = base_similarity[indices]
    baseline = semantic_logistic_oof(
        frame,
        folds,
        base_features[indices],
        dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    candidate, fit_summaries = sparse_oof(frame, folds, matrix, dense)
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
                    "pred_sparse_cross": candidate,
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_sparse_objective_interaction")
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
        "cache_sha256": metadata["cache_sha256"],
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
    parser = argparse.ArgumentParser(description="Run frozen E720 sparse screen.")
    parser.add_argument("stage", choices=("benchmark", "build-cache", "validate"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build-cache":
        result = build_cache(args.project_root)
    else:
        result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
