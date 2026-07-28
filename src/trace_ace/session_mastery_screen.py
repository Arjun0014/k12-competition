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
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from trace_ace.bge_base_dual_pooling import _load_pilot
from trace_ace.bge_base_multiview_screen import (
    _current_rss_bytes,
    _session_bootstrap,
    assert_runtime,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.hard_validation import (
    _fold_masks,
    prepare_hash_matrices,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


PROTOCOL_ID = "E740_session_mastery_soft_target_v1"
PILOT_ROWS = 4_096
ALPHA = 3e-5
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
MAX_RSS_BYTES = 8 * 1024**3


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_soft_targets(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
) -> pd.DataFrame:
    if len(frame) != len(train_mask):
        raise ValueError("E740 training mask is misaligned.")
    training = frame.loc[train_mask, ["session_id", "target"]].copy()
    if training.empty:
        raise ValueError("E740 has no legal training rows.")
    grouped = (
        training.groupby("session_id", sort=True, as_index=False)
        .agg(soft_target=("target", "mean"), response_count=("target", "size"))
        .sort_values("session_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        grouped["session_id"].duplicated().any()
        or not grouped["soft_target"].between(0, 1).all()
        or (grouped["response_count"] <= 0).any()
    ):
        raise ValueError("E740 session aggregation failed its audit.")
    return grouped


def soft_label_examples(
    matrix: sparse.csr_matrix,
    soft_target: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    values = np.asarray(soft_target, dtype=np.float64)
    if matrix.shape[0] != len(values) or not np.isfinite(values).all():
        raise ValueError("E740 soft-label inputs are misaligned.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("E740 soft targets must be probabilities.")
    examples = sparse.vstack([matrix, matrix], format="csr")
    labels = np.concatenate(
        [np.ones(len(values), dtype=np.int8), np.zeros(len(values), dtype=np.int8)]
    )
    weights = np.concatenate([values, 1.0 - values])
    if not np.allclose(weights[: len(values)] + weights[len(values) :], 1.0):
        raise ValueError("E740 does not assign equal total weight per session.")
    return examples, labels, weights


def _session_lookup(cache_dir: Path) -> pd.Series:
    order = pd.read_parquet(cache_dir / "hash_session_order.parquet")["session_id"]
    if order.duplicated().any():
        raise ValueError("E740 hash session order contains duplicates.")
    return pd.Series(np.arange(len(order), dtype=np.int64), index=order.astype(str))


def _fit_fold(
    frame: pd.DataFrame,
    session_matrix: sparse.csr_matrix,
    session_lookup: pd.Series,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    fold: int,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    grouped = session_soft_targets(frame, train_mask)
    train_rows = grouped["session_id"].astype(str).map(session_lookup)
    validation_rows = frame.loc[validation_mask, "session_id"].astype(str).map(
        session_lookup
    )
    if train_rows.isna().any() or validation_rows.isna().any():
        raise ValueError("E740 session hash lookup is incomplete.")
    x_sessions = session_matrix[train_rows.to_numpy(dtype=np.int64)]
    examples, labels, weights = soft_label_examples(
        x_sessions, grouped["soft_target"].to_numpy(dtype=np.float64)
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=ALPHA,
        max_iter=200,
        tol=1e-4,
        shuffle=True,
        random_state=20260728 + fold,
        average=True,
    )
    model.fit(examples, labels, sample_weight=weights)
    prediction = model.predict_proba(
        session_matrix[validation_rows.to_numpy(dtype=np.int64)]
    )[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("E740 fold predictions are non-finite.")
    validation_sessions = frame.loc[validation_mask, "session_id"].astype(str)
    audit = pd.DataFrame(
        {"session_id": validation_sessions.to_numpy(), "prediction": prediction}
    )
    if (
        audit.groupby("session_id", sort=False)["prediction"].nunique(dropna=False)
        > 1
    ).any():
        raise ValueError("E740 gave different predictions to one session transcript.")
    summary = {
        "fold": fold,
        "training_rows": int(train_mask.sum()),
        "training_sessions": int(len(grouped)),
        "validation_rows": int(validation_mask.sum()),
        "validation_sessions": int(validation_sessions.nunique()),
        "mean_training_soft_target": float(grouped["soft_target"].mean()),
        "mixed_training_sessions": int(
            grouped["soft_target"].between(0, 1, inclusive="neither").sum()
        ),
        "iterations": int(model.n_iter_),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return prediction, summary


def session_mastery_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    session_matrix: sparse.csr_matrix,
    session_lookup: pd.Series,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        values, summary = _fit_fold(
            frame,
            session_matrix,
            session_lookup,
            train_mask,
            validation_mask,
            fold=fold,
        )
        prediction[validation_mask] = values
        summaries.append(summary)
        if _current_rss_bytes() >= MAX_RSS_BYTES:
            raise MemoryError("E740 exceeded the 8 GiB RSS gate.")
    if not np.isfinite(prediction).all():
        raise RuntimeError("E740 OOF predictions are incomplete.")
    return prediction, summaries


def _load_inputs(
    project_root: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, sparse.csr_matrix, pd.Series]:
    paths = discover_project_paths(project_root)
    frame, _, indices, folds = _load_pilot(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    session_matrix, _ = prepare_hash_matrices(full, paths.cache_dir)
    lookup = _session_lookup(paths.cache_dir)
    if session_matrix.shape[0] != len(lookup):
        raise ValueError("E740 session matrix and order do not match.")
    return frame, indices, folds, session_matrix, lookup


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    frame, indices, folds, session_matrix, lookup = _load_inputs(project_root)
    synthetic = frame.copy()
    synthetic["target"] = (indices % 2).astype(np.int8)
    train_mask, validation_mask = _fold_masks(synthetic, folds, 0)
    _, summary = _fit_fold(
        synthetic,
        session_matrix,
        lookup,
        train_mask,
        validation_mask,
        fold=0,
    )
    projected = float(summary["elapsed_seconds"]) * 5
    result = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "one_fold": summary,
        "projected_five_fold_seconds": projected,
        "peak_rss_bytes": _current_rss_bytes(),
        "proceed": bool(projected <= 2 * 3600 and _current_rss_bytes() < MAX_RSS_BYTES),
        "competition_outcomes_accessed": False,
    }
    paths = discover_project_paths(project_root)
    path = paths.cache_dir / "session_mastery_e740_benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["benchmark_path"] = str(path)
    result["benchmark_sha256"] = _sha256(path)
    return result


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    frame, indices, folds, session_matrix, lookup = _load_inputs(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features, base_similarity = prepare_bge_base_features(paths.cache_dir, full)
    _, dense_full, _ = prepare_semantic_features(paths.cache_dir, full)
    dense = dense_full[indices].copy()
    dense[:, -1:] = base_similarity[indices]
    baseline = semantic_logistic_oof(
        frame,
        folds,
        base_features[indices],
        dense,
        regularization_c=0.1,
        n_splits=5,
    )
    candidate, fit_summaries = session_mastery_oof(
        frame, folds, session_matrix, lookup
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
                    "pred_session_mastery": candidate,
                    "prediction": prediction,
                }
            )
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["log_loss", "roc_auc", "blend_weight"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = metrics.iloc[0].to_dict()
    selected_predictions = pd.concat(prediction_rows, ignore_index=True).loc[
        lambda value: value["blend_weight"].eq(selected["blend_weight"])
    ].reset_index(drop=True)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = folds == fold
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_session_mastery")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
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
        "competition_outcomes_accessed": True,
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
    parser = argparse.ArgumentParser(description="Run frozen E740 session mastery screen.")
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
