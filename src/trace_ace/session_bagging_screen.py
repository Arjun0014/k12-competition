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

from trace_ace.bge_base_dual_pooling import _load_pilot, verify_sources
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


PROTOCOL_ID = "E600_session_bagged_bge_v1"
PILOT_PROTOCOL = "semantic_k50_s0"
PILOT_ROWS = 4_096
N_FOLDS = 5
N_BAGS = 5
DENSE_SCALE = 0.08
REGULARIZATION_C = 0.1
BOOTSTRAP_REPLICATES = 5_000
MAX_PROJECTED_HOURS = 2.0
MAX_RSS_BYTES = 8 * 1024**3
SEED_BASE = 20260727


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_path(project_root: str | Path) -> Path:
    return (
        discover_project_paths(project_root).cache_dir
        / "session_bagging_e600_benchmark.json"
    )


def session_slice(session_id: object) -> int:
    digest = hashlib.sha256(f"E600|{session_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % N_BAGS


def assign_session_slices(frame: pd.DataFrame) -> np.ndarray:
    if "session_id" not in frame:
        raise ValueError("E600 requires session_id.")
    values = frame["session_id"].astype(str)
    mapping = {value: session_slice(value) for value in values.unique()}
    slices = values.map(mapping).to_numpy(dtype=np.int8)
    if slices.shape != (len(frame),) or np.any((slices < 0) | (slices >= N_BAGS)):
        raise ValueError("E600 session-slice assignment is invalid.")
    audit = pd.DataFrame({"session_id": values, "slice": slices}).groupby(
        "session_id", sort=False
    )["slice"].nunique()
    if not audit.eq(1).all():
        raise ValueError("E600 assigned one session to multiple slices.")
    return slices


def _load_features(
    project_root: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame, _, indices, folds = _load_pilot(project_root)
    paths = discover_project_paths(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    semantic_full, similarity_full = prepare_bge_base_features(
        paths.cache_dir, full
    )
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    semantic = semantic_full[indices]
    dense = dense_full[indices].copy()
    dense[:, -1:] = similarity_full[indices]
    if (
        semantic.shape != (PILOT_ROWS, 4 * 768)
        or dense.shape != (PILOT_ROWS, 35)
        or len(dense_names) != 35
        or not np.isfinite(semantic).all()
        or not np.isfinite(dense).all()
    ):
        raise ValueError("E600 pilot feature contract changed.")
    return frame, indices, folds, semantic, dense


def _fit_one_bag(
    semantic: np.ndarray,
    dense: np.ndarray,
    target: np.ndarray,
    retained_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if (
        semantic.ndim != 2
        or dense.ndim != 2
        or len(semantic) != len(dense)
        or target.shape != (len(semantic),)
        or retained_mask.shape != target.shape
        or validation_mask.shape != target.shape
        or np.any(retained_mask & validation_mask)
        or set(np.unique(target[retained_mask]).tolist()) != {0, 1}
    ):
        raise ValueError("E600 bag inputs violate the frozen contract.")
    scaler = StandardScaler()
    dense_train = scaler.fit_transform(dense[retained_mask]).astype(np.float32)
    dense_validation = scaler.transform(dense[validation_mask]).astype(np.float32)
    x_train = np.column_stack(
        [semantic[retained_mask], dense_train * np.float32(DENSE_SCALE)]
    )
    x_validation = np.column_stack(
        [semantic[validation_mask], dense_validation * np.float32(DENSE_SCALE)]
    )
    model = LogisticRegression(
        C=REGULARIZATION_C,
        solver="lbfgs",
        max_iter=400,
        tol=1e-5,
        random_state=seed,
    )
    model.fit(x_train, target[retained_mask])
    prediction = model.predict_proba(x_validation)[:, 1]
    if (
        prediction.shape != (int(validation_mask.sum()),)
        or not np.isfinite(prediction).all()
        or np.any((prediction <= 0) | (prediction >= 1))
    ):
        raise ValueError("E600 bag prediction audit failed.")
    return prediction.astype(np.float64)


def bagged_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    semantic: np.ndarray,
    dense: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    slices = assign_session_slices(frame)
    summaries: list[dict[str, object]] = []
    for fold in range(N_FOLDS):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        fold_predictions: list[np.ndarray] = []
        for bag in range(N_BAGS):
            retained_mask = train_mask & (slices != bag)
            bag_prediction = _fit_one_bag(
                semantic,
                dense,
                target,
                retained_mask,
                validation_mask,
                seed=SEED_BASE + fold * N_BAGS + bag,
            )
            fold_predictions.append(bag_prediction)
            summaries.append(
                {
                    "fold": fold,
                    "excluded_slice": bag,
                    "legal_outer_training_rows": int(train_mask.sum()),
                    "retained_rows": int(retained_mask.sum()),
                    "retained_fraction": float(
                        retained_mask.sum() / train_mask.sum()
                    ),
                    "validation_rows": int(validation_mask.sum()),
                    "retained_sessions": int(
                        frame.loc[retained_mask, "session_id"].nunique()
                    ),
                }
            )
        prediction[validation_mask] = np.mean(
            np.vstack(fold_predictions), axis=0
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("E600 OOF predictions are incomplete.")
    return prediction, summaries


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    frame, indices, folds, semantic, dense = _load_features(project_root)
    train_mask, validation_mask = _fold_masks(frame, folds, 0)
    slices = assign_session_slices(frame)
    synthetic_target = (indices % 2).astype(np.int8)
    started = time.perf_counter()
    retained_rows: list[int] = []
    for bag in range(N_BAGS):
        retained_mask = train_mask & (slices != bag)
        _fit_one_bag(
            semantic,
            dense,
            synthetic_target,
            retained_mask,
            validation_mask,
            seed=SEED_BASE + bag,
        )
        retained_rows.append(int(retained_mask.sum()))
    elapsed = time.perf_counter() - started
    projected_seconds = elapsed * N_FOLDS
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "one outer fold and all five leave-slice-out estimators",
        "elapsed_seconds": elapsed,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "retained_rows": retained_rows,
        "frozen_folds": N_FOLDS,
        "frozen_bags": N_BAGS,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "outcome_labels_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _benchmark_path(project_root).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    path = _benchmark_path(project_root)
    if not path.exists():
        raise FileNotFoundError("Run the E600 benchmark before validation.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol_id") != PROTOCOL_ID
        or result.get("source_hashes") != verify_sources(project_root)
        or result.get("proceed") is not True
        or result.get("outcome_labels_accessed") is not False
    ):
        raise ValueError("E600 benchmark binding or gate is invalid.")
    return result


def _fold_metrics(
    frame: pd.DataFrame,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> pd.DataFrame:
    target = frame["target"].to_numpy(dtype=np.int8)
    rows: list[dict[str, object]] = []
    for fold in range(N_FOLDS):
        mask = folds == fold
        baseline_metrics = binary_metrics(target[mask], baseline[mask])
        candidate_metrics = binary_metrics(target[mask], candidate[mask])
        rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                **{
                    f"bge_{key}": value
                    for key, value in baseline_metrics.items()
                },
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                **{
                    f"delta_{key}_vs_bge": (
                        candidate_metrics[key] - baseline_metrics[key]
                    )
                    for key in (
                        "log_loss",
                        "roc_auc",
                        "brier_score",
                        "ece_10",
                        "prediction_mean",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    benchmark_result = _load_benchmark(project_root)
    frame, _, folds, semantic, dense = _load_features(project_root)
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline = semantic_logistic_oof(
        frame,
        folds,
        semantic,
        dense,
        regularization_c=REGULARIZATION_C,
        n_splits=N_FOLDS,
    )
    candidate, bag_summaries = bagged_oof(
        frame, folds, semantic, dense, target
    )
    baseline_metrics = binary_metrics(target, baseline)
    candidate_metrics = binary_metrics(target, candidate)
    deltas = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in (
            "log_loss",
            "roc_auc",
            "brier_score",
            "ece_10",
            "prediction_mean",
        )
    }
    bootstrap = _session_bootstrap(
        frame,
        baseline,
        candidate,
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    fold_metrics = _fold_metrics(frame, folds, baseline, candidate)
    loss_path = bool(
        deltas["log_loss"] <= -0.0015
        and deltas["roc_auc"] >= 0
        and deltas["brier_score"] <= 0
        and deltas["ece_10"] <= 0
    )
    auc_path = bool(
        deltas["roc_auc"] >= 0.005
        and deltas["log_loss"] <= 0
        and deltas["brier_score"] <= 0
        and deltas["ece_10"] <= 0
    )
    passes = bool(
        (loss_path or auc_path)
        and bootstrap["support_positive_log_loss_gain"] >= 0.90
    )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_session_bagging")
    paths = discover_project_paths(project_root)
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "oof_predictions.parquet"
    fold_path = run_dir / "fold_metrics.csv"
    bootstrap_path = run_dir / "bootstrap.csv"
    summary_path = run_dir / "bag_summaries.csv"
    pd.DataFrame(
        [
            {
                **{f"bge_{key}": value for key, value in baseline_metrics.items()},
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                **{f"delta_{key}_vs_bge": value for key, value in deltas.items()},
            }
        ]
    ).to_csv(metrics_path, index=False, lineterminator="\n")
    pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "session_id": frame["session_id"],
            "learning_objective_id": frame["learning_objective_id"],
            "fold": folds,
            "target": target,
            "pred_bge_base": baseline,
            "pred_session_bagged": candidate,
        }
    ).to_parquet(predictions_path, index=False)
    fold_metrics.to_csv(fold_path, index=False, lineterminator="\n")
    pd.DataFrame([bootstrap]).to_csv(
        bootstrap_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(bag_summaries).to_csv(
        summary_path, index=False, lineterminator="\n"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E600_session_bagged_bge",
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "configuration": {
            "outer_folds": N_FOLDS,
            "bags_per_fold": N_BAGS,
            "slice_rule": 'int(SHA256("E600|" + session_id)[:16],16) mod 5',
            "excluded_slices_per_fold": list(range(N_BAGS)),
            "equal_prediction_weights": [1 / N_BAGS] * N_BAGS,
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "dense_scale": DENSE_SCALE,
            "class_weight": None,
            "calibration": None,
            "blend": None,
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas_candidate_minus_bge": deltas,
        "fold_log_loss_changes": fold_metrics[
            "delta_log_loss_vs_bge"
        ].tolist(),
        "paired_session_bootstrap": bootstrap,
        "screen_clauses": {
            "loss_path": loss_path,
            "auc_path": auc_path,
            "session_bootstrap_support_at_least_0_90": (
                bootstrap["support_positive_log_loss_gain"] >= 0.90
            ),
        },
        "passes_screen": passes,
        "bag_summaries": bag_summaries,
        "benchmark": benchmark_result,
        "peak_rss_bytes": _current_rss_bytes(),
        "source_hashes": source_hashes,
        "runtime": runtime,
        "validation_runtime_seconds": time.perf_counter() - started,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        "metrics": _sha256(metrics_path),
        "predictions": _sha256(predictions_path),
        "fold_metrics": _sha256(fold_path),
        "bootstrap": _sha256(bootstrap_path),
        "bag_summaries": _sha256(summary_path),
    }
    report["artifact_sha256"] = artifact_hashes
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report": report,
        "report_sha256": _sha256(report_path),
        "passes_screen": passes,
        "artifact_sha256": artifact_hashes,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E600 session-bagged BGE screen."
    )
    parser.add_argument("stage", choices=("benchmark", "validate", "pipeline"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "validate":
        result = validate(args.project_root)
    else:
        benchmark_result = benchmark(args.project_root)
        if not benchmark_result["proceed"]:
            raise RuntimeError("E600 benchmark failed; validation is prohibited.")
        result = {
            "benchmark": benchmark_result,
            "validation": validate(args.project_root),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
