from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.bge_base_multiview_screen import (
    BATCH_SIZE,
    EMBEDDING_DIMENSION,
    MAX_RSS_BYTES,
    _encode,
    assert_runtime,
)
from trace_ace.io import discover_project_paths
from trace_ace.simulator_outcome_transfer import (
    LICENSE_SHA256,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    _collapse,
    _load_encoder_left,
    _sha256,
    verify_sources,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E650_simulatorarena_quality_v1"
TASK_LINE = "Task: represent the pedagogical quality of this tutoring interaction."
EXPECTED_ROWS = 450
FOLDS = 5
RIDGE_ALPHA = 10.0
MAX_PROJECTED_HOURS = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
CANONICAL_CONTENT_SHA256 = (
    "0d493508458104eebf05810762b704397052841b28e785bae5eac12fe12bc202"
)
CANONICAL_PARQUET_SHA256 = (
    "85b27544e041459e038ed6d302fec112c5d870873b450d0a05e6a656eed4c0de"
)
GATE_THRESHOLDS = {
    "pearson": 0.25,
    "spearman": 0.25,
    "rmse": 0.27,
    "rmse_gain": 0.01,
    "mae_gain": 0.01,
    "positive_fold_gains": 4,
    "worker_bootstrap_support": 0.90,
    "problem_bootstrap_support": 0.90,
}


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    return {
        "source": (
            paths.root
            / "Datasets"
            / "SimulatorArena"
            / "data"
            / "math_tutoring_annotations_redacted.json"
        ),
        "canonical": paths.cache_dir / "simulatorarena_e650_canonical.parquet",
        "canonical_metadata": (
            paths.cache_dir / "simulatorarena_e650_canonical.metadata.json"
        ),
        "benchmark": paths.cache_dir / "simulatorarena_e650_benchmark.json",
        "embeddings": paths.cache_dir / "simulatorarena_e650_embeddings.npy",
        "embedding_metadata": (
            paths.cache_dir / "simulatorarena_e650_embeddings.metadata.json"
        ),
        "progress": paths.cache_dir / "simulatorarena_e650_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def quality_dialogue_text(row: dict[str, object]) -> str:
    queries = row.get("user_queries")
    responses = row.get("ai_responses")
    if not isinstance(queries, list) or not isinstance(responses, list):
        raise ValueError("E650 dialogue arrays are missing.")
    released_turns = int(row.get("problem_1_turns", -1))
    available = min(len(queries), len(responses))
    turn_count = available if released_turns <= 0 else min(released_turns, available)
    if turn_count <= 0:
        raise ValueError("E650 first-problem dialogue is empty.")
    lines: list[str] = []
    for student, tutor in zip(queries[:turn_count], responses[:turn_count]):
        student_text = _collapse(student)
        tutor_text = _collapse(tutor)
        if student_text:
            lines.append(f"Student: {student_text}")
        if tutor_text:
            lines.append(f"Tutor: {tutor_text}")
    if not lines:
        raise ValueError("E650 first-problem dialogue has no usable text.")
    lines.append(TASK_LINE)
    return "\n".join(lines)


def fold_for_worker(worker_id: str) -> int:
    digest = hashlib.sha256(f"E650|{worker_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % FOLDS


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_index",
        "example_id",
        "worker_id",
        "problem_id",
        "fold",
        "target",
        "text",
    ]
    digest = hashlib.sha256()
    for values in frame[columns].itertuples(index=False, name=None):
        digest.update(
            json.dumps(
                list(values), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _fold_masks(
    frame: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = frame["fold"].eq(fold).to_numpy()
    valid_problems = set(frame.loc[valid, "problem_id"])
    train = ~valid & ~frame["problem_id"].isin(valid_problems).to_numpy()
    return train, valid


def _audit_folds(frame: pd.DataFrame) -> list[dict[str, int]]:
    audits: list[dict[str, int]] = []
    for fold in range(FOLDS):
        train, valid = _fold_masks(frame, fold)
        train_workers = set(frame.loc[train, "worker_id"])
        valid_workers = set(frame.loc[valid, "worker_id"])
        train_problems = set(frame.loc[train, "problem_id"])
        valid_problems = set(frame.loc[valid, "problem_id"])
        if train_workers & valid_workers or train_problems & valid_problems:
            raise ValueError(f"E650 fold {fold} leaks worker or problem.")
        audits.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "train_workers": len(train_workers),
                "validation_workers": len(valid_workers),
                "train_problems": len(train_problems),
                "validation_problems": len(valid_problems),
                "worker_overlap": 0,
                "problem_overlap": 0,
            }
        )
    return audits


def build_canonical(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    paths = _paths(project_root)
    rows = json.loads(paths["source"].read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("E650 source row count changed.")
    records: list[dict[str, object]] = []
    for source_index, row in enumerate(rows):
        rating = int(row["overall_rating"])
        if not 1 <= rating <= 10:
            raise ValueError(f"E650 invalid released rating: {rating}")
        worker_id = str(row["workerId"])
        problem_id = int(row["problem_id"])
        identity = "|".join(
            [
                "E650",
                str(source_index),
                worker_id,
                str(problem_id),
                str(row.get("model", "")),
                str(row.get("user_id", "")),
            ]
        )
        records.append(
            {
                "source_index": source_index,
                "example_id": hashlib.sha256(identity.encode()).hexdigest(),
                "worker_id": worker_id,
                "problem_id": problem_id,
                "fold": fold_for_worker(worker_id),
                "target": (rating - 1) / 9.0,
                "text": quality_dialogue_text(row),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_ROWS or frame["example_id"].duplicated().any():
        raise ValueError("E650 canonical identity audit failed.")
    fold_audit = _audit_folds(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_sha256": LICENSE_SHA256,
        "rows": len(frame),
        "target_mean": float(frame["target"].mean()),
        "target_std": float(frame["target"].std(ddof=0)),
        "fold_audit": fold_audit,
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "rating_input": False,
        "final_answer_or_solution_input": False,
        "correctness_or_self_report_input": False,
        "second_problem_input": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["canonical_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_canonical(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    if not CANONICAL_CONTENT_SHA256 or not CANONICAL_PARQUET_SHA256:
        raise RuntimeError("E650 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E650 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E650 canonical ordered-content SHA-256 changed.")
    _audit_folds(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"], columns=["example_id", "text"])
    order = frame["example_id"].map(
        lambda value: hashlib.sha256(f"E650-benchmark|{value}".encode()).hexdigest()
    )
    sample = frame.assign(_order=order).sort_values("_order").head(32)
    model = _load_encoder_left(project_root)
    started = time.perf_counter()
    matrix = _encode(model, sample["text"].tolist())
    elapsed = time.perf_counter() - started
    projected_seconds = elapsed / len(sample) * len(frame)
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_rows": len(sample),
        "benchmark_batches": math.ceil(len(sample) / BATCH_SIZE),
        "elapsed_seconds": elapsed,
        "projected_rows": len(frame),
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600,
        "peak_rss_bytes": peak_rss,
        "shape": list(matrix.shape),
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "runtime": runtime,
        "outcome_labels_accessed": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_external_cache(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    benchmark_result = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    if benchmark_result.get("proceed") is not True:
        raise ValueError("E650 benchmark did not authorize cache construction.")
    frame = pd.read_parquet(paths["canonical"], columns=["text"])
    model = _load_encoder_left(project_root)
    matrix = np.lib.format.open_memmap(
        paths["embeddings"],
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), EMBEDDING_DIMENSION),
    )
    started = time.perf_counter()
    for start in range(0, len(frame), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(frame))
        matrix[start:stop] = _encode(model, frame.iloc[start:stop]["text"].tolist())
        matrix.flush()
        paths["progress"].write_text(
            json.dumps(
                {
                    "completed_rows": stop,
                    "total_rows": len(frame),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if stop % 128 == 0 or stop == len(frame):
            print(f"e650_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0)
    ):
        raise ValueError("E650 embedding cache audit failed.")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "shape": list(loaded.shape),
        "elapsed_seconds": time.perf_counter() - started,
        "cache_sha256": _sha256(paths["embeddings"]),
        "runtime": runtime,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["embedding_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _regression_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    return {
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "mae": float(mean_absolute_error(target, prediction)),
        "pearson": float(pearsonr(target, prediction).statistic),
        "spearman": float(spearmanr(target, prediction).statistic),
        "prediction_mean": float(prediction.mean()),
    }


def _group_bootstrap(
    gains: np.ndarray, groups: Sequence[object], salt: int
) -> dict[str, object]:
    group_values = np.asarray([str(value) for value in groups])
    unique = np.unique(group_values)
    grouped = np.array(
        [gains[group_values == group].mean() for group in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + salt)
    draws = grouped[
        rng.integers(
            0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped))
        )
    ].mean(axis=1)
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "groups": len(unique),
        "mean_squared_error_gain": float(draws.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "support": float(np.mean(draws > 0)),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import Ridge

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(
        paths["embedding_metadata"].read_text(encoding="utf-8")
    )
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E650 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    target = frame["target"].to_numpy(dtype=np.float64)
    prediction = np.full(len(frame), np.nan)
    prior = np.full(len(frame), np.nan)
    fold_rows: list[dict[str, object]] = []
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    for fold in range(FOLDS):
        train, valid = _fold_masks(frame, fold)
        model = Ridge(
            alpha=RIDGE_ALPHA,
            fit_intercept=True,
            solver="lsqr",
            max_iter=1000,
            tol=1e-6,
        )
        model.fit(embeddings[train], target[train])
        prediction[valid] = np.clip(model.predict(embeddings[valid]), 0.0, 1.0)
        prior[valid] = float(target[train].mean())
        candidate_metrics = _regression_metrics(target[valid], prediction[valid])
        prior_rmse = float(np.mean((target[valid] - prior[valid]) ** 2) ** 0.5)
        prior_mae = float(np.mean(np.abs(target[valid] - prior[valid])))
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "candidate": candidate_metrics,
                "prior_rmse": prior_rmse,
                "prior_mae": prior_mae,
                "rmse_gain": prior_rmse - candidate_metrics["rmse"],
            }
        )
        coefficients.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(float(model.intercept_))
    if not np.isfinite(prediction).all() or not np.isfinite(prior).all():
        raise ValueError("E650 OOF predictions are incomplete.")
    metrics = _regression_metrics(target, prediction)
    prior_rmse = float(np.mean((target - prior) ** 2) ** 0.5)
    prior_mae = float(np.mean(np.abs(target - prior)))
    rmse_gain = prior_rmse - metrics["rmse"]
    mae_gain = prior_mae - metrics["mae"]
    squared_error_gain = (target - prior) ** 2 - (target - prediction) ** 2
    worker_bootstrap = _group_bootstrap(
        squared_error_gain, frame["worker_id"], salt=1
    )
    problem_bootstrap = _group_bootstrap(
        squared_error_gain, frame["problem_id"], salt=2
    )
    positive_fold_gains = sum(
        float(row["rmse_gain"]) > 0 for row in fold_rows
    )
    clauses = {
        "pearson": metrics["pearson"] >= GATE_THRESHOLDS["pearson"],
        "spearman": metrics["spearman"] >= GATE_THRESHOLDS["spearman"],
        "rmse": metrics["rmse"] <= GATE_THRESHOLDS["rmse"],
        "rmse_gain": rmse_gain >= GATE_THRESHOLDS["rmse_gain"],
        "mae_gain": mae_gain >= GATE_THRESHOLDS["mae_gain"],
        "positive_fold_gains": (
            positive_fold_gains
            >= GATE_THRESHOLDS["positive_fold_gains"]
        ),
        "worker_bootstrap_support": (
            worker_bootstrap["support"]
            >= GATE_THRESHOLDS["worker_bootstrap_support"]
        ),
        "problem_bootstrap_support": (
            problem_bootstrap["support"]
            >= GATE_THRESHOLDS["problem_bootstrap_support"]
        ),
    }
    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_simulator_quality_transfer"
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = frame[
        [
            "example_id",
            "source_index",
            "worker_id",
            "problem_id",
            "fold",
            "target",
        ]
    ].copy()
    predictions["prediction"] = prediction
    predictions["prior_prediction"] = prior
    predictions_path = run_dir / "oof_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    model_path = run_dir / "fold_models.npz"
    np.savez_compressed(
        model_path,
        coefficients=np.vstack(coefficients),
        intercepts=np.asarray(intercepts),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "rows": len(frame),
        "metrics": metrics,
        "prior_rmse": prior_rmse,
        "prior_mae": prior_mae,
        "rmse_gain_vs_prior": rmse_gain,
        "mae_gain_vs_prior": mae_gain,
        "positive_fold_gains": positive_fold_gains,
        "folds": fold_rows,
        "worker_bootstrap": worker_bootstrap,
        "problem_bootstrap": problem_bootstrap,
        "thresholds": GATE_THRESHOLDS,
        "clauses": clauses,
        "passes_external_gate": all(clauses.values()),
        "runtime": runtime,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "canonical_content_sha256": CANONICAL_CONTENT_SHA256,
        "canonical_parquet_sha256": CANONICAL_PARQUET_SHA256,
        "embedding_sha256": embedding_hash,
        "predictions_sha256": _sha256(predictions_path),
        "fold_models_sha256": _sha256(model_path),
        "competition_cache_built": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["report_path"] = str(report_path)
    result["report_sha256"] = _sha256(report_path)
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E650 SimulatorArena quality transfer."
    )
    parser.add_argument(
        "stage",
        choices=("canonicalize", "benchmark", "build-cache", "validate"),
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "canonicalize":
        result = build_canonical(args.project_root)
    elif args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build-cache":
        result = build_external_cache(args.project_root)
    else:
        result = validate_external(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
