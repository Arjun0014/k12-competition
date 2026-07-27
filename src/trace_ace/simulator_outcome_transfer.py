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
    MODEL_DIRECTORY,
    MODEL_HASHES,
    _configure_torch,
    _encode,
    _load_encoder,
    assert_runtime,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E640_simulatorarena_real_outcome_v1"
SOURCE_COMMIT = "e9f677c4975496fdd37f28bb6343ec3c1c54c8b4"
SOURCE_SHA256 = "b2909d037da14ebfafd72e66ad8985ab700f0f164bc5d94713ca6fc43aaf651a"
LICENSE_SHA256 = "9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b"
CANONICAL_CONTENT_SHA256 = (
    "950c0341583c6721b2e51fdca28c53d2a00d33d2a3d46d5ce2122c05671cf447"
)
CANONICAL_PARQUET_SHA256 = (
    "e743a05e140b10f154ad1f8e2d8d431d90cade9883c9e8e8f4c4988d55d57f60"
)
EXPECTED_SOURCE_ROWS = 450
EXPECTED_USABLE_ROWS = 449
EXPECTED_LABEL_COUNTS = {0: 153, 1: 296}
FOLDS = 5
REGULARIZATION_C = 0.1
MAX_SEQUENCE_LENGTH = 256
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
MAX_PROJECTED_HOURS = 1.0
TASK_LINE = (
    "Task: predict whether the student will answer the held-out assessment "
    "correctly after tutoring."
)
GATE_THRESHOLDS = {
    "roc_auc": 0.60,
    "macro_f1": 0.55,
    "log_loss": 0.65,
    "log_loss_gain": 0.02,
    "brier_gain": 0.008,
    "ece_10": 0.15,
    "positive_fold_gains": 4,
    "worker_bootstrap_support": 0.90,
    "problem_bootstrap_support": 0.90,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "SimulatorArena"
    model_root = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "source": source_root / "data" / "math_tutoring_annotations_redacted.json",
        "license": source_root / "LICENSE",
        "model": model_root / "model.safetensors",
        "config": model_root / "config.json",
        "modules": model_root / "modules.json",
        "pooling": model_root / "1_Pooling" / "config.json",
        "canonical": paths.cache_dir / "simulatorarena_e640_canonical.parquet",
        "canonical_metadata": (
            paths.cache_dir / "simulatorarena_e640_canonical.metadata.json"
        ),
        "benchmark": paths.cache_dir / "simulatorarena_e640_benchmark.json",
        "embeddings": paths.cache_dir / "simulatorarena_e640_embeddings.npy",
        "embedding_metadata": (
            paths.cache_dir / "simulatorarena_e640_embeddings.metadata.json"
        ),
        "progress": paths.cache_dir / "simulatorarena_e640_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    expected = {
        "source": SOURCE_SHA256,
        "license": LICENSE_SHA256,
        **MODEL_HASHES,
    }
    for name, digest in expected.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing E640 {name}: {path}")
        observed = _sha256(path)
        if observed != digest:
            raise ValueError(f"E640 {name} SHA-256 changed: {observed}")
    return expected


def _collapse(value: object) -> str:
    return " ".join(str(value).split())


def canonical_dialogue_text(row: dict[str, object]) -> str:
    queries = row.get("user_queries")
    responses = row.get("ai_responses")
    if not isinstance(queries, list) or not isinstance(responses, list):
        raise ValueError("E640 dialogue arrays are missing.")
    released_turns = int(row.get("problem_1_turns", -1))
    available = min(len(queries), len(responses))
    turn_count = available if released_turns <= 0 else min(released_turns, available)
    if turn_count <= 0:
        raise ValueError("E640 first-problem dialogue is empty.")
    lines: list[str] = []
    for student, tutor in zip(queries[:turn_count], responses[:turn_count]):
        student_text = _collapse(student)
        tutor_text = _collapse(tutor)
        if student_text:
            lines.append(f"Student: {student_text}")
        if tutor_text:
            lines.append(f"Tutor: {tutor_text}")
    if not lines:
        raise ValueError("E640 first-problem dialogue has no usable text.")
    lines.append(TASK_LINE)
    return "\n".join(lines)


def fold_for_worker(worker_id: str) -> int:
    digest = hashlib.sha256(f"E640|{worker_id}".encode("utf-8")).hexdigest()
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
        payload = json.dumps(
            list(values), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _audit_folds(frame: pd.DataFrame) -> list[dict[str, int]]:
    audits: list[dict[str, int]] = []
    for fold in range(FOLDS):
        valid = frame["fold"].eq(fold)
        valid_workers = set(frame.loc[valid, "worker_id"])
        valid_problems = set(frame.loc[valid, "problem_id"])
        train = ~valid & ~frame["problem_id"].isin(valid_problems)
        train_workers = set(frame.loc[train, "worker_id"])
        train_problems = set(frame.loc[train, "problem_id"])
        if valid_workers & train_workers or valid_problems & train_problems:
            raise ValueError(f"E640 fold {fold} leaks worker or problem identity.")
        if set(frame.loc[train, "target"]) != {0, 1}:
            raise ValueError(f"E640 fold {fold} training partition lacks a class.")
        if set(frame.loc[valid, "target"]) != {0, 1}:
            raise ValueError(f"E640 fold {fold} validation partition lacks a class.")
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
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError("E640 source row count changed.")
    records: list[dict[str, object]] = []
    released_counts = {"correct": 0, "incorrect": 0, "unknown": 0}
    for source_index, row in enumerate(rows):
        label = str(row.get("problem_1_correctness", ""))
        released_counts[label] = released_counts.get(label, 0) + 1
        if label not in {"correct", "incorrect"}:
            continue
        worker_id = str(row["workerId"])
        problem_id = int(row["problem_id"])
        identity = "|".join(
            [
                "E640",
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
                "example_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "worker_id": worker_id,
                "problem_id": problem_id,
                "fold": fold_for_worker(worker_id),
                "target": int(label == "correct"),
                "text": canonical_dialogue_text(row),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_USABLE_ROWS or frame["example_id"].duplicated().any():
        raise ValueError("E640 canonical row identity audit failed.")
    counts = frame["target"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"E640 label counts changed: {counts}")
    if released_counts != {"correct": 296, "incorrect": 153, "unknown": 1}:
        raise ValueError(f"E640 released labels changed: {released_counts}")
    fold_audit = _audit_folds(frame)
    frame.to_parquet(paths["canonical"], index=False)
    content_hash = _ordered_content_hash(frame)
    parquet_hash = _sha256(paths["canonical"])
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_sha256": LICENSE_SHA256,
        "source_rows": len(rows),
        "usable_rows": len(frame),
        "label_counts": {str(key): value for key, value in counts.items()},
        "released_label_counts": released_counts,
        "fold_audit": fold_audit,
        "canonical_content_sha256": content_hash,
        "canonical_parquet_sha256": parquet_hash,
        "input_contract": "first_problem_dialogue_only",
        "final_answer_or_solution_input": False,
        "self_report_or_rating_input": False,
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
        raise RuntimeError("E640 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E640 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    observed_content = _ordered_content_hash(frame)
    if observed_content != CANONICAL_CONTENT_SHA256:
        raise ValueError("E640 canonical ordered-content SHA-256 changed.")
    _audit_folds(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def _load_encoder_left(project_root: str | Path):
    _configure_torch()
    model = _load_encoder(project_root)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model.tokenizer.truncation_side = "left"
    if model.tokenizer.truncation_side != "left":
        raise ValueError("E640 tokenizer did not retain left truncation.")
    return model


def _fixed_benchmark_rows(frame: pd.DataFrame, count: int = 32) -> pd.DataFrame:
    order = frame["example_id"].map(
        lambda value: hashlib.sha256(f"E640-benchmark|{value}".encode()).hexdigest()
    )
    return frame.assign(_order=order).sort_values("_order").head(count)


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"], columns=["example_id", "text"])
    sample = _fixed_benchmark_rows(frame)
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
        "seconds_per_row": elapsed / len(sample),
        "projected_rows": len(frame),
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600,
        "peak_rss_bytes": peak_rss,
        "embedding_shape": list(matrix.shape),
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "runtime": runtime,
        "source_sha256": SOURCE_SHA256,
        "canonical_content_sha256": CANONICAL_CONTENT_SHA256,
        "canonical_parquet_sha256": CANONICAL_PARQUET_SHA256,
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
        raise ValueError("E640 benchmark did not authorize cache construction.")
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
            print(f"e640_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0)
    ):
        raise ValueError("E640 embedding cache audit failed.")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "shape": list(loaded.shape),
        "elapsed_seconds": time.perf_counter() - started,
        "cache_sha256": _sha256(paths["embeddings"]),
        "runtime": runtime,
        "source_sha256": SOURCE_SHA256,
        "canonical_content_sha256": CANONICAL_CONTENT_SHA256,
        "canonical_parquet_sha256": CANONICAL_PARQUET_SHA256,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["embedding_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _classification_metrics(
    target: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    metrics = binary_metrics(target, probability)
    prediction = probability >= 0.5
    metrics["accuracy"] = float(accuracy_score(target, prediction))
    metrics["macro_f1"] = float(f1_score(target, prediction, average="macro"))
    return metrics


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
        "mean_gain": float(draws.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "support": float(np.mean(draws > 0)),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(
        paths["embedding_metadata"].read_text(encoding="utf-8")
    )
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E640 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    target = frame["target"].to_numpy(dtype=np.int8)
    probability = np.full(len(frame), np.nan, dtype=np.float64)
    prior_probability = np.full(len(frame), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    coefficients: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []
    for fold in range(FOLDS):
        valid = frame["fold"].eq(fold).to_numpy()
        valid_problems = set(frame.loc[valid, "problem_id"])
        train = (
            ~valid
            & ~frame["problem_id"].isin(valid_problems).to_numpy()
        )
        train_workers = set(frame.loc[train, "worker_id"])
        valid_workers = set(frame.loc[valid, "worker_id"])
        if train_workers & valid_workers:
            raise ValueError(f"E640 fold {fold} worker leakage.")
        if set(frame.loc[train, "problem_id"]) & valid_problems:
            raise ValueError(f"E640 fold {fold} problem leakage.")
        model = LogisticRegression(
            C=REGULARIZATION_C,
            solver="lbfgs",
            max_iter=1000,
            random_state=BOOTSTRAP_SEED,
        )
        model.fit(embeddings[train], target[train])
        probability[valid] = model.predict_proba(embeddings[valid])[:, 1]
        prior_probability[valid] = float(target[train].mean())
        candidate_metrics = _classification_metrics(
            target[valid], probability[valid]
        )
        prior_metrics = _classification_metrics(
            target[valid], prior_probability[valid]
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "candidate": candidate_metrics,
                "prior": prior_metrics,
                "log_loss_gain": (
                    prior_metrics["log_loss"] - candidate_metrics["log_loss"]
                ),
            }
        )
        coefficients.append(model.coef_.astype(np.float64))
        intercepts.append(model.intercept_.astype(np.float64))
    if not np.isfinite(probability).all() or not np.isfinite(
        prior_probability
    ).all():
        raise ValueError("E640 OOF predictions are incomplete.")
    metrics = _classification_metrics(target, probability)
    prior_metrics = _classification_metrics(target, prior_probability)
    gains = (
        -np.log(
            np.where(target == 1, prior_probability, 1.0 - prior_probability)
        )
        + np.log(np.where(target == 1, probability, 1.0 - probability))
    )
    worker_bootstrap = _group_bootstrap(gains, frame["worker_id"], salt=1)
    problem_bootstrap = _group_bootstrap(gains, frame["problem_id"], salt=2)
    positive_fold_gains = sum(
        float(row["log_loss_gain"]) > 0 for row in fold_rows
    )
    log_loss_gain = prior_metrics["log_loss"] - metrics["log_loss"]
    brier_gain = prior_metrics["brier_score"] - metrics["brier_score"]
    clauses = {
        "roc_auc": metrics["roc_auc"] >= GATE_THRESHOLDS["roc_auc"],
        "macro_f1": metrics["macro_f1"] >= GATE_THRESHOLDS["macro_f1"],
        "log_loss": metrics["log_loss"] <= GATE_THRESHOLDS["log_loss"],
        "log_loss_gain": (
            log_loss_gain >= GATE_THRESHOLDS["log_loss_gain"]
        ),
        "brier_gain": brier_gain >= GATE_THRESHOLDS["brier_gain"],
        "ece_10": metrics["ece_10"] <= GATE_THRESHOLDS["ece_10"],
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
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_simulator_outcome_transfer"
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
    predictions["probability"] = probability
    predictions["prior_probability"] = prior_probability
    predictions_path = run_dir / "oof_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    model_path = run_dir / "fold_models.npz"
    np.savez_compressed(
        model_path,
        coefficients=np.vstack(coefficients),
        intercepts=np.concatenate(intercepts),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "rows": len(frame),
        "metrics": metrics,
        "prior_metrics": prior_metrics,
        "log_loss_gain_vs_prior": log_loss_gain,
        "brier_gain_vs_prior": brier_gain,
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
        description="Run the frozen E640 SimulatorArena real-outcome transfer."
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
