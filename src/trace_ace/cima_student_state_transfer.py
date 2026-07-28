from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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
from trace_ace.metrics import binary_metrics
from trace_ace.simulator_outcome_transfer import (
    _collapse,
    _load_encoder_left,
    _sha256,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E680_cima_student_guess_v1"
SOURCE_COMMIT = "3fa48593c001046893f5f1320ab031f7237e7998"
SOURCE_SHA256 = "544dfc50dd05b14579e1a04b4751f4ffda8a72d3d398410d1aba0bcc2dd57405"
LICENSE_SHA256 = "23680d5c37de5b69436172372f14f88ddc31376a27738256d07bcaa78642897b"
EXPECTED_ROWS = 1_135
EXPECTED_TARGET_COUNTS = {0: 621, 1: 514}
FOLDS = 5
REGULARIZATION_C = 0.1
MAX_PROJECTED_HOURS = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
TASK_LINE = (
    "Task: represent whether the final student turn is an independent answer "
    "attempt rather than help-seeking or acknowledgment."
)
CANONICAL_CONTENT_SHA256 = (
    "2bef00ad2bb18eba376fcb70501a445d681395ccd3d76223e4a509bfd96c5a8a"
)
CANONICAL_PARQUET_SHA256 = (
    "5c97d1bbec9302bedebbdda6d264ffa42ae574ef89b2aac1f0191d50ffe508c7"
)
GATE_THRESHOLDS = {
    "roc_auc": 0.70,
    "macro_f1": 0.65,
    "log_loss_gain": 0.030,
    "brier_gain": 0.010,
    "ece_10": 0.10,
    "bootstrap_support": 0.95,
}


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "CIMA"
    return {
        "source_root": source_root,
        "source": source_root / "dataset.json",
        "license": source_root / "README.md",
        "canonical": paths.cache_dir / "cima_e680_canonical.parquet",
        "canonical_metadata": paths.cache_dir / "cima_e680_canonical.metadata.json",
        "benchmark": paths.cache_dir / "cima_e680_benchmark.json",
        "embeddings": paths.cache_dir / "cima_e680_embeddings.npy",
        "embedding_metadata": paths.cache_dir
        / "cima_e680_embeddings.metadata.json",
        "progress": paths.cache_dir / "cima_e680_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    if _sha256(paths["source"]) != SOURCE_SHA256:
        raise ValueError("E680 source JSON SHA-256 changed.")
    if _sha256(paths["license"]) != LICENSE_SHA256:
        raise ValueError("E680 CC BY dataset-card SHA-256 changed.")
    commit = subprocess.run(
        ["git", "-C", str(paths["source_root"]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"E680 source commit changed: {commit}")
    return {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_sha256": LICENSE_SHA256,
    }


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized not in {"true", "false"}:
        raise ValueError(f"E680 action flag changed: {value!r}")
    return normalized == "true"


def canonical_dialogue_text(history: object) -> str:
    if not isinstance(history, list) or len(history) not in {2, 4, 6, 8, 10}:
        raise ValueError("E680 history length changed.")
    lines: list[str] = []
    for index, value in enumerate(history):
        text = _collapse(value)
        if not text:
            raise ValueError("E680 history contains an empty turn.")
        role = "Tutor" if index % 2 == 0 else "Student"
        lines.append(f"{role}: {text}")
    if not lines[-1].startswith("Student:"):
        raise ValueError("E680 history no longer ends with the student.")
    lines.append(TASK_LINE)
    return "\n".join(lines)


def concept_key(row: dict[str, object]) -> str:
    values = [str(row[name]) for name in ("engPrep", "engObj", "engColor")]
    if any(not value.strip() for value in values):
        raise ValueError("E680 concept triple contains an empty value.")
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def group_fold(protocol: str, value: str) -> int:
    if protocol == "exercise_disjoint":
        prefix = "exercise"
    elif protocol == "concept_disjoint":
        prefix = "concept"
    else:
        raise ValueError(f"Unknown E680 protocol: {protocol}")
    digest = hashlib.sha256(
        f"E680|{prefix}|{value}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % FOLDS


def protocol_splits(
    frame: pd.DataFrame, protocol: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if protocol == "exercise_disjoint":
        column = "exercise_fold"
    elif protocol == "concept_disjoint":
        column = "concept_fold"
    else:
        raise ValueError(f"Unknown E680 protocol: {protocol}")
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for fold in range(FOLDS):
        valid = frame[column].eq(fold).to_numpy()
        train = ~valid
        if (
            not train.any()
            or not valid.any()
            or set(frame.loc[train, "target"]) != {0, 1}
            or set(frame.loc[valid, "target"]) != {0, 1}
        ):
            raise ValueError(f"E680 {protocol} fold {fold} is invalid.")
        result.append((str(fold), train, valid))
    return result


def _audit_protocols(frame: pd.DataFrame) -> dict[str, list[dict[str, int]]]:
    result: dict[str, list[dict[str, int]]] = {}
    for protocol, group_column in (
        ("exercise_disjoint", "exercise_id"),
        ("concept_disjoint", "concept_id"),
    ):
        rows: list[dict[str, int]] = []
        for split, train, valid in protocol_splits(frame, protocol):
            train_groups = set(frame.loc[train, group_column])
            valid_groups = set(frame.loc[valid, group_column])
            if train_groups & valid_groups:
                raise ValueError(f"E680 {protocol} fold {split} leaks groups.")
            rows.append(
                {
                    "split": int(split),
                    "train_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "validation_positive": int(frame.loc[valid, "target"].sum()),
                    "train_groups": len(train_groups),
                    "validation_groups": len(valid_groups),
                    "held_out_group_overlap": 0,
                }
            )
        result[protocol] = rows
    return result


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_index",
        "example_id",
        "exercise_id",
        "concept_id",
        "exercise_fold",
        "concept_fold",
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


def build_canonical(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    paths = _paths(project_root)
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    if set(source) != {"prepDataset", "shapeDataset"} or source["shapeDataset"]:
        raise ValueError("E680 top-level source schema changed.")
    prep = source["prepDataset"]
    if not isinstance(prep, dict) or len(prep) != EXPECTED_ROWS:
        raise ValueError("E680 preparation row count changed.")
    required = {
        "past_convo",
        "img",
        "engPrep",
        "engObj",
        "engColor",
        "studentActions",
        "tutorActions",
        "tutorResponses",
        "tutorKeys",
        "grammarRules",
    }
    records: list[dict[str, object]] = []
    for source_index, source_key in enumerate(
        sorted(prep, key=lambda value: int(value))
    ):
        row = prep[source_key]
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("E680 preparation schema changed.")
        actions = row["studentActions"]
        if not isinstance(actions, list) or len(actions) != 4:
            raise ValueError("E680 student action vector changed.")
        target = int(_boolean(actions[0]))
        exercise = str(row["img"])
        concept = concept_key(row)
        if not exercise.strip():
            raise ValueError("E680 exercise identifier is empty.")
        text = canonical_dialogue_text(row["past_convo"])
        identity = hashlib.sha256(
            f"E680|{source_key}|{text}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_index": source_index,
                "example_id": identity,
                "exercise_id": exercise,
                "concept_id": concept,
                "exercise_fold": group_fold("exercise_disjoint", exercise),
                "concept_fold": group_fold("concept_disjoint", concept),
                "target": target,
                "text": text,
            }
        )
    frame = pd.DataFrame.from_records(records)
    counts = frame["target"].value_counts().sort_index().to_dict()
    if len(frame) != EXPECTED_ROWS or counts != EXPECTED_TARGET_COUNTS:
        raise ValueError(f"E680 target counts changed: {counts}")
    if frame["example_id"].duplicated().any():
        raise ValueError("E680 canonical identity is duplicated.")
    protocol_audit = _audit_protocols(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "target_counts": counts,
        "unique_exercises": int(frame["exercise_id"].nunique()),
        "unique_concepts": int(frame["concept_id"].nunique()),
        "protocol_audit": protocol_audit,
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "action_or_concept_in_text": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        **verify_sources(project_root),
    }
    paths["canonical_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_canonical(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    if not CANONICAL_CONTENT_SHA256 or not CANONICAL_PARQUET_SHA256:
        raise RuntimeError("E680 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E680 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E680 canonical ordered-content SHA-256 changed.")
    _audit_protocols(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"], columns=["example_id", "text"])
    order = frame["example_id"].map(
        lambda value: hashlib.sha256(
            f"E680-benchmark|{value}".encode("utf-8")
        ).hexdigest()
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
        "action_labels_accessed": False,
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
        raise ValueError("E680 benchmark did not authorize cache construction.")
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
            print(f"e680_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(
            np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0
        )
    ):
        raise ValueError("E680 embedding cache audit failed.")
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


def _group_bootstrap(
    gains: np.ndarray, groups: Sequence[object], salt: int
) -> dict[str, object]:
    group_values = np.asarray([str(value) for value in groups])
    unique = np.unique(group_values)
    sums = np.asarray(
        [gains[group_values == group].sum() for group in unique],
        dtype=np.float64,
    )
    counts = np.asarray(
        [(group_values == group).sum() for group in unique], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + salt)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 64):
        stop = min(start + 64, BOOTSTRAP_REPLICATES)
        sample = rng.integers(
            0, len(unique), size=(stop - start, len(unique))
        )
        draws[start:stop] = (
            sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "groups": len(unique),
        "mean_log_loss_gain": float(draws.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "support": float(np.mean(draws > 0)),
    }


def external_gate(
    metrics: dict[str, float],
    prior_metrics: dict[str, float],
    fold_rows: Sequence[dict[str, object]],
    bootstrap: dict[str, object],
) -> dict[str, bool]:
    return {
        "roc_auc": metrics["roc_auc"] >= GATE_THRESHOLDS["roc_auc"],
        "macro_f1": metrics["macro_f1"] >= GATE_THRESHOLDS["macro_f1"],
        "log_loss_gain": (
            prior_metrics["log_loss"] - metrics["log_loss"]
            >= GATE_THRESHOLDS["log_loss_gain"]
        ),
        "brier_gain": (
            prior_metrics["brier_score"] - metrics["brier_score"]
            >= GATE_THRESHOLDS["brier_gain"]
        ),
        "ece_10": metrics["ece_10"] <= GATE_THRESHOLDS["ece_10"],
        "positive_log_loss_gain_every_fold": all(
            float(row["log_loss_gain"]) > 0 for row in fold_rows
        ),
        "bootstrap_support": (
            float(bootstrap["support"])
            >= GATE_THRESHOLDS["bootstrap_support"]
        ),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(paths["embedding_metadata"].read_text(encoding="utf-8"))
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E680 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    if embeddings.shape != (len(frame), EMBEDDING_DIMENSION):
        raise ValueError("E680 embedding cache shape changed.")
    target = frame["target"].to_numpy(dtype=np.int8)

    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    protocol_results: dict[str, dict[str, object]] = {}
    for protocol_index, (protocol, group_column) in enumerate(
        (
            ("exercise_disjoint", "exercise_id"),
            ("concept_disjoint", "concept_id"),
        )
    ):
        probability = np.full(len(frame), np.nan)
        prior = np.full(len(frame), np.nan)
        fold_rows: list[dict[str, object]] = []
        for split, train, valid in protocol_splits(frame, protocol):
            model = LogisticRegression(
                C=REGULARIZATION_C,
                solver="lbfgs",
                max_iter=1000,
                random_state=20260728,
            )
            model.fit(embeddings[train], target[train])
            probability[valid] = model.predict_proba(embeddings[valid])[:, 1]
            prior[valid] = float(target[train].mean())
            fold_metrics = binary_metrics(target[valid], probability[valid])
            fold_prior = binary_metrics(target[valid], prior[valid])
            fold_macro_f1 = float(
                f1_score(
                    target[valid],
                    probability[valid] >= 0.5,
                    average="macro",
                )
            )
            fold_rows.append(
                {
                    "split": split,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    **fold_metrics,
                    "macro_f1": fold_macro_f1,
                    "prior_log_loss": fold_prior["log_loss"],
                    "prior_brier_score": fold_prior["brier_score"],
                    "log_loss_gain": fold_prior["log_loss"]
                    - fold_metrics["log_loss"],
                    "brier_gain": fold_prior["brier_score"]
                    - fold_metrics["brier_score"],
                }
            )
            model_rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "coefficient": model.coef_[0].astype(np.float64),
                    "intercept": float(model.intercept_[0]),
                }
            )
        if not np.isfinite(probability).all() or not np.isfinite(prior).all():
            raise ValueError(f"E680 {protocol} OOF prediction is incomplete.")
        metrics = binary_metrics(target, probability)
        metrics["macro_f1"] = float(
            f1_score(target, probability >= 0.5, average="macro")
        )
        prior_metrics = binary_metrics(target, prior)
        target_float = target.astype(np.float64)
        candidate_loss = -(
            target_float * np.log(np.clip(probability, 1e-6, 1 - 1e-6))
            + (1 - target_float)
            * np.log1p(-np.clip(probability, 1e-6, 1 - 1e-6))
        )
        prior_loss = -(
            target_float * np.log(np.clip(prior, 1e-6, 1 - 1e-6))
            + (1 - target_float)
            * np.log1p(-np.clip(prior, 1e-6, 1 - 1e-6))
        )
        bootstrap = _group_bootstrap(
            prior_loss - candidate_loss,
            frame[group_column],
            salt=protocol_index + 1,
        )
        clauses = external_gate(metrics, prior_metrics, fold_rows, bootstrap)
        protocol_results[protocol] = {
            "metrics": metrics,
            "prior_metrics": prior_metrics,
            "log_loss_gain": prior_metrics["log_loss"] - metrics["log_loss"],
            "brier_gain": prior_metrics["brier_score"]
            - metrics["brier_score"],
            "folds": fold_rows,
            "bootstrap": bootstrap,
            "clauses": clauses,
            "passes": all(clauses.values()),
        }
        predictions = frame[
            [
                "example_id",
                "source_index",
                "exercise_id",
                "concept_id",
                "target",
            ]
        ].copy()
        predictions.insert(0, "protocol", protocol)
        predictions["probability"] = probability
        predictions["fold_prior"] = prior
        prediction_frames.append(predictions)

    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_cima_student_state_transfer"
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "oof_predictions.parquet"
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        predictions_path, index=False
    )
    model_path = run_dir / "fold_models.npz"
    np.savez_compressed(
        model_path,
        protocols=np.asarray([row["protocol"] for row in model_rows]),
        splits=np.asarray([row["split"] for row in model_rows]),
        coefficients=np.vstack([row["coefficient"] for row in model_rows]),
        intercepts=np.asarray([row["intercept"] for row in model_rows]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "rows": len(frame),
        "protocols": protocol_results,
        "thresholds": GATE_THRESHOLDS,
        "passes_external_gate": all(
            value["passes"] for value in protocol_results.values()
        ),
        "runtime": runtime,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_sha256": LICENSE_SHA256,
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
        description="Run the frozen E680 CIMA student-state transfer."
    )
    parser.add_argument(
        "stage", choices=("canonicalize", "benchmark", "build-cache", "validate")
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
