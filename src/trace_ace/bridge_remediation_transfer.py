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
from trace_ace.metrics import binary_metrics
from trace_ace.simulator_outcome_transfer import (
    _collapse,
    _load_encoder_left,
    _sha256,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E670_bridge_expert_remediation_v1"
SOURCE_COMMIT = "8f469883aa7d7a5c1d64e5c961a033ce71d21f5e"
SOURCE_CARD_SHA256 = (
    "279b838af2dc1b73fe69d2757ccb4b8e00251708621d1f66d69ce78703584e63"
)
SOURCE_HASHES = {
    "train": "870fa10d299711d315718d1995f234988582d7168fa9c054c267ededd89ba260",
    "validation": "1d87ffece5fba05b6f5e91a2bcd0e5d556afb00973dd47e09969c875576bf327",
    "test": "7c1f5eccce6f635924ca495439c7a9d70885c73122dd4caf8ad9066c657265e9",
}
EXPECTED_SPLIT_ROWS = {"train": 419, "validation": 71, "test": 210}
EXPECTED_ROWS = 700
FOLDS = 5
REGULARIZATION_C = 0.1
MAX_PROJECTED_HOURS = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
TASK_LINE = (
    "Task: assess how effectively the candidate tutor response remediates the "
    "demonstrated mathematical misunderstanding."
)
CANONICAL_CONTENT_SHA256 = (
    "1a6a300fe87904aa83b9a3ce6e804f6583d763b5e9805566c55d5e7c9dea2b32"
)
CANONICAL_PARQUET_SHA256 = (
    "a460bfa5d864c29dfd02e08e9b92e8173bfad9829fc69ebd1fa205707aa734f1"
)
GATE_THRESHOLDS = {
    "accuracy": 0.60,
    "roc_auc": 0.65,
    "log_loss_gain": 0.025,
    "brier_gain": 0.010,
    "ece_10": 0.10,
    "bootstrap_support": 0.95,
}


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "Bridge"
    return {
        "source_root": source_root,
        "train": source_root / "train.json",
        "validation": source_root / "validation.json",
        "test": source_root / "test.json",
        "canonical": paths.cache_dir / "bridge_e670_canonical.parquet",
        "canonical_metadata": paths.cache_dir
        / "bridge_e670_canonical.metadata.json",
        "benchmark": paths.cache_dir / "bridge_e670_benchmark.json",
        "embeddings": paths.cache_dir / "bridge_e670_embeddings.npy",
        "embedding_metadata": paths.cache_dir
        / "bridge_e670_embeddings.metadata.json",
        "progress": paths.cache_dir / "bridge_e670_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    for split, expected in SOURCE_HASHES.items():
        if _sha256(paths[split]) != expected:
            raise ValueError(f"E670 {split} JSON SHA-256 changed.")
    return {
        "source_commit": SOURCE_COMMIT,
        "source_card_sha256": SOURCE_CARD_SHA256,
        **{f"{split}_sha256": value for split, value in SOURCE_HASHES.items()},
    }


def _response_lines(
    values: object,
    *,
    required_role: str | None = None,
    require_nonempty_component: bool = False,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("E670 dialogue component is empty or not a list.")
    lines: list[str] = []
    nonempty = 0
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("E670 dialogue turn is not an object.")
        role = str(item.get("user", "")).strip().casefold()
        if role not in {"student", "tutor"}:
            raise ValueError(f"E670 dialogue role changed: {role!r}")
        if required_role is not None and role != required_role:
            raise ValueError(f"E670 candidate response contains {role!r} turn.")
        text = _collapse(item.get("text", ""))
        nonempty += int(bool(text))
        lines.append(f"{role.title()}: {text}".rstrip())
    if require_nonempty_component and nonempty == 0:
        raise ValueError("E670 candidate response has no nonempty text.")
    return lines


def canonical_candidate_text(history: object, candidate: object) -> str:
    history_lines = _response_lines(history)
    if len(history_lines) != 4:
        raise ValueError("E670 requires exactly four released history turns.")
    candidate_lines = _response_lines(
        candidate,
        required_role="tutor",
        require_nonempty_component=True,
    )
    return "\n".join([*history_lines, *candidate_lines, TASK_LINE])


def session_root(c_id: str) -> str:
    value = str(c_id)
    if "_" not in value:
        raise ValueError(f"E670 c_id lacks a turn suffix: {value!r}")
    root, suffix = value.rsplit("_", 1)
    if not root or not suffix:
        raise ValueError(f"E670 c_id is malformed: {value!r}")
    return root


def group_fold(protocol: str, value: str) -> int:
    if protocol == "session_disjoint":
        prefix = "session"
    elif protocol == "lesson_disjoint":
        prefix = "lesson"
    else:
        raise ValueError(f"Unknown E670 protocol: {protocol}")
    digest = hashlib.sha256(
        f"E670|{prefix}|{value}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % FOLDS


def protocol_splits(
    frame: pd.DataFrame, protocol: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if protocol == "session_disjoint":
        column = "session_fold"
    elif protocol == "lesson_disjoint":
        column = "lesson_fold"
    else:
        raise ValueError(f"Unknown E670 protocol: {protocol}")
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for fold in range(FOLDS):
        valid = frame[column].eq(fold).to_numpy()
        train = ~valid
        if not train.any() or not valid.any():
            raise ValueError(f"E670 {protocol} fold {fold} is empty.")
        result.append((str(fold), train, valid))
    return result


def _audit_protocols(frame: pd.DataFrame) -> dict[str, list[dict[str, int]]]:
    result: dict[str, list[dict[str, int]]] = {}
    for protocol, group_column in (
        ("session_disjoint", "session_root"),
        ("lesson_disjoint", "lesson_topic"),
    ):
        audits: list[dict[str, int]] = []
        for split, train, valid in protocol_splits(frame, protocol):
            train_groups = set(frame.loc[train, group_column])
            valid_groups = set(frame.loc[valid, group_column])
            if train_groups & valid_groups:
                raise ValueError(f"E670 {protocol} fold {split} leaks groups.")
            if set(frame.loc[train, "pair_id"]) & set(frame.loc[valid, "pair_id"]):
                raise ValueError(f"E670 {protocol} fold {split} leaks pairs.")
            audits.append(
                {
                    "split": int(split),
                    "train_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "train_groups": len(train_groups),
                    "validation_groups": len(valid_groups),
                    "held_out_group_overlap": 0,
                }
            )
        result[protocol] = audits
    return result


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_split",
        "source_index",
        "pair_id",
        "c_id",
        "session_root",
        "lesson_topic",
        "session_fold",
        "lesson_fold",
        "novice_text",
        "expert_text",
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
    required = {
        "c_id",
        "lesson_topic",
        "c_h",
        "c_r",
        "c_r_",
        "c_revision",
        "e",
        "z_what",
        "z_why",
    }
    records: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        rows = json.loads(paths[split].read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != EXPECTED_SPLIT_ROWS[split]:
            raise ValueError(f"E670 {split} split count changed.")
        for source_index, row in enumerate(rows):
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"E670 {split} source schema changed.")
            c_id = str(row["c_id"])
            root = session_root(c_id)
            lesson = str(row["lesson_topic"])
            if not lesson.strip():
                raise ValueError("E670 lesson topic is empty.")
            identity = hashlib.sha256(
                f"E670|{split}|{source_index}|{c_id}".encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "source_split": split,
                    "source_index": source_index,
                    "pair_id": identity,
                    "c_id": c_id,
                    "session_root": root,
                    "lesson_topic": lesson,
                    "session_fold": group_fold("session_disjoint", root),
                    "lesson_fold": group_fold("lesson_disjoint", lesson),
                    "novice_text": canonical_candidate_text(
                        row["c_h"], row["c_r"]
                    ),
                    "expert_text": canonical_candidate_text(
                        row["c_h"], row["c_r_"]
                    ),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_ROWS or frame["pair_id"].duplicated().any():
        raise ValueError("E670 canonical identity audit failed.")
    if (frame["novice_text"] == frame["expert_text"]).any():
        raise ValueError("E670 contains an identical canonical response pair.")
    c_id_lessons = frame.groupby("c_id")["lesson_topic"].nunique()
    if int(c_id_lessons.max()) != 1:
        raise ValueError("E670 duplicate c_id spans multiple lessons.")
    protocol_audit = _audit_protocols(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "split_counts": frame["source_split"].value_counts().sort_index().to_dict(),
        "unique_c_ids": int(frame["c_id"].nunique()),
        "unique_session_roots": int(frame["session_root"].nunique()),
        "unique_lessons": int(frame["lesson_topic"].nunique()),
        "protocol_audit": protocol_audit,
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "forbidden_annotations_in_text": False,
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
        raise RuntimeError("E670 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E670 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E670 canonical ordered-content SHA-256 changed.")
    _audit_protocols(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"])
    order = frame["pair_id"].map(
        lambda value: hashlib.sha256(
            f"E670-benchmark|{value}".encode("utf-8")
        ).hexdigest()
    )
    sample = frame.assign(_order=order).sort_values("_order").head(32)
    texts: list[str] = []
    for row in sample.itertuples(index=False):
        texts.extend([row.novice_text, row.expert_text])
    model = _load_encoder_left(project_root)
    started = time.perf_counter()
    matrix = _encode(model, texts)
    elapsed = time.perf_counter() - started
    projected_texts = 2 * len(frame)
    projected_seconds = elapsed / len(texts) * projected_texts
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_pairs": len(sample),
        "benchmark_texts": len(texts),
        "benchmark_batches": math.ceil(len(texts) / BATCH_SIZE),
        "elapsed_seconds": elapsed,
        "projected_pairs": len(frame),
        "projected_texts": projected_texts,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600,
        "peak_rss_bytes": peak_rss,
        "shape": list(matrix.shape),
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "runtime": runtime,
        "preference_labels_accessed": False,
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
        raise ValueError("E670 benchmark did not authorize cache construction.")
    frame = pd.read_parquet(
        paths["canonical"], columns=["novice_text", "expert_text"]
    )
    texts: list[str] = []
    for row in frame.itertuples(index=False):
        texts.extend([row.novice_text, row.expert_text])
    model = _load_encoder_left(project_root)
    matrix = np.lib.format.open_memmap(
        paths["embeddings"],
        mode="w+",
        dtype=np.float32,
        shape=(len(texts), EMBEDDING_DIMENSION),
    )
    started = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(texts))
        matrix[start:stop] = _encode(model, texts[start:stop])
        matrix.flush()
        paths["progress"].write_text(
            json.dumps(
                {
                    "completed_texts": stop,
                    "total_texts": len(texts),
                    "completed_pairs": stop // 2,
                    "total_pairs": len(frame),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if stop % 128 == 0 or stop == len(texts):
            print(
                f"e670_cache completed_texts={stop}/{len(texts)}",
                flush=True,
            )
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (2 * len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(
            np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0
        )
    ):
        raise ValueError("E670 embedding cache audit failed.")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": len(frame),
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


def paired_training_matrix(
    differences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(differences, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != EMBEDDING_DIMENSION
        or not np.isfinite(values).all()
    ):
        raise ValueError("E670 pair-difference matrix is invalid.")
    features = np.vstack([values, -values])
    target = np.concatenate(
        [
            np.ones(len(values), dtype=np.int8),
            np.zeros(len(values), dtype=np.int8),
        ]
    )
    return features, target


def _balanced_metrics(expert_probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(expert_probability, dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("E670 expert probabilities are invalid.")
    target = np.concatenate(
        [
            np.ones(len(probability), dtype=np.int8),
            np.zeros(len(probability), dtype=np.int8),
        ]
    )
    balanced_probability = np.concatenate([probability, 1.0 - probability])
    metrics = binary_metrics(target, balanced_probability)
    metrics["accuracy"] = float(np.mean(probability > 0.5))
    metrics["expert_probability_mean"] = float(probability.mean())
    return metrics


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
    fold_rows: Sequence[dict[str, object]],
    bootstrap: dict[str, object],
) -> dict[str, bool]:
    return {
        "accuracy": metrics["accuracy"] >= GATE_THRESHOLDS["accuracy"],
        "roc_auc": metrics["roc_auc"] >= GATE_THRESHOLDS["roc_auc"],
        "log_loss_gain": (
            math.log(2.0) - metrics["log_loss"]
            >= GATE_THRESHOLDS["log_loss_gain"]
        ),
        "brier_gain": (
            0.25 - metrics["brier_score"]
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

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(paths["embedding_metadata"].read_text(encoding="utf-8"))
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E670 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    if embeddings.shape != (2 * len(frame), EMBEDDING_DIMENSION):
        raise ValueError("E670 embedding cache shape changed.")
    differences = embeddings[1::2] - embeddings[0::2]

    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    protocol_results: dict[str, dict[str, object]] = {}
    for protocol_index, (protocol, group_column) in enumerate(
        (
            ("session_disjoint", "session_root"),
            ("lesson_disjoint", "lesson_topic"),
        )
    ):
        expert_probability = np.full(len(frame), np.nan)
        fold_rows: list[dict[str, object]] = []
        for split, train, valid in protocol_splits(frame, protocol):
            train_features, train_target = paired_training_matrix(
                differences[train]
            )
            model = LogisticRegression(
                C=REGULARIZATION_C,
                solver="lbfgs",
                fit_intercept=False,
                max_iter=1000,
                random_state=20260728,
            )
            model.fit(train_features, train_target)
            if tuple(model.classes_) != (0, 1):
                raise ValueError("E670 pairwise classifier classes changed.")
            probability = model.predict_proba(differences[valid])[:, 1]
            expert_probability[valid] = probability
            fold_metrics = _balanced_metrics(probability)
            fold_rows.append(
                {
                    "split": split,
                    "train_pairs": int(train.sum()),
                    "validation_pairs": int(valid.sum()),
                    **fold_metrics,
                    "log_loss_gain": math.log(2.0)
                    - fold_metrics["log_loss"],
                    "brier_gain": 0.25 - fold_metrics["brier_score"],
                }
            )
            model_rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "coefficient": model.coef_[0].astype(np.float64),
                }
            )
        if not np.isfinite(expert_probability).all():
            raise ValueError(f"E670 {protocol} OOF prediction is incomplete.")
        metrics = _balanced_metrics(expert_probability)
        per_pair_gain = math.log(2.0) + np.log(
            np.clip(expert_probability, 1e-6, 1.0 - 1e-6)
        )
        bootstrap = _group_bootstrap(
            per_pair_gain,
            frame[group_column],
            salt=protocol_index + 1,
        )
        clauses = external_gate(metrics, fold_rows, bootstrap)
        protocol_results[protocol] = {
            "metrics": metrics,
            "prior_log_loss": math.log(2.0),
            "prior_brier_score": 0.25,
            "log_loss_gain": math.log(2.0) - metrics["log_loss"],
            "brier_gain": 0.25 - metrics["brier_score"],
            "folds": fold_rows,
            "bootstrap": bootstrap,
            "clauses": clauses,
            "passes": all(clauses.values()),
        }
        predictions = frame[
            [
                "pair_id",
                "source_split",
                "source_index",
                "c_id",
                "session_root",
                "lesson_topic",
            ]
        ].copy()
        predictions.insert(0, "protocol", protocol)
        predictions["expert_probability"] = expert_probability
        prediction_frames.append(predictions)

    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_bridge_remediation_transfer"
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
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "pairs": len(frame),
        "protocols": protocol_results,
        "thresholds": GATE_THRESHOLDS,
        "passes_external_gate": all(
            value["passes"] for value in protocol_results.values()
        ),
        "runtime": runtime,
        "source_commit": SOURCE_COMMIT,
        "source_card_sha256": SOURCE_CARD_SHA256,
        "source_hashes": SOURCE_HASHES,
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
        description="Run the frozen E670 Bridge expert-remediation transfer."
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
