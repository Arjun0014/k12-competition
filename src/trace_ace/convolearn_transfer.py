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
from trace_ace.simulator_outcome_transfer import (
    _collapse,
    _load_encoder_left,
    _sha256,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E660_convolearn_quality_v1"
SOURCE_COMMIT = "f250e930356f2092462c4d1cd6bb2aae85689b1f"
SOURCE_SHA256 = "c1599655c3a2a3ef5fd199906200f02afd6b26a1bd4b5fbc16a18d6b784d5c04"
LICENSE_SHA256 = "63d12a3c43645e188d03599f4e500a49309048f77670e17209628cdea56f3c09"
TASK_LINE = (
    "Task: represent the demonstrated tutoring quality and pedagogical approach."
)
EXPECTED_ROWS = 2_134
SUBDIMENSION_FOLDS = 5
RIDGE_ALPHA = 10.0
CLASSIFIER_C = 0.1
MAX_PROJECTED_HOURS = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
CANONICAL_CONTENT_SHA256 = (
    "952587761234e94183217b77f5637d131f11cc22aa25f59e013ef7dadb0717b8"
)
CANONICAL_PARQUET_SHA256 = (
    "664eabd969d837cb0fba8f490caa897f9ac2f9c28290b8e8c94d960d7d513941"
)
DIMENSIONS = (
    "Accountability",
    "Cognitive Engagement",
    "Cultural Responsiveness",
    "Formative Assessment",
    "Metacognition",
    "Power Dynamics",
)
GATE_THRESHOLDS = {
    "effectiveness_pearson": 0.25,
    "effectiveness_spearman": 0.25,
    "effectiveness_rmse_gain": 0.015,
    "completeness_pearson": 0.20,
    "completeness_spearman": 0.20,
    "completeness_rmse_gain": 0.010,
    "dimension_macro_f1": 0.45,
    "dimension_log_loss_gain": 0.10,
    "bootstrap_support": 0.90,
}


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "ConvoLearn"
    return {
        "source_root": source_root,
        "source": source_root / "data" / "train-00000-of-00001.parquet",
        "license": source_root / "README.md",
        "canonical": paths.cache_dir / "convolearn_e660_canonical.parquet",
        "canonical_metadata": paths.cache_dir / "convolearn_e660_canonical.metadata.json",
        "benchmark": paths.cache_dir / "convolearn_e660_benchmark.json",
        "embeddings": paths.cache_dir / "convolearn_e660_embeddings.npy",
        "embedding_metadata": paths.cache_dir / "convolearn_e660_embeddings.metadata.json",
        "progress": paths.cache_dir / "convolearn_e660_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def verify_sources(project_root: str | Path) -> None:
    paths = _paths(project_root)
    if _sha256(paths["source"]) != SOURCE_SHA256:
        raise ValueError("E660 source Parquet SHA-256 changed.")
    if _sha256(paths["license"]) != LICENSE_SHA256:
        raise ValueError("E660 MIT dataset-card SHA-256 changed.")
    commit = subprocess.run(
        ["git", "-C", str(paths["source_root"]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"E660 source commit changed: {commit}")


def compact_dialogue(value: object) -> str:
    lines = [_collapse(line) for line in str(value).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("E660 dialogue is empty.")
    selected: list[int] = []
    for positions in np.array_split(np.arange(len(lines), dtype=np.int64), 5):
        if len(positions):
            selected.extend([int(positions[0]), int(positions[-1])])
    ordered = sorted(set(selected))
    compact = [" ".join(lines[index].split()[:16]) for index in ordered]
    compact.append(TASK_LINE)
    return "\n".join(compact)


def subdimension_fold(value: str) -> int:
    digest = hashlib.sha256(f"E660|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % SUBDIMENSION_FOLDS


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_index",
        "example_id",
        "subdimension",
        "dimension",
        "topic",
        "subdimension_fold",
        "effectiveness_target",
        "completeness_target",
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


def protocol_splits(
    frame: pd.DataFrame, protocol: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if protocol == "subdimension_disjoint":
        values: Sequence[object] = tuple(range(SUBDIMENSION_FOLDS))
        column = "subdimension_fold"
    elif protocol == "topic_disjoint":
        values = tuple(sorted(frame["topic"].unique()))
        column = "topic"
    else:
        raise ValueError(f"Unknown E660 protocol: {protocol}")
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    for value in values:
        valid = frame[column].eq(value).to_numpy()
        train = ~valid
        if not train.any() or not valid.any():
            raise ValueError(f"E660 empty {protocol} split: {value}")
        splits.append((str(value), train, valid))
    return splits


def _audit_protocols(frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for protocol in ("subdimension_disjoint", "topic_disjoint"):
        rows: list[dict[str, object]] = []
        for split, train, valid in protocol_splits(frame, protocol):
            train_subdimensions = set(frame.loc[train, "subdimension"])
            valid_subdimensions = set(frame.loc[valid, "subdimension"])
            train_topics = set(frame.loc[train, "topic"])
            valid_topics = set(frame.loc[valid, "topic"])
            if protocol == "subdimension_disjoint":
                overlap = train_subdimensions & valid_subdimensions
            else:
                overlap = train_topics & valid_topics
            if overlap:
                raise ValueError(f"E660 {protocol} split {split} leaks groups.")
            rows.append(
                {
                    "split": split,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "train_subdimensions": len(train_subdimensions),
                    "validation_subdimensions": len(valid_subdimensions),
                    "train_topics": len(train_topics),
                    "validation_topics": len(valid_topics),
                    "held_out_group_overlap": 0,
                }
            )
        result[protocol] = rows
    return result


def build_canonical(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    paths = _paths(project_root)
    source = pd.read_parquet(paths["source"])
    required = {
        "kb_subdim",
        "kb_dim",
        "effectiveness_consensus",
        "completeness_consensus",
        "cleaned_conversation",
        "earthscience_topic",
        "num_exchanges",
    }
    if len(source) != EXPECTED_ROWS or not required.issubset(source.columns):
        raise ValueError("E660 source schema or row count changed.")
    if source[list(required)].isna().any().any():
        raise ValueError("E660 source contains missing values.")
    if source["cleaned_conversation"].duplicated().any():
        raise ValueError("E660 source contains duplicate dialogue text.")
    if tuple(sorted(source["kb_dim"].unique())) != DIMENSIONS:
        raise ValueError("E660 dimension labels changed.")

    records: list[dict[str, object]] = []
    for source_index, row in source.reset_index(drop=True).iterrows():
        effectiveness = float(row["effectiveness_consensus"])
        completeness = float(row["completeness_consensus"])
        if not 1.0 <= effectiveness <= 5.0 or not 1.0 <= completeness <= 3.0:
            raise ValueError("E660 rating is outside the released scale.")
        identity = hashlib.sha256(
            f"E660|{source_index}|{row['cleaned_conversation']}".encode("utf-8")
        ).hexdigest()
        subdimension = str(row["kb_subdim"])
        records.append(
            {
                "source_index": int(source_index),
                "example_id": identity,
                "subdimension": subdimension,
                "dimension": str(row["kb_dim"]),
                "topic": str(row["earthscience_topic"]),
                "subdimension_fold": subdimension_fold(subdimension),
                "effectiveness_target": (effectiveness - 1.0) / 4.0,
                "completeness_target": (completeness - 1.0) / 2.0,
                "text": compact_dialogue(row["cleaned_conversation"]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_ROWS or frame["example_id"].duplicated().any():
        raise ValueError("E660 canonical identity audit failed.")
    protocol_audit = _audit_protocols(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_sha256": LICENSE_SHA256,
        "rows": len(frame),
        "effectiveness_mean": float(frame["effectiveness_target"].mean()),
        "effectiveness_std": float(frame["effectiveness_target"].std(ddof=0)),
        "completeness_mean": float(frame["completeness_target"].mean()),
        "completeness_std": float(frame["completeness_target"].std(ddof=0)),
        "dimension_counts": frame["dimension"].value_counts().sort_index().to_dict(),
        "topic_counts": frame["topic"].value_counts().sort_index().to_dict(),
        "protocol_audit": protocol_audit,
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "ratings_or_labels_in_text": False,
        "dimension_or_topic_in_text": False,
        "competition_outcomes_accessed": False,
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
        raise RuntimeError("E660 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E660 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E660 canonical ordered-content SHA-256 changed.")
    _audit_protocols(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"], columns=["example_id", "text"])
    order = frame["example_id"].map(
        lambda value: hashlib.sha256(f"E660-benchmark|{value}".encode()).hexdigest()
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
        raise ValueError("E660 benchmark did not authorize cache construction.")
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
            print(f"e660_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0)
    ):
        raise ValueError("E660 embedding cache audit failed.")
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
        [gains[group_values == group].mean() for group in unique], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + salt)
    draws = grouped[
        rng.integers(0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped)))
    ].mean(axis=1)
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "groups": len(unique),
        "mean_squared_error_gain": float(draws.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "support": float(np.mean(draws > 0)),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, log_loss

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(paths["embedding_metadata"].read_text(encoding="utf-8"))
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E660 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    effectiveness = frame["effectiveness_target"].to_numpy(dtype=np.float64)
    completeness = frame["completeness_target"].to_numpy(dtype=np.float64)
    dimension_lookup = {value: index for index, value in enumerate(DIMENSIONS)}
    dimension = frame["dimension"].map(dimension_lookup).to_numpy(dtype=np.int64)

    all_prediction_rows: list[pd.DataFrame] = []
    all_model_rows: list[dict[str, object]] = []
    protocol_results: dict[str, dict[str, object]] = {}
    for protocol_index, protocol in enumerate(
        ("subdimension_disjoint", "topic_disjoint")
    ):
        effect_prediction = np.full(len(frame), np.nan)
        complete_prediction = np.full(len(frame), np.nan)
        dimension_probability = np.full((len(frame), len(DIMENSIONS)), np.nan)
        effect_prior = np.full(len(frame), np.nan)
        complete_prior = np.full(len(frame), np.nan)
        dimension_prior = np.full((len(frame), len(DIMENSIONS)), np.nan)
        fold_rows: list[dict[str, object]] = []
        for split, train, valid in protocol_splits(frame, protocol):
            effect_model = Ridge(
                alpha=RIDGE_ALPHA,
                fit_intercept=True,
                solver="lsqr",
                max_iter=1000,
                tol=1e-6,
            )
            complete_model = Ridge(
                alpha=RIDGE_ALPHA,
                fit_intercept=True,
                solver="lsqr",
                max_iter=1000,
                tol=1e-6,
            )
            dimension_model = LogisticRegression(
                C=CLASSIFIER_C,
                solver="lbfgs",
                max_iter=1000,
                random_state=20260728,
            )
            effect_model.fit(embeddings[train], effectiveness[train])
            complete_model.fit(embeddings[train], completeness[train])
            dimension_model.fit(embeddings[train], dimension[train])
            effect_prediction[valid] = np.clip(
                effect_model.predict(embeddings[valid]), 0.0, 1.0
            )
            complete_prediction[valid] = np.clip(
                complete_model.predict(embeddings[valid]), 0.0, 1.0
            )
            probability = dimension_model.predict_proba(embeddings[valid])
            if tuple(dimension_model.classes_) != tuple(range(len(DIMENSIONS))):
                raise ValueError("E660 fold is missing a broad dimension class.")
            dimension_probability[valid] = probability
            effect_prior[valid] = float(effectiveness[train].mean())
            complete_prior[valid] = float(completeness[train].mean())
            counts = np.bincount(dimension[train], minlength=len(DIMENSIONS)) + 1.0
            dimension_prior[valid] = counts / counts.sum()
            effect_fold = _regression_metrics(
                effectiveness[valid], effect_prediction[valid]
            )
            complete_fold = _regression_metrics(
                completeness[valid], complete_prediction[valid]
            )
            fold_rows.append(
                {
                    "split": split,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "effectiveness": effect_fold,
                    "effectiveness_prior_rmse": float(
                        np.mean((effectiveness[valid] - effect_prior[valid]) ** 2)
                        ** 0.5
                    ),
                    "effectiveness_rmse_gain": float(
                        np.mean((effectiveness[valid] - effect_prior[valid]) ** 2)
                        ** 0.5
                        - effect_fold["rmse"]
                    ),
                    "completeness": complete_fold,
                    "completeness_prior_rmse": float(
                        np.mean((completeness[valid] - complete_prior[valid]) ** 2)
                        ** 0.5
                    ),
                    "completeness_rmse_gain": float(
                        np.mean((completeness[valid] - complete_prior[valid]) ** 2)
                        ** 0.5
                        - complete_fold["rmse"]
                    ),
                }
            )
            all_model_rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "effect_coef": effect_model.coef_.astype(np.float64),
                    "effect_intercept": float(effect_model.intercept_),
                    "complete_coef": complete_model.coef_.astype(np.float64),
                    "complete_intercept": float(complete_model.intercept_),
                    "dimension_coef": dimension_model.coef_.astype(np.float64),
                    "dimension_intercept": dimension_model.intercept_.astype(np.float64),
                }
            )
        if (
            not np.isfinite(effect_prediction).all()
            or not np.isfinite(complete_prediction).all()
            or not np.isfinite(dimension_probability).all()
        ):
            raise ValueError(f"E660 {protocol} OOF predictions are incomplete.")
        effect_metrics = _regression_metrics(effectiveness, effect_prediction)
        complete_metrics = _regression_metrics(completeness, complete_prediction)
        effect_prior_rmse = float(
            np.mean((effectiveness - effect_prior) ** 2) ** 0.5
        )
        complete_prior_rmse = float(
            np.mean((completeness - complete_prior) ** 2) ** 0.5
        )
        effect_gain = effect_prior_rmse - effect_metrics["rmse"]
        complete_gain = complete_prior_rmse - complete_metrics["rmse"]
        dimension_loss = float(
            log_loss(dimension, dimension_probability, labels=range(len(DIMENSIONS)))
        )
        prior_loss = float(
            log_loss(dimension, dimension_prior, labels=range(len(DIMENSIONS)))
        )
        dimension_macro_f1 = float(
            f1_score(
                dimension,
                np.argmax(dimension_probability, axis=1),
                average="macro",
            )
        )
        group_column = "subdimension" if protocol == "subdimension_disjoint" else "topic"
        squared_error_gain = (
            (effectiveness - effect_prior) ** 2
            - (effectiveness - effect_prediction) ** 2
        )
        bootstrap = _group_bootstrap(
            squared_error_gain,
            frame[group_column],
            salt=protocol_index + 1,
        )
        clauses = {
            "effectiveness_pearson": (
                effect_metrics["pearson"]
                >= GATE_THRESHOLDS["effectiveness_pearson"]
            ),
            "effectiveness_spearman": (
                effect_metrics["spearman"]
                >= GATE_THRESHOLDS["effectiveness_spearman"]
            ),
            "effectiveness_rmse_gain": (
                effect_gain >= GATE_THRESHOLDS["effectiveness_rmse_gain"]
            ),
            "completeness_pearson": (
                complete_metrics["pearson"]
                >= GATE_THRESHOLDS["completeness_pearson"]
            ),
            "completeness_spearman": (
                complete_metrics["spearman"]
                >= GATE_THRESHOLDS["completeness_spearman"]
            ),
            "completeness_rmse_gain": (
                complete_gain >= GATE_THRESHOLDS["completeness_rmse_gain"]
            ),
            "dimension_macro_f1": (
                dimension_macro_f1 >= GATE_THRESHOLDS["dimension_macro_f1"]
            ),
            "dimension_log_loss_gain": (
                prior_loss - dimension_loss
                >= GATE_THRESHOLDS["dimension_log_loss_gain"]
            ),
            "positive_effectiveness_gain_every_fold": all(
                float(row["effectiveness_rmse_gain"]) > 0 for row in fold_rows
            ),
            "positive_completeness_gain_every_fold": all(
                float(row["completeness_rmse_gain"]) > 0 for row in fold_rows
            ),
            "bootstrap_support": (
                bootstrap["support"] >= GATE_THRESHOLDS["bootstrap_support"]
            ),
        }
        protocol_results[protocol] = {
            "effectiveness": effect_metrics,
            "effectiveness_prior_rmse": effect_prior_rmse,
            "effectiveness_rmse_gain": effect_gain,
            "completeness": complete_metrics,
            "completeness_prior_rmse": complete_prior_rmse,
            "completeness_rmse_gain": complete_gain,
            "dimension_macro_f1": dimension_macro_f1,
            "dimension_log_loss": dimension_loss,
            "dimension_prior_log_loss": prior_loss,
            "dimension_log_loss_gain": prior_loss - dimension_loss,
            "effectiveness_bootstrap": bootstrap,
            "folds": fold_rows,
            "clauses": clauses,
            "passes": all(clauses.values()),
        }
        predictions = frame[
            [
                "example_id",
                "source_index",
                "subdimension",
                "dimension",
                "topic",
                "effectiveness_target",
                "completeness_target",
            ]
        ].copy()
        predictions.insert(0, "protocol", protocol)
        predictions["effectiveness_prediction"] = effect_prediction
        predictions["effectiveness_prior"] = effect_prior
        predictions["completeness_prediction"] = complete_prediction
        predictions["completeness_prior"] = complete_prior
        for index, name in enumerate(DIMENSIONS):
            predictions[f"dimension_probability_{index}_{name}"] = (
                dimension_probability[:, index]
            )
        all_prediction_rows.append(predictions)

    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_convolearn_transfer"
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "oof_predictions.parquet"
    pd.concat(all_prediction_rows, ignore_index=True).to_parquet(
        predictions_path, index=False
    )
    model_path = run_dir / "fold_models.npz"
    np.savez_compressed(
        model_path,
        protocols=np.asarray([row["protocol"] for row in all_model_rows]),
        splits=np.asarray([row["split"] for row in all_model_rows]),
        effect_coefficients=np.vstack(
            [row["effect_coef"] for row in all_model_rows]
        ),
        effect_intercepts=np.asarray(
            [row["effect_intercept"] for row in all_model_rows]
        ),
        complete_coefficients=np.vstack(
            [row["complete_coef"] for row in all_model_rows]
        ),
        complete_intercepts=np.asarray(
            [row["complete_intercept"] for row in all_model_rows]
        ),
        dimension_coefficients=np.stack(
            [row["dimension_coef"] for row in all_model_rows]
        ),
        dimension_intercepts=np.stack(
            [row["dimension_intercept"] for row in all_model_rows]
        ),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "rows": len(frame),
        "protocols": protocol_results,
        "thresholds": GATE_THRESHOLDS,
        "passes_external_gate": all(
            result["passes"] for result in protocol_results.values()
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
        description="Run the frozen E660 ConvoLearn pedagogical-quality transfer."
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
