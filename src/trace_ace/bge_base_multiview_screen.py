from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E520_bge_base_multiview_v1"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_DIRECTORY = "bge-base-en-v1.5"
MODEL_LICENSE = "MIT"
MODEL_HASHES = {
    "model": "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
    "config": "bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591",
    "modules": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "pooling": "c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b",
}
SOURCE_HASHES = {
    "multiview": "e7b28d679220672ed379ed0de8f6cef9ece38cbeb7b0473cae11ae80584d4607",
    "indices": "4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba",
    "modeling": "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede",
    "baseline": "1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6",
    "base_context": "b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a",
    "base_objective": "6cc754a287f9ec303aa5f81dd98ab0321c7c5299b2a4e251d6f86f63afe6eb27",
}
BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
PILOT_ROWS = 4_096
EMBEDDING_DIMENSION = 768
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 16
CPU_THREADS = 6
BENCHMARK_ROWS_PER_VIEW = 32
MAX_PROJECTED_HOURS = 2.0
MAX_RSS_BYTES = 8 * 1024**3
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
CANDIDATE_NAMES = ("student", "feedback", "student_feedback_mean")
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260724
EXPECTED_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "scikit_learn": "1.8.0",
    "torch": "2.13.0+cpu",
    "transformers": "5.14.1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing E520 {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"E520 {label} SHA-256 changed: {observed}")


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    baseline = (
        paths.experiments_dir
        / "runs"
        / BASELINE_RUN_ID
        / "oof_predictions.parquet"
    )
    return {
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "modules": model / "modules.json",
        "pooling": model / "1_Pooling" / "config.json",
        "multiview": paths.cache_dir / "response_multiview_texts.parquet",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "baseline": baseline,
        "base_context": paths.cache_dir / "bge_base_context_256.npy",
        "base_objective": paths.cache_dir / "bge_base_objective_256.npy",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    expected = {**MODEL_HASHES, **SOURCE_HASHES}
    for name, digest in expected.items():
        _verify(_paths(project_root)[name], digest, name)
    return expected


def assert_runtime() -> dict[str, str]:
    import sklearn
    import torch
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if observed != EXPECTED_RUNTIME:
        raise RuntimeError(f"E520 runtime differs from frozen .venv: {observed}")
    return observed


def _configure_torch() -> None:
    import torch

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_encoder(project_root: str | Path):
    from sentence_transformers import SentenceTransformer

    root = Path(project_root).resolve()
    model_dir = root / "assets" / "pretrained" / MODEL_DIRECTORY
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.float()
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    if int(model.get_embedding_dimension()) != EMBEDDING_DIMENSION:
        raise ValueError("E520 expected 768-dimensional BGE-base.")
    pooling = json.loads(
        (model_dir / "1_Pooling" / "config.json").read_text(encoding="utf-8")
    )
    if (
        pooling.get("pooling_mode_cls_token") is not True
        or pooling.get("pooling_mode_mean_tokens") is not False
    ):
        raise ValueError(f"E520 pooling contract changed: {pooling}")
    return model


def _load_pilot(
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    paths = discover_project_paths(project_root)
    verify_sources(project_root)
    indices = np.load(paths.cache_dir / "qwen3_pilot_indices.npy")
    if (
        indices.shape != (PILOT_ROWS,)
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != PILOT_ROWS
        or not np.all(np.diff(indices) > 0)
    ):
        raise ValueError("E520 pilot indices violate the frozen contract.")
    modeling = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    texts = pd.read_parquet(
        paths.cache_dir / "response_multiview_texts.parquet"
    ).reset_index(drop=True)
    if list(modeling["response_id"]) != list(texts["response_id"]):
        raise ValueError("E520 multiview rows do not align with modeling rows.")
    if set(texts["view_schema_version"].astype(str)) != {
        "2026-07-17-v2-budgeted"
    }:
        raise ValueError("E520 multiview schema changed.")
    baseline = pd.read_parquet(
        paths.experiments_dir
        / "runs"
        / BASELINE_RUN_ID
        / "oof_predictions.parquet"
    )
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(
        drop=True
    )
    if list(modeling["response_id"]) != list(baseline["response_id"]):
        raise ValueError("E520 baseline rows do not align with modeling rows.")
    frame = modeling.iloc[indices].reset_index(drop=True)
    pilot_texts = texts.iloc[indices].reset_index(drop=True)
    folds = baseline.iloc[indices]["fold"].to_numpy(dtype=np.int8)
    if set(folds.tolist()) != set(range(5)) or set(frame["target"]) != {0, 1}:
        raise ValueError("E520 pilot folds or labels are invalid.")
    return frame, pilot_texts, indices.astype(np.int64), folds


def _encode(model, texts: Sequence[str]) -> np.ndarray:
    values = model.encode(
        list(texts),
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    if values.shape != (len(texts), EMBEDDING_DIMENSION):
        raise ValueError(f"E520 encoder shape is invalid: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("E520 encoder returned non-finite values.")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-5, rtol=0):
        raise ValueError("E520 encoder output is not normalized.")
    return values


def _token_lengths(model, texts: Sequence[str]) -> np.ndarray:
    values = np.empty(len(texts), dtype=np.int32)
    for start in range(0, len(texts), 128):
        stop = min(start + 128, len(texts))
        encoded = model.tokenizer(
            list(texts[start:stop]),
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        values[start:stop] = [len(row) for row in encoded["input_ids"]]
    return values


def select_length_quantiles(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths)
    if lengths.shape != (PILOT_ROWS,):
        raise ValueError("E520 length quantiles require every pilot row.")
    order = np.argsort(lengths, kind="mergesort")
    positions = np.rint(
        np.linspace(0, PILOT_ROWS - 1, BENCHMARK_ROWS_PER_VIEW)
    ).astype(np.int64)
    selected = order[positions]
    if len(np.unique(selected)) != BENCHMARK_ROWS_PER_VIEW:
        raise RuntimeError("E520 benchmark quantiles are not unique.")
    return selected


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    output = paths.cache_dir / "bge_base_multiview_e520_benchmark.json"
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("source_hashes") != verify_sources(project_root):
            raise ValueError("Existing E520 benchmark source hashes changed.")
        return result
    _, texts, _, _ = _load_pilot(project_root)
    _configure_torch()
    model = _load_encoder(project_root)
    rows: dict[str, dict[str, object]] = {}
    total_elapsed = 0.0
    peak_rss = _current_rss_bytes()
    for name, column in (
        ("student", "student_evidence"),
        ("feedback", "feedback_evidence"),
    ):
        values = texts[column].astype(str).tolist()
        lengths = _token_lengths(model, values)
        selected = select_length_quantiles(lengths)
        started = time.perf_counter()
        _encode(model, [values[index] for index in selected])
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        peak_rss = max(peak_rss, _current_rss_bytes())
        rows[name] = {
            "elapsed_seconds": elapsed,
            "seconds_per_row": elapsed / BENCHMARK_ROWS_PER_VIEW,
            "selected_positions": selected.tolist(),
            "selected_token_lengths": lengths[selected].tolist(),
            "token_length_min": int(lengths.min()),
            "token_length_median": float(np.median(lengths)),
            "token_length_max": int(lengths.max()),
            "at_or_above_limit": int((lengths >= MAX_SEQUENCE_LENGTH).sum()),
        }
    projected_hours = (
        total_elapsed
        / (2 * BENCHMARK_ROWS_PER_VIEW)
        * (2 * PILOT_ROWS)
        / 3600.0
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_rows_per_view": BENCHMARK_ROWS_PER_VIEW,
        "batch_size": BATCH_SIZE,
        "cpu_threads": CPU_THREADS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "views": rows,
        "projected_encoded_rows": 2 * PILOT_ROWS,
        "projected_hours": projected_hours,
        "peak_rss_bytes": peak_rss,
        "max_projected_hours": MAX_PROJECTED_HOURS,
        "max_rss_bytes": MAX_RSS_BYTES,
        "proceed": bool(
            projected_hours < MAX_PROJECTED_HOURS and peak_rss < MAX_RSS_BYTES
        ),
        "runtime": runtime,
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build(project_root: str | Path, flush_every: int = 128) -> dict[str, object]:
    if flush_every <= 0:
        raise ValueError("E520 flush interval must be positive.")
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    benchmark_result = benchmark(project_root)
    if not benchmark_result["proceed"]:
        raise RuntimeError("E520 operational benchmark failed.")
    frame, texts, indices, _ = _load_pilot(project_root)
    _configure_torch()
    model = _load_encoder(project_root)
    cache_paths = {
        "student": paths.cache_dir / "bge_base_e520_student_256.npy",
        "feedback": paths.cache_dir / "bge_base_e520_feedback_256.npy",
    }
    progress_path = paths.cache_dir / "bge_base_e520.progress.json"
    metadata_path = paths.cache_dir / "bge_base_e520.metadata.json"
    progress = {"student": 0, "feedback": 0}
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            saved.get("protocol_id") != PROTOCOL_ID
            or saved.get("batch_size") != BATCH_SIZE
        ):
            raise ValueError("E520 progress contract changed.")
        progress.update(
            {name: int(saved.get("completed", {}).get(name, 0)) for name in progress}
        )
    peak_rss = _current_rss_bytes()
    started = time.perf_counter()
    for name, column in (
        ("student", "student_evidence"),
        ("feedback", "feedback_evidence"),
    ):
        path = cache_paths[name]
        if path.exists():
            matrix = np.lib.format.open_memmap(path, mode="r+")
            if matrix.shape != (PILOT_ROWS, EMBEDDING_DIMENSION):
                raise ValueError(f"E520 {name} cache shape changed.")
        else:
            matrix = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(PILOT_ROWS, EMBEDDING_DIMENSION),
            )
        values = texts[column].astype(str).tolist()
        for start in range(progress[name], PILOT_ROWS, flush_every):
            stop = min(start + flush_every, PILOT_ROWS)
            matrix[start:stop] = _encode(model, values[start:stop])
            matrix.flush()
            progress[name] = stop
            peak_rss = max(peak_rss, _current_rss_bytes())
            if peak_rss >= MAX_RSS_BYTES:
                raise MemoryError("E520 cache exceeded 8 GB RSS.")
            progress_path.write_text(
                json.dumps(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "completed": progress,
                        "batch_size": BATCH_SIZE,
                        "max_sequence_length": MAX_SEQUENCE_LENGTH,
                        "peak_rss_bytes": peak_rss,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Encoded E520 {name} for {stop}/{PILOT_ROWS}.", flush=True)
        del matrix
    elapsed = time.perf_counter() - started
    for name, path in cache_paths.items():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape != (PILOT_ROWS, EMBEDDING_DIMENSION):
            raise ValueError(f"E520 {name} completed cache shape is invalid.")
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix, axis=1), 1.0, atol=2e-5, rtol=0
        ):
            raise ValueError(f"E520 {name} completed cache failed audit.")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": SOURCE_HASHES["indices"],
        "pilot_response_sha256": hashlib.sha256(
            "\n".join(frame["response_id"].astype(str)).encode()
        ).hexdigest(),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "pooling": "SentenceTransformers CLS pooling plus L2 normalization",
        "view_schema": "2026-07-17-v2-budgeted",
        "cache_hashes": {
            name: _sha256(path) for name, path in cache_paths.items()
        },
        "cache_bytes": {
            name: path.stat().st_size for name, path in cache_paths.items()
        },
        "elapsed_seconds_this_invocation": elapsed,
        "peak_rss_bytes": peak_rss,
        "runtime": runtime,
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress_path.unlink(missing_ok=True)
    return {
        "cache_paths": {name: str(path) for name, path in cache_paths.items()},
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def normalized_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("E520 mean-view inputs must have matching matrices.")
    values = np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("E520 mean-view normalization is undefined.")
    return (values / norms).astype(np.float32)


def interaction_features(
    context: np.ndarray, objective: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        context.shape != objective.shape
        or context.ndim != 2
        or context.shape[1] != EMBEDDING_DIMENSION
    ):
        raise ValueError("E520 interaction embedding shape is invalid.")
    features = np.column_stack(
        [
            context * np.float32(0.7),
            objective * np.float32(0.7),
            context * objective * np.float32(6.0),
            np.abs(context - objective) * np.float32(0.7),
        ]
    ).astype(np.float32)
    similarity = np.sum(context * objective, axis=1, keepdims=True).astype(
        np.float32
    )
    return features, similarity


def _session_bootstrap(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    target = frame["target"].to_numpy(dtype=np.float64)
    baseline = np.clip(baseline, 1e-6, 1 - 1e-6)
    candidate = np.clip(candidate, 1e-6, 1 - 1e-6)
    baseline_loss = -(
        target * np.log(baseline) + (1 - target) * np.log1p(-baseline)
    )
    candidate_loss = -(
        target * np.log(candidate) + (1 - target) * np.log1p(-candidate)
    )
    grouped = (
        pd.DataFrame(
            {
                "session_id": frame["session_id"].astype(str),
                "gain": baseline_loss - candidate_loss,
            }
        )
        .groupby("session_id", sort=True)["gain"]
        .agg(["sum", "count"])
    )
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(n_replicates, dtype=np.float64)
    batch = 64
    for start in range(0, n_replicates, batch):
        stop = min(start + batch, n_replicates)
        sample = rng.integers(0, len(grouped), size=(stop - start, len(grouped)))
        values[start:stop] = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return {
        "resampler": "session_id",
        "unique_groups": int(len(grouped)),
        "replicates": int(n_replicates),
        "seed": BOOTSTRAP_SEED,
        "support_positive_log_loss_gain": float(np.mean(values > 0)),
        "mean_log_loss_gain": float(values.mean()),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def validate(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    frame, _, indices, folds = _load_pilot(project_root)
    benchmark_result = benchmark(project_root)
    if not benchmark_result["proceed"]:
        raise RuntimeError("E520 benchmark did not pass.")
    metadata_path = paths.cache_dir / "bge_base_e520.metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Build E520 caches before validation.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_paths = {
        "student": paths.cache_dir / "bge_base_e520_student_256.npy",
        "feedback": paths.cache_dir / "bge_base_e520_feedback_256.npy",
    }
    for name, path in cache_paths.items():
        if metadata["cache_hashes"].get(name) != _sha256(path):
            raise ValueError(f"E520 {name} cache hash differs from metadata.")
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features_full, base_similarity_full = prepare_bge_base_features(
        paths.cache_dir, full
    )
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    objective = np.load(paths.cache_dir / "bge_base_objective_256.npy")[indices]
    base_dense = dense_full[indices].copy()
    base_dense[:, -1:] = base_similarity_full[indices]
    baseline_prediction = semantic_logistic_oof(
        frame,
        folds,
        base_features_full[indices],
        base_dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    student = np.load(cache_paths["student"])
    feedback = np.load(cache_paths["feedback"])
    contexts = {
        "student": student,
        "feedback": feedback,
        "student_feedback_mean": normalized_mean(student, feedback),
    }
    candidate_predictions: dict[str, np.ndarray] = {}
    for name in CANDIDATE_NAMES:
        features, similarity = interaction_features(contexts[name], objective)
        dense = dense_full[indices].copy()
        dense[:, -1:] = similarity
        candidate_predictions[name] = semantic_logistic_oof(
            frame,
            folds,
            features,
            dense,
            regularization_c=REGULARIZATION_C,
            n_splits=5,
        )
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline_metrics = binary_metrics(target, baseline_prediction)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for name in CANDIDATE_NAMES:
        standalone = candidate_predictions[name]
        for weight in BLEND_WEIGHTS:
            prediction = (1 - weight) * baseline_prediction + weight * standalone
            metrics = binary_metrics(target, prediction)
            metric_rows.append(
                {
                    "candidate": name,
                    "blend_weight": weight,
                    **metrics,
                    **{
                        f"delta_{metric}_vs_bge": (
                            metrics[metric] - baseline_metrics[metric]
                        )
                        for metric in (
                            "log_loss",
                            "roc_auc",
                            "brier_score",
                            "ece_10",
                            "prediction_mean",
                        )
                    },
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "response_id": frame["response_id"],
                        "session_id": frame["session_id"],
                        "learning_objective_id": frame["learning_objective_id"],
                        "fold": folds,
                        "target": target,
                        "candidate": name,
                        "blend_weight": weight,
                        "pred_bge_base": baseline_prediction,
                        "pred_view": standalone,
                        "prediction": prediction,
                    }
                )
            )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["log_loss", "roc_auc", "candidate", "blend_weight"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    selected = metrics.iloc[0].to_dict()
    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_predictions = all_predictions.loc[
        all_predictions["candidate"].eq(selected["candidate"])
        & all_predictions["blend_weight"].eq(selected["blend_weight"])
    ].reset_index(drop=True)
    bootstrap_result = _session_bootstrap(
        frame,
        selected_predictions["pred_bge_base"].to_numpy(),
        selected_predictions["prediction"].to_numpy(),
    )
    loss_path = bool(
        selected["delta_log_loss_vs_bge"] <= -0.0015
        and selected["delta_roc_auc_vs_bge"] >= 0
        and selected["delta_brier_score_vs_bge"] <= 0
        and selected["delta_ece_10_vs_bge"] <= 0
    )
    auc_path = bool(
        selected["delta_roc_auc_vs_bge"] >= 0.005
        and selected["delta_log_loss_vs_bge"] <= 0
        and selected["delta_brier_score_vs_bge"] <= 0
        and selected["delta_ece_10_vs_bge"] <= 0
    )
    passes = bool(
        (loss_path or auc_path)
        and bootstrap_result["support_positive_log_loss_gain"] >= 0.90
    )
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_bge_base_multiview")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    all_predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame([bootstrap_result]).to_csv(
        run_dir / "bootstrap.csv", index=False, lineterminator="\n"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate_views": list(CANDIDATE_NAMES),
        "blend_weights": list(BLEND_WEIGHTS),
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": SOURCE_HASHES["indices"],
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "fixed_regularization_c": REGULARIZATION_C,
        "dense_features": dense_names,
        "baseline_metrics": baseline_metrics,
        "selected": selected,
        "paired_session_bootstrap": bootstrap_result,
        "screen_clauses": {
            "loss_path": loss_path,
            "auc_path": auc_path,
            "session_bootstrap_support_at_least_0_90": (
                bootstrap_result["support_positive_log_loss_gain"] >= 0.90
            ),
        },
        "passes_screen": passes,
        "full_competition_cache_built": False,
        "benchmark": benchmark_result,
        "cache_hashes": metadata["cache_hashes"],
        "source_hashes": verify_sources(project_root),
        "runtime": runtime,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E520 BGE-base multi-view screen."
    )
    parser.add_argument("command", choices=("benchmark", "build", "validate", "pipeline"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--flush-every", type=int, default=128)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "benchmark":
        result = benchmark(args.project_root)
    elif args.command == "build":
        result = build(args.project_root, flush_every=args.flush_every)
    elif args.command == "validate":
        result = validate(args.project_root)
    else:
        benchmark(args.project_root)
        build(args.project_root, flush_every=args.flush_every)
        result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
