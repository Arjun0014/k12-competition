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
from trace_ace.semantic_cache import compact_objective_context
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)
from trace_ace.source_robust_validation import _current_rss_bytes


MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_DIRECTORY = "bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
MODEL_LICENSE = "MIT"
MODEL_SAFETENSORS_SHA256 = (
    "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
)
MODEL_CONFIG_SHA256 = (
    "446712fac367857b4b1302762fe1cd7bfa8b3c4b77b4dc5d77c4025407660896"
)
MODEL_MODULES_SHA256 = (
    "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
)
MODEL_POOLING_SHA256 = (
    "e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352"
)
PILOT_INDEX_SHA256 = (
    "4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba"
)
MODELING_BASE_SHA256 = (
    "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
)
OBJECTIVE_CONTEXT_SHA256 = (
    "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6"
)
BASELINE_OOF_SHA256 = (
    "1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6"
)
BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
PILOT_ROWS = 4096
EMBEDDING_DIMENSION = 1024
MAX_SEQUENCE_LENGTH = 256
BENCHMARK_ROWS = 32
BATCH_SIZE = 1
CPU_THREADS = 6
MAX_PROJECTED_HOURS = 8.0
MAX_RSS_BYTES = 8 * 1024**3
FIXED_REGULARIZATION_C = 0.1
BOOTSTRAP_REPLICATES = 5000
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


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed: {observed}")


def assert_e450_runtime() -> dict[str, str]:
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
        raise RuntimeError(
            f"E450 runtime differs from the frozen .venv contract: {observed}"
        )
    return observed


def _source_paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    model_dir = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "model": model_dir / "model.safetensors",
        "config": model_dir / "config.json",
        "modules": model_dir / "modules.json",
        "pooling": model_dir / "1_Pooling" / "config.json",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "contexts": paths.cache_dir / "response_objective_context.parquet",
        "baseline": paths.experiments_dir
        / "runs"
        / BASELINE_RUN_ID
        / "oof_predictions.parquet",
    }


def verify_e450_sources(project_root: str | Path) -> dict[str, str]:
    expected = {
        "model": MODEL_SAFETENSORS_SHA256,
        "config": MODEL_CONFIG_SHA256,
        "modules": MODEL_MODULES_SHA256,
        "pooling": MODEL_POOLING_SHA256,
        "indices": PILOT_INDEX_SHA256,
        "modeling": MODELING_BASE_SHA256,
        "contexts": OBJECTIVE_CONTEXT_SHA256,
        "baseline": BASELINE_OOF_SHA256,
    }
    for key, expected_sha in expected.items():
        _verify_file(_source_paths(project_root)[key], expected_sha, f"E450 {key}")
    return expected


def _configure_torch() -> None:
    import torch

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_encoder(project_root: str | Path):
    from sentence_transformers import SentenceTransformer

    paths = discover_project_paths(project_root)
    model_dir = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.float()
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    if int(model.get_embedding_dimension()) != EMBEDDING_DIMENSION:
        raise ValueError("E450 expected BGE-large embedding dimension 1024.")
    pooling = json.loads(
        (model_dir / "1_Pooling" / "config.json").read_text(encoding="utf-8")
    )
    expected_pooling = {
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    if any(pooling.get(key) != value for key, value in expected_pooling.items()):
        raise ValueError(f"E450 pooling contract changed: {pooling}")
    return model


def _load_pilot(
    project_root: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    paths = discover_project_paths(project_root)
    verify_e450_sources(project_root)
    indices = np.load(paths.cache_dir / "qwen3_pilot_indices.npy")
    if indices.shape != (PILOT_ROWS,) or indices.dtype.kind not in "iu":
        raise ValueError("E450 pilot indices have the wrong shape or dtype.")
    if len(np.unique(indices)) != PILOT_ROWS or not np.all(np.diff(indices) > 0):
        raise ValueError("E450 pilot indices must be unique and strictly increasing.")

    modeling = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    if list(modeling["response_id"]) != list(contexts["response_id"]):
        raise ValueError("E450 context rows do not align with modeling rows.")
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
        raise ValueError("E450 baseline rows do not align with modeling rows.")

    frame = modeling.iloc[indices].reset_index(drop=True)
    folds = baseline.iloc[indices]["fold"].to_numpy(dtype=np.int8)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("E450 pilot must cover all five frozen folds.")
    if set(frame["target"].astype(int)) != {0, 1}:
        raise ValueError("E450 pilot must contain both outcome labels.")
    compact_contexts = [
        compact_objective_context(value)
        for value in contexts.iloc[indices]["objective_context"]
    ]
    return frame, indices.astype(np.int64), folds, compact_contexts


def select_benchmark_indices(token_lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(token_lengths)
    if lengths.shape != (PILOT_ROWS,):
        raise ValueError("E450 benchmark token lengths must cover the pilot.")
    order = np.argsort(lengths, kind="mergesort")
    positions = np.rint(np.linspace(0, PILOT_ROWS - 1, BENCHMARK_ROWS)).astype(
        np.int64
    )
    selected = order[positions]
    if len(np.unique(selected)) != BENCHMARK_ROWS:
        raise RuntimeError("E450 benchmark quantiles did not produce unique rows.")
    return selected.astype(np.int64)


def _token_lengths(model, texts: Sequence[str]) -> np.ndarray:
    values = np.empty(len(texts), dtype=np.int32)
    tokenizer = model.tokenizer
    for start in range(0, len(texts), 128):
        end = min(len(texts), start + 128)
        encoded = tokenizer(
            list(texts[start:end]),
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        values[start:end] = [len(row) for row in encoded["input_ids"]]
    return values


def _encode(model, texts: Sequence[str]) -> np.ndarray:
    values = model.encode(
        list(texts),
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"E450 encoder returned invalid shape/content: {values.shape}")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5, rtol=0.0):
        raise ValueError("E450 encoder output is not L2-normalized.")
    return values


def benchmark_bge_large(project_root: str | Path) -> dict[str, object]:
    runtime = assert_e450_runtime()
    paths = discover_project_paths(project_root)
    benchmark_path = paths.cache_dir / "bge_large_e450_benchmark.json"
    if benchmark_path.exists():
        existing = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if existing.get("source_hashes") != verify_e450_sources(project_root):
            raise ValueError("Existing E450 benchmark has different source hashes.")
        return existing

    frame, indices, _, contexts = _load_pilot(project_root)
    _configure_torch()
    model = _load_encoder(project_root)
    token_lengths = _token_lengths(model, contexts)
    selected = select_benchmark_indices(token_lengths)
    selected_texts = [contexts[index] for index in selected]
    rss_before = _current_rss_bytes()
    started = time.perf_counter()
    _encode(model, selected_texts)
    elapsed = time.perf_counter() - started
    peak_rss = max(rss_before, _current_rss_bytes())

    unique_objectives = int(frame["learning_objective"].nunique(dropna=False))
    projected_rows = PILOT_ROWS + unique_objectives
    projected_hours = elapsed / BENCHMARK_ROWS * projected_rows / 3600.0
    proceed = bool(
        projected_hours < MAX_PROJECTED_HOURS and peak_rss < MAX_RSS_BYTES
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E450_bge_large_capacity",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": PILOT_INDEX_SHA256,
        "benchmark_rows": BENCHMARK_ROWS,
        "benchmark_batch_size": BATCH_SIZE,
        "cpu_threads": CPU_THREADS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "selected_pilot_positions": selected.tolist(),
        "selected_untruncated_token_lengths": token_lengths[selected].tolist(),
        "pilot_token_lengths": {
            "minimum": int(token_lengths.min()),
            "median": float(np.median(token_lengths)),
            "maximum": int(token_lengths.max()),
            "at_or_above_limit": int((token_lengths >= MAX_SEQUENCE_LENGTH).sum()),
        },
        "elapsed_seconds": elapsed,
        "seconds_per_row": elapsed / BENCHMARK_ROWS,
        "unique_pilot_objectives": unique_objectives,
        "projected_encoded_rows": projected_rows,
        "projected_hours": projected_hours,
        "peak_rss_bytes": peak_rss,
        "max_projected_hours": MAX_PROJECTED_HOURS,
        "max_rss_bytes": MAX_RSS_BYTES,
        "proceed": proceed,
        "runtime_contract": runtime,
        "source_hashes": verify_e450_sources(project_root),
        "V_final_accessed": False,
    }
    benchmark_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_bge_large_pilot_cache(
    project_root: str | Path, flush_every: int = 32
) -> dict[str, object]:
    if flush_every <= 0:
        raise ValueError("E450 flush interval must be positive.")
    runtime = assert_e450_runtime()
    paths = discover_project_paths(project_root)
    benchmark = benchmark_bge_large(project_root)
    if not benchmark["proceed"]:
        raise RuntimeError("E450 operational benchmark did not pass.")
    if benchmark["benchmark_batch_size"] != BATCH_SIZE:
        raise ValueError("E450 benchmark batch size changed.")

    frame, indices, _, contexts = _load_pilot(project_root)
    _configure_torch()
    model = _load_encoder(project_root)
    context_path = paths.cache_dir / "bge_large_e450_context_256.npy"
    objective_path = paths.cache_dir / "bge_large_e450_objective_256.npy"
    progress_path = paths.cache_dir / "bge_large_e450_context_256.progress.json"
    metadata_path = paths.cache_dir / "bge_large_e450_semantic_256.metadata.json"
    expected_shape = (PILOT_ROWS, EMBEDDING_DIMENSION)

    if context_path.exists():
        context_values = np.lib.format.open_memmap(context_path, mode="r+")
        if context_values.shape != expected_shape:
            raise ValueError("Existing E450 context cache has the wrong shape.")
    else:
        context_values = np.lib.format.open_memmap(
            context_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("model_revision") != MODEL_REVISION
            or progress.get("batch_size") != BATCH_SIZE
            or progress.get("max_sequence_length") != MAX_SEQUENCE_LENGTH
        ):
            raise ValueError("E450 progress file violates the frozen cache contract.")
        completed = int(progress.get("completed_rows", 0))
        if completed < 0 or completed > PILOT_ROWS:
            raise ValueError("E450 progress row count is invalid.")

    peak_rss = _current_rss_bytes()
    started = time.perf_counter()
    for start in range(completed, PILOT_ROWS, flush_every):
        end = min(PILOT_ROWS, start + flush_every)
        context_values[start:end] = _encode(model, contexts[start:end])
        context_values.flush()
        peak_rss = max(peak_rss, _current_rss_bytes())
        if peak_rss >= MAX_RSS_BYTES:
            raise MemoryError("E450 cache exceeded the frozen 8 GB RSS limit.")
        progress_path.write_text(
            json.dumps(
                {
                    "completed_rows": end,
                    "total_rows": PILOT_ROWS,
                    "model": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
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
        print(f"Encoded E450 contexts for {end}/{PILOT_ROWS} rows.", flush=True)

    objectives = frame["learning_objective"].fillna("").astype(str)
    if not objective_path.exists():
        unique_objectives = objectives.drop_duplicates().tolist()
        unique_embeddings = _encode(model, unique_objectives)
        lookup = dict(zip(unique_objectives, unique_embeddings, strict=True))
        objective_values = np.vstack([lookup[value] for value in objectives]).astype(
            np.float32
        )
        np.save(objective_path, objective_values)
        peak_rss = max(peak_rss, _current_rss_bytes())
    elapsed = time.perf_counter() - started

    del context_values
    context = np.load(context_path, mmap_mode="r")
    objective = np.load(objective_path, mmap_mode="r")
    for name, values in (("context", context), ("objective", objective)):
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Completed E450 {name} cache failed audit.")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5, rtol=0.0):
            raise ValueError(f"Completed E450 {name} cache is not normalized.")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E450_bge_large_capacity",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": PILOT_INDEX_SHA256,
        "pilot_response_sha256": hashlib.sha256(
            "\n".join(frame["response_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "pilot_indices_min": int(indices.min()),
        "pilot_indices_max": int(indices.max()),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "pooling": "SentenceTransformers CLS pooling plus L2 normalization",
        "instruction_prefix": None,
        "context_selection": "compact_objective_context",
        "context_cache": context_path.name,
        "context_cache_sha256": _sha256(context_path),
        "context_cache_bytes": context_path.stat().st_size,
        "objective_cache": objective_path.name,
        "objective_cache_sha256": _sha256(objective_path),
        "objective_cache_bytes": objective_path.stat().st_size,
        "unique_pilot_objectives": int(objectives.nunique(dropna=False)),
        "elapsed_seconds_this_invocation": elapsed,
        "peak_rss_bytes": peak_rss,
        "runtime_contract": runtime,
        "source_hashes": verify_e450_sources(project_root),
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress_path.unlink(missing_ok=True)
    return {
        "context_path": str(context_path),
        "objective_path": str(objective_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def interaction_features(
    context: np.ndarray, objective: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if context.shape != objective.shape:
        raise ValueError("E450 context and objective shapes differ.")
    if context.ndim != 2 or context.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("E450 embedding shape is invalid.")
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


def _paired_session_bootstrap(
    frame: pd.DataFrame,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, object]:
    target = frame["target"].to_numpy(dtype=np.float64)
    baseline = np.clip(baseline_probability, 1e-6, 1.0 - 1e-6)
    candidate = np.clip(candidate_probability, 1e-6, 1.0 - 1e-6)
    baseline_loss = -(target * np.log(baseline) + (1 - target) * np.log1p(-baseline))
    candidate_loss = -(
        target * np.log(candidate) + (1 - target) * np.log1p(-candidate)
    )
    gains = pd.DataFrame(
        {
            "session_id": frame["session_id"].astype(str),
            "gain": baseline_loss - candidate_loss,
        }
    )
    grouped = gains.groupby("session_id", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        values[replicate] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "resampler": "session_id",
        "unique_groups": int(len(grouped)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "support_positive_log_loss_gain": float((values > 0.0).mean()),
        "mean_log_loss_gain": float(values.mean()),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def validate_bge_large_pilot(project_root: str | Path) -> dict[str, object]:
    runtime = assert_e450_runtime()
    paths = discover_project_paths(project_root)
    frame, indices, folds, _ = _load_pilot(project_root)
    benchmark = benchmark_bge_large(project_root)
    if not benchmark["proceed"]:
        raise RuntimeError("E450 operational benchmark did not pass.")
    context_path = paths.cache_dir / "bge_large_e450_context_256.npy"
    objective_path = paths.cache_dir / "bge_large_e450_objective_256.npy"
    metadata_path = paths.cache_dir / "bge_large_e450_semantic_256.metadata.json"
    if not context_path.exists() or not objective_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Build and audit the E450 cache before validation.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("context_cache_sha256") != _sha256(context_path):
        raise ValueError("E450 context cache hash differs from metadata.")
    if metadata.get("objective_cache_sha256") != _sha256(objective_path):
        raise ValueError("E450 objective cache hash differs from metadata.")

    full_frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features_full, base_similarity_full = prepare_bge_base_features(
        paths.cache_dir, full_frame
    )
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full_frame)
    base_features = base_features_full[indices]
    dense_base = dense_full[indices].copy()
    dense_base[:, -1:] = base_similarity_full[indices]

    context = np.load(context_path)
    objective = np.load(objective_path)
    candidate_features, candidate_similarity = interaction_features(context, objective)
    dense_candidate = dense_full[indices].copy()
    dense_candidate[:, -1:] = candidate_similarity

    baseline_prediction = semantic_logistic_oof(
        frame,
        folds,
        base_features,
        dense_base,
        regularization_c=FIXED_REGULARIZATION_C,
        n_splits=5,
    )
    candidate_prediction = semantic_logistic_oof(
        frame,
        folds,
        candidate_features,
        dense_candidate,
        regularization_c=FIXED_REGULARIZATION_C,
        n_splits=5,
    )
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline_metrics = binary_metrics(target, baseline_prediction)
    candidate_metrics = binary_metrics(target, candidate_prediction)
    deltas = {
        key: float(candidate_metrics[key] - baseline_metrics[key])
        for key in ("log_loss", "roc_auc", "brier_score", "ece_10", "prediction_mean")
    }
    bootstrap = _paired_session_bootstrap(
        frame, baseline_prediction, candidate_prediction
    )
    loss_path = (
        deltas["log_loss"] <= -0.0015
        and deltas["roc_auc"] >= 0.0
        and deltas["brier_score"] <= 0.0
        and deltas["ece_10"] <= 0.0
    )
    auc_path = (
        deltas["roc_auc"] >= 0.005
        and deltas["log_loss"] <= 0.0
        and deltas["brier_score"] <= 0.0
        and deltas["ece_10"] <= 0.0
    )
    passes_screen = bool(
        (loss_path or auc_path)
        and bootstrap["support_positive_log_loss_gain"] >= 0.90
    )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_bge_large_capacity")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        [
            {"model": "bge_base_fair_comparator", **baseline_metrics},
            {"model": "E450_bge_large_capacity", **candidate_metrics},
        ]
    ).to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "session_id": frame["session_id"],
            "learning_objective_id": frame["learning_objective_id"],
            "fold": folds,
            "target": target,
            "pred_bge_base": baseline_prediction,
            "pred_bge_large": candidate_prediction,
        }
    ).to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame([bootstrap]).to_csv(
        run_dir / "bootstrap.csv", index=False, lineterminator="\n"
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E450_bge_large_capacity",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": PILOT_INDEX_SHA256,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling": metadata["pooling"],
        "interaction": "[0.7*context, 0.7*objective, 6.0*product, 0.7*absolute_difference]",
        "fixed_regularization_c": FIXED_REGULARIZATION_C,
        "dense_features": dense_names,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_baseline": deltas,
        "paired_session_bootstrap": bootstrap,
        "screen_clauses": {
            "loss_path": bool(loss_path),
            "auc_path": bool(auc_path),
            "session_bootstrap_support_at_least_0_90": bool(
                bootstrap["support_positive_log_loss_gain"] >= 0.90
            ),
        },
        "passes_screen": passes_screen,
        "full_competition_cache_built": False,
        "V_final_accessed": False,
        "cache_hashes": {
            "context": metadata["context_cache_sha256"],
            "objective": metadata["objective_cache_sha256"],
        },
        "source_hashes": verify_e450_sources(project_root),
        "runtime_contract": runtime,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered E450 BGE-large capacity screen."
    )
    parser.add_argument(
        "command", choices=("benchmark", "build", "validate", "pipeline")
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--flush-every", type=int, default=32)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "benchmark":
        result = benchmark_bge_large(args.project_root)
    elif args.command == "build":
        result = build_bge_large_pilot_cache(
            args.project_root, flush_every=args.flush_every
        )
    elif args.command == "validate":
        result = validate_bge_large_pilot(args.project_root)
    else:
        benchmark_bge_large(args.project_root)
        build_bge_large_pilot_cache(
            args.project_root, flush_every=args.flush_every
        )
        result = validate_bge_large_pilot(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
