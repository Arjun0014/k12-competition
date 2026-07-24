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
    MAX_SEQUENCE_LENGTH,
    MODEL_DIRECTORY,
    MODEL_HASHES,
    MODEL_LICENSE,
    MODEL_NAME,
    _configure_torch,
    _current_rss_bytes,
    _encode,
    _load_encoder,
    _session_bootstrap,
    _token_lengths,
    assert_runtime,
    interaction_features,
    select_length_quantiles,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


PROTOCOL_ID = "E550_bge_base_dual_pooling_v1"
PILOT_ROWS = 4_096
PILOT_PROTOCOL = "semantic_k50_s0"
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BENCHMARK_ROWS_PER_VIEW = 32
MAX_PROJECTED_HOURS = 2.0
CLS_PARITY_ATOL = 2e-6
BOOTSTRAP_REPLICATES = 5_000
SOURCE_HASHES = {
    "objective_context": "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6",
    "indices": "4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba",
    "modeling": "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede",
    "baseline": "1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6",
    "base_context": "b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a",
    "base_objective": "6cc754a287f9ec303aa5f81dd98ab0321c7c5299b2a4e251d6f86f63afe6eb27",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "modules": model / "modules.json",
        "pooling": model / "1_Pooling" / "config.json",
        "objective_context": paths.cache_dir / "response_objective_context.parquet",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "baseline": (
            paths.experiments_dir
            / "runs"
            / "20260716T183434Z_robust_validation"
            / "oof_predictions.parquet"
        ),
        "base_context": paths.cache_dir / "bge_base_context_256.npy",
        "base_objective": paths.cache_dir / "bge_base_objective_256.npy",
        "context_mean": paths.cache_dir / "bge_base_e550_context_mean_256.npy",
        "objective_mean": paths.cache_dir / "bge_base_e550_objective_mean_256.npy",
        "progress": paths.cache_dir / "bge_base_e550.progress.json",
        "metadata": paths.cache_dir / "bge_base_e550.metadata.json",
        "benchmark": paths.cache_dir / "bge_base_e550_benchmark.json",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    expected = {**MODEL_HASHES, **SOURCE_HASHES}
    for name, digest in expected.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing E550 {name}: {path}")
        observed = _sha256(path)
        if observed != digest:
            raise ValueError(f"E550 {name} SHA-256 changed: {observed}")
    return expected


def _load_pilot(
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    verify_sources(project_root)
    paths = _paths(project_root)
    indices = np.load(paths["indices"])
    if (
        indices.shape != (PILOT_ROWS,)
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != PILOT_ROWS
        or not np.all(np.diff(indices) > 0)
    ):
        raise ValueError("E550 pilot indices violate the frozen contract.")
    modeling = pd.read_parquet(paths["modeling"]).reset_index(drop=True)
    contexts = pd.read_parquet(
        paths["objective_context"], columns=["response_id", "objective_context"]
    ).reset_index(drop=True)
    if list(modeling["response_id"]) != list(contexts["response_id"]):
        raise ValueError("E550 objective contexts do not align with modeling rows.")
    baseline = pd.read_parquet(paths["baseline"])
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(
        drop=True
    )
    if list(modeling["response_id"]) != list(baseline["response_id"]):
        raise ValueError("E550 baseline rows do not align with modeling rows.")
    frame = modeling.iloc[indices].reset_index(drop=True)
    texts = pd.DataFrame(
        {
            "response_id": frame["response_id"].astype(str),
            "context": contexts.iloc[indices]["objective_context"].astype(str).tolist(),
            "objective": frame["learning_objective"].astype(str),
        }
    )
    folds = baseline.iloc[indices]["fold"].to_numpy(dtype=np.int8)
    if set(folds.tolist()) != set(range(5)) or set(frame["target"]) != {0, 1}:
        raise ValueError("E550 pilot folds or labels are invalid.")
    if texts[["context", "objective"]].eq("").any().any():
        raise ValueError("E550 received empty pilot text.")
    return frame, texts, indices.astype(np.int64), folds


def masked_mean_pool(
    last_hidden_state: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    hidden = np.asarray(last_hidden_state, dtype=np.float32)
    mask = np.asarray(attention_mask)
    if (
        hidden.ndim != 3
        or mask.shape != hidden.shape[:2]
        or hidden.shape[2] != EMBEDDING_DIMENSION
    ):
        raise ValueError("E550 masked-mean inputs have invalid shapes.")
    weights = mask.astype(np.float32)[..., None]
    counts = weights.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("E550 cannot pool an empty attention mask.")
    pooled = (hidden * weights).sum(axis=1) / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("E550 masked-mean normalization is undefined.")
    pooled /= norms
    return pooled.astype(np.float32)


def _encode_dual(model, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    values = list(texts)
    if not values:
        return (
            np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32),
            np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32),
        )
    transformer = model[0]
    auto_model = transformer.auto_model
    auto_model.eval()
    cls_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), BATCH_SIZE):
            encoded = model.tokenizer(
                values[start : start + BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
            )
            output = auto_model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            cls = functional.normalize(hidden[:, 0, :], p=2, dim=1)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            mean = functional.normalize(mean, p=2, dim=1)
            cls_rows.append(cls.cpu().numpy().astype(np.float32))
            mean_rows.append(mean.cpu().numpy().astype(np.float32))
    cls_values = np.concatenate(cls_rows)
    mean_values = np.concatenate(mean_rows)
    expected = (len(values), EMBEDDING_DIMENSION)
    if cls_values.shape != expected or mean_values.shape != expected:
        raise ValueError("E550 raw encoder returned an invalid shape.")
    if not np.isfinite(cls_values).all() or not np.isfinite(mean_values).all():
        raise ValueError("E550 raw encoder returned non-finite values.")
    if (
        not np.allclose(np.linalg.norm(cls_values, axis=1), 1.0, atol=2e-5, rtol=0)
        or not np.allclose(
            np.linalg.norm(mean_values, axis=1), 1.0, atol=2e-5, rtol=0
        )
    ):
        raise ValueError("E550 encoder outputs are not normalized.")
    return cls_values, mean_values


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    _configure_torch()
    source_hashes = verify_sources(project_root)
    paths = _paths(project_root)
    _, texts, _, _ = _load_pilot(project_root)
    model = _load_encoder(project_root)
    view_rows: list[dict[str, object]] = []
    projected_seconds = 0.0
    parity_max = 0.0
    peak_rss = _current_rss_bytes()
    for view in ("context", "objective"):
        values = texts[view].tolist()
        lengths = _token_lengths(model, values)
        selected = select_length_quantiles(lengths)
        selected_texts = [values[index] for index in selected]
        started = time.perf_counter()
        raw_cls, _ = _encode_dual(model, selected_texts)
        elapsed = time.perf_counter() - started
        packaged_cls = _encode(model, selected_texts)
        view_parity = float(np.max(np.abs(raw_cls - packaged_cls)))
        parity_max = max(parity_max, view_parity)
        per_row = elapsed / BENCHMARK_ROWS_PER_VIEW
        projected_seconds += per_row * PILOT_ROWS
        peak_rss = max(peak_rss, _current_rss_bytes())
        view_rows.append(
            {
                "view": view,
                "sample_rows": BENCHMARK_ROWS_PER_VIEW,
                "elapsed_seconds": elapsed,
                "seconds_per_row": per_row,
                "projected_seconds": per_row * PILOT_ROWS,
                "token_min": int(lengths.min()),
                "token_median": float(np.median(lengths)),
                "token_p95": float(np.quantile(lengths, 0.95)),
                "token_max": int(lengths.max()),
                "cls_parity_max_abs": view_parity,
            }
        )
    parity_pass = bool(parity_max <= CLS_PARITY_ATOL)
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "views": view_rows,
        "projected_encoded_rows": 2 * PILOT_ROWS,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "peak_rss_bytes": peak_rss,
        "cls_parity_atol": CLS_PARITY_ATOL,
        "cls_parity_max_abs": parity_max,
        "cls_parity_pass": parity_pass,
        "proceed": bool(
            parity_pass
            and projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "batch_size": BATCH_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "V_final_accessed": False,
    }
    paths["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    path = _paths(project_root)["benchmark"]
    if not path.exists():
        raise FileNotFoundError("Run the frozen E550 benchmark before cache building.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("E550 benchmark protocol is stale.")
    if result.get("source_hashes") != {**MODEL_HASHES, **SOURCE_HASHES}:
        raise ValueError("E550 benchmark source binding is stale.")
    if result.get("proceed") is not True:
        raise RuntimeError("E550 benchmark did not pass its operational gate.")
    return result


def build(project_root: str | Path, *, flush_every: int = 128) -> dict[str, object]:
    if flush_every <= 0:
        raise ValueError("E550 flush interval must be positive.")
    runtime = assert_runtime()
    _configure_torch()
    source_hashes = verify_sources(project_root)
    benchmark_result = _load_benchmark(project_root)
    paths = _paths(project_root)
    cache_paths = {
        "context": paths["context_mean"],
        "objective": paths["objective_mean"],
    }
    if paths["metadata"].exists() and all(path.exists() for path in cache_paths.values()):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("protocol_id") == PROTOCOL_ID
            and metadata.get("source_hashes") == source_hashes
            and all(
                metadata["cache_sha256"].get(name) == _sha256(path)
                for name, path in cache_paths.items()
            )
        ):
            return {"metadata": metadata, "reused": True}

    _, texts, _, _ = _load_pilot(project_root)
    progress = {"context": 0, "objective": 0}
    if paths["progress"].exists():
        state = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if (
            state.get("protocol_id") != PROTOCOL_ID
            or state.get("source_hashes") != source_hashes
        ):
            raise ValueError("E550 progress binding is stale.")
        progress = {name: int(state["completed_rows"][name]) for name in progress}
    elif any(path.exists() for path in cache_paths.values()):
        raise ValueError("E550 cache exists without bound progress or metadata.")

    expected_shape = (PILOT_ROWS, EMBEDDING_DIMENSION)
    matrices: dict[str, np.memmap] = {}
    for name, path in cache_paths.items():
        if path.exists():
            matrix = np.lib.format.open_memmap(path, mode="r+")
            if matrix.shape != expected_shape:
                raise ValueError(f"E550 {name} partial cache shape changed.")
        else:
            matrix = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=expected_shape
            )
        matrices[name] = matrix

    model = _load_encoder(project_root)
    started = time.perf_counter()
    peak_rss = _current_rss_bytes()
    for name in ("context", "objective"):
        values = texts[name].tolist()
        for start in range(progress[name], PILOT_ROWS, flush_every):
            stop = min(start + flush_every, PILOT_ROWS)
            _, mean = _encode_dual(model, values[start:stop])
            matrices[name][start:stop] = mean
            matrices[name].flush()
            progress[name] = stop
            peak_rss = max(peak_rss, _current_rss_bytes())
            if peak_rss >= MAX_RSS_BYTES:
                raise MemoryError("E550 cache exceeded 8 GB RSS.")
            paths["progress"].write_text(
                json.dumps(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "source_hashes": source_hashes,
                        "completed_rows": progress,
                        "peak_rss_bytes": peak_rss,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"Encoded E550 {name} rows={stop}/{PILOT_ROWS}.",
                flush=True,
            )
    del matrices
    for name, path in cache_paths.items():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f"E550 completed {name} cache failed audit.")
        if not np.allclose(
            np.linalg.norm(matrix, axis=1), 1.0, atol=2e-5, rtol=0
        ):
            raise ValueError(f"E550 completed {name} cache is not normalized.")
    elapsed = time.perf_counter() - started
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "shape": list(expected_shape),
        "pooling": "L2-normalized final-hidden-state attention-mask mean",
        "batch_size": BATCH_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "cache_sha256": {
            name: _sha256(path) for name, path in cache_paths.items()
        },
        "cache_bytes": {
            name: path.stat().st_size for name, path in cache_paths.items()
        },
        "elapsed_seconds_this_invocation": elapsed,
        "peak_rss_bytes": peak_rss,
        "benchmark": benchmark_result,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "V_final_accessed": False,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["progress"].unlink(missing_ok=True)
    return {"metadata": metadata, "reused": False}


def dual_interaction_features(
    cls_context: np.ndarray,
    cls_objective: np.ndarray,
    mean_context: np.ndarray,
    mean_objective: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cls_features, _ = interaction_features(cls_context, cls_objective)
    mean_features, mean_similarity = interaction_features(mean_context, mean_objective)
    scale = np.float32(1.0 / math.sqrt(2.0))
    features = np.column_stack([cls_features * scale, mean_features * scale])
    if not np.isfinite(features).all():
        raise ValueError("E550 dual interaction features are non-finite.")
    return features.astype(np.float32), mean_similarity.astype(np.float32)


def _fold_metrics(
    frame: pd.DataFrame,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target = frame["target"].to_numpy(dtype=np.int8)
    for fold in range(5):
        mask = folds == fold
        base_metrics = binary_metrics(target[mask], baseline[mask])
        candidate_metrics = binary_metrics(target[mask], candidate[mask])
        rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                **{f"bge_{key}": value for key, value in base_metrics.items()},
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                **{
                    f"delta_{key}_vs_bge": candidate_metrics[key]
                    - base_metrics[key]
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
    paths = _paths(project_root)
    frame, _, indices, folds = _load_pilot(project_root)
    if not paths["metadata"].exists():
        raise FileNotFoundError("Build the frozen E550 cache before validation.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    cache_paths = {
        "context": paths["context_mean"],
        "objective": paths["objective_mean"],
    }
    if (
        metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("source_hashes") != source_hashes
        or any(
            metadata["cache_sha256"].get(name) != _sha256(path)
            for name, path in cache_paths.items()
        )
    ):
        raise ValueError("E550 cache metadata binding is stale.")

    project_paths = discover_project_paths(project_root)
    full = pd.read_parquet(paths["modeling"]).reset_index(drop=True)
    base_features_full, base_similarity_full = prepare_bge_base_features(
        project_paths.cache_dir, full
    )
    _, dense_full, dense_names = prepare_semantic_features(
        project_paths.cache_dir, full
    )
    cls_context = np.load(paths["base_context"], mmap_mode="r")[indices]
    cls_objective = np.load(paths["base_objective"], mmap_mode="r")[indices]
    mean_context = np.load(paths["context_mean"], mmap_mode="r")
    mean_objective = np.load(paths["objective_mean"], mmap_mode="r")

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

    candidate_features, mean_similarity = dual_interaction_features(
        cls_context, cls_objective, mean_context, mean_objective
    )
    candidate_dense = np.column_stack([base_dense, mean_similarity])
    candidate_prediction = semantic_logistic_oof(
        frame,
        folds,
        candidate_features,
        candidate_dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline_metrics = binary_metrics(target, baseline_prediction)
    metric_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for weight in BLEND_WEIGHTS:
        prediction = (
            (1.0 - weight) * baseline_prediction + weight * candidate_prediction
        )
        metrics = binary_metrics(target, prediction)
        metric_rows.append(
            {
                "blend_weight": weight,
                **metrics,
                **{
                    f"delta_{key}_vs_bge": metrics[key] - baseline_metrics[key]
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
        predictions.append(
            pd.DataFrame(
                {
                    "response_id": frame["response_id"],
                    "session_id": frame["session_id"],
                    "learning_objective_id": frame["learning_objective_id"],
                    "fold": folds,
                    "target": target,
                    "blend_weight": weight,
                    "pred_bge_base": baseline_prediction,
                    "pred_dual_pooling": candidate_prediction,
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
    all_predictions = pd.concat(predictions, ignore_index=True)
    selected_predictions = all_predictions.loc[
        all_predictions["blend_weight"].eq(selected["blend_weight"])
    ].reset_index(drop=True)
    bootstrap = _session_bootstrap(
        frame,
        selected_predictions["pred_bge_base"].to_numpy(),
        selected_predictions["prediction"].to_numpy(),
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    fold_metrics = _fold_metrics(
        frame,
        folds,
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
        and bootstrap["support_positive_log_loss_gain"] >= 0.90
    )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_bge_base_dual_pooling")
    run_dir = project_paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "oof_predictions.parquet"
    fold_path = run_dir / "fold_metrics.csv"
    bootstrap_path = run_dir / "bootstrap.csv"
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    all_predictions.to_parquet(predictions_path, index=False)
    fold_metrics.to_csv(fold_path, index=False, lineterminator="\n")
    pd.DataFrame([bootstrap]).to_csv(
        bootstrap_path, index=False, lineterminator="\n"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E550_bge_base_dual_pooling",
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "pooling": metadata["pooling"],
        "dual_block_scale": 1.0 / math.sqrt(2.0),
        "blend_weights": list(BLEND_WEIGHTS),
        "fixed_regularization_c": REGULARIZATION_C,
        "dense_features": [*dense_names, "bge_mean_objective_context_cosine"],
        "baseline_metrics": baseline_metrics,
        "selected": selected,
        "fold_log_loss_changes": fold_metrics["delta_log_loss_vs_bge"].tolist(),
        "paired_session_bootstrap": bootstrap,
        "screen_clauses": {
            "loss_path": loss_path,
            "auc_path": auc_path,
            "session_bootstrap_support_at_least_0_90": (
                bootstrap["support_positive_log_loss_gain"] >= 0.90
            ),
        },
        "passes_screen": passes,
        "mean_pool_cosine_mean": float(np.mean(mean_similarity)),
        "benchmark": benchmark_result,
        "cache_sha256": metadata["cache_sha256"],
        "source_hashes": source_hashes,
        "runtime": runtime,
        "validation_runtime_seconds": time.perf_counter() - started,
        "full_competition_cache_built": False,
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
        description="Run the frozen E550 BGE-base dual-pooling screen."
    )
    parser.add_argument(
        "stage", choices=("benchmark", "build", "validate", "pipeline")
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--flush-every", type=int, default=128)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build":
        result = build(args.project_root, flush_every=args.flush_every)
    elif args.stage == "validate":
        result = validate(args.project_root)
    else:
        benchmark_result = benchmark(args.project_root)
        if not benchmark_result["proceed"]:
            raise RuntimeError("E550 benchmark failed; cache build is prohibited.")
        build_result = build(args.project_root, flush_every=args.flush_every)
        result = {
            "benchmark": benchmark_result,
            "build": build_result,
            "validation": validate(args.project_root),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
