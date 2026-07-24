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


PROTOCOL_ID = "E540_multi_instance_semantic_attention_v1"
PILOT_ROWS = 4_096
PILOT_PROTOCOL = "semantic_k50_s0"
SEGMENT_COUNT = 4
SEGMENT_LINE_CAP = 8
LINE_WORD_CAP = 16
ATTENTION_TEMPERATURE = 10.0
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BENCHMARK_ROWS_PER_POSITION = 32
MAX_PROJECTED_HOURS = 2.0
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
        "segment_texts": paths.cache_dir / "bge_base_e540_segment_texts.parquet",
        "segment_cache": paths.cache_dir / "bge_base_e540_segments_256.npy",
        "progress": paths.cache_dir / "bge_base_e540.progress.json",
        "metadata": paths.cache_dir / "bge_base_e540.metadata.json",
        "benchmark": paths.cache_dir / "bge_base_e540_benchmark.json",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    expected = {**MODEL_HASHES, **SOURCE_HASHES}
    for name, digest in expected.items():
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Missing E540 {name}: {path}")
        observed = _sha256(path)
        if observed != digest:
            raise ValueError(f"E540 {name} SHA-256 changed: {observed}")
    return expected


def _compact_line(value: str) -> str:
    return " ".join(str(value).split()[:LINE_WORD_CAP])


def segment_objective_context(value: str) -> tuple[str, str, str, str]:
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not lines:
        raise ValueError("E540 received an empty objective context.")
    header = lines[0]
    discussion = lines[1:]
    if not discussion:
        discussion = [header]
    segments: list[str] = []
    for partition in np.array_split(np.arange(len(discussion)), SEGMENT_COUNT):
        indices = partition.tolist()
        if len(indices) > SEGMENT_LINE_CAP:
            half = SEGMENT_LINE_CAP // 2
            indices = [*indices[:half], *indices[-half:]]
        selected = [_compact_line(discussion[index]) for index in indices]
        segments.append("\n".join([header, *selected]))
    if len(segments) != SEGMENT_COUNT or any(not value for value in segments):
        raise RuntimeError("E540 segment construction failed.")
    return tuple(segments)  # type: ignore[return-value]


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
        raise ValueError("E540 pilot indices violate the frozen contract.")
    modeling = pd.read_parquet(paths["modeling"]).reset_index(drop=True)
    contexts = pd.read_parquet(
        paths["objective_context"], columns=["response_id", "objective_context"]
    ).reset_index(drop=True)
    if list(modeling["response_id"]) != list(contexts["response_id"]):
        raise ValueError("E540 objective contexts do not align with modeling rows.")
    baseline = pd.read_parquet(paths["baseline"])
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(
        drop=True
    )
    if list(modeling["response_id"]) != list(baseline["response_id"]):
        raise ValueError("E540 baseline rows do not align with modeling rows.")
    frame = modeling.iloc[indices].reset_index(drop=True)
    pilot_contexts = contexts.iloc[indices].reset_index(drop=True)
    folds = baseline.iloc[indices]["fold"].to_numpy(dtype=np.int8)
    if set(folds.tolist()) != set(range(5)) or set(frame["target"]) != {0, 1}:
        raise ValueError("E540 pilot folds or labels are invalid.")
    return frame, pilot_contexts, indices.astype(np.int64), folds


def build_segment_frame(project_root: str | Path) -> pd.DataFrame:
    frame, contexts, _, _ = _load_pilot(project_root)
    segments = [segment_objective_context(value) for value in contexts["objective_context"]]
    result = pd.DataFrame(
        {
            "response_id": frame["response_id"].astype(str),
            **{
                f"segment_{position}": [row[position] for row in segments]
                for position in range(SEGMENT_COUNT)
            },
        }
    )
    if result.shape != (PILOT_ROWS, SEGMENT_COUNT + 1):
        raise RuntimeError(f"E540 segment frame shape changed: {result.shape}")
    if result.drop(columns="response_id").eq("").any().any():
        raise RuntimeError("E540 generated an empty segment.")
    return result


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    _configure_torch()
    verify_sources(project_root)
    paths = _paths(project_root)
    texts = build_segment_frame(project_root)
    model = _load_encoder(project_root)
    position_rows: list[dict[str, object]] = []
    projected_seconds = 0.0
    peak_rss = _current_rss_bytes()
    for position in range(SEGMENT_COUNT):
        values = texts[f"segment_{position}"].tolist()
        lengths = _token_lengths(model, values)
        selected = select_length_quantiles(lengths)
        started = time.perf_counter()
        _encode(model, [values[index] for index in selected])
        elapsed = time.perf_counter() - started
        per_row = elapsed / BENCHMARK_ROWS_PER_POSITION
        projected_seconds += per_row * PILOT_ROWS
        peak_rss = max(peak_rss, _current_rss_bytes())
        position_rows.append(
            {
                "position": position,
                "sample_rows": BENCHMARK_ROWS_PER_POSITION,
                "elapsed_seconds": elapsed,
                "seconds_per_row": per_row,
                "projected_seconds": per_row * PILOT_ROWS,
                "token_min": int(lengths.min()),
                "token_median": float(np.median(lengths)),
                "token_p95": float(np.quantile(lengths, 0.95)),
                "token_max": int(lengths.max()),
            }
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "positions": position_rows,
        "projected_encoded_rows": PILOT_ROWS * SEGMENT_COUNT,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "batch_size": BATCH_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "runtime": runtime,
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    paths["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    path = _paths(project_root)["benchmark"]
    if not path.exists():
        raise FileNotFoundError("Run the frozen E540 benchmark before cache building.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("E540 benchmark protocol is stale.")
    if result.get("source_hashes") != {**MODEL_HASHES, **SOURCE_HASHES}:
        raise ValueError("E540 benchmark source binding is stale.")
    if result.get("proceed") is not True:
        raise RuntimeError("E540 benchmark did not pass its operational gate.")
    return result


def build(project_root: str | Path, *, flush_every: int = 128) -> dict[str, object]:
    if flush_every <= 0:
        raise ValueError("E540 flush interval must be positive.")
    runtime = assert_runtime()
    _configure_torch()
    source_hashes = verify_sources(project_root)
    benchmark_result = _load_benchmark(project_root)
    paths = _paths(project_root)
    if paths["metadata"].exists() and paths["segment_cache"].exists():
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("protocol_id") == PROTOCOL_ID
            and metadata.get("source_hashes") == source_hashes
            and metadata.get("cache_sha256") == _sha256(paths["segment_cache"])
        ):
            return {
                "cache_path": str(paths["segment_cache"]),
                "metadata_path": str(paths["metadata"]),
                "metadata": metadata,
                "reused": True,
            }

    texts = build_segment_frame(project_root)
    texts.to_parquet(paths["segment_texts"], index=False, compression="zstd")
    text_sha = _sha256(paths["segment_texts"])
    progress = [0] * SEGMENT_COUNT
    if paths["progress"].exists():
        state = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if (
            state.get("protocol_id") != PROTOCOL_ID
            or state.get("source_hashes") != source_hashes
            or state.get("segment_text_sha256") != text_sha
        ):
            raise ValueError("E540 progress binding is stale.")
        progress = [int(value) for value in state["completed_rows"]]
    elif paths["segment_cache"].exists():
        raise ValueError("E540 cache exists without bound progress or metadata.")

    expected_shape = (PILOT_ROWS, SEGMENT_COUNT, EMBEDDING_DIMENSION)
    if paths["segment_cache"].exists():
        matrix = np.lib.format.open_memmap(paths["segment_cache"], mode="r+")
        if matrix.shape != expected_shape:
            raise ValueError(f"E540 partial cache shape changed: {matrix.shape}")
    else:
        matrix = np.lib.format.open_memmap(
            paths["segment_cache"],
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
    model = _load_encoder(project_root)
    started = time.perf_counter()
    peak_rss = _current_rss_bytes()
    for position in range(SEGMENT_COUNT):
        values = texts[f"segment_{position}"].tolist()
        for start in range(progress[position], PILOT_ROWS, flush_every):
            stop = min(start + flush_every, PILOT_ROWS)
            matrix[start:stop, position, :] = _encode(model, values[start:stop])
            matrix.flush()
            progress[position] = stop
            peak_rss = max(peak_rss, _current_rss_bytes())
            if peak_rss >= MAX_RSS_BYTES:
                raise MemoryError("E540 cache exceeded 8 GB RSS.")
            paths["progress"].write_text(
                json.dumps(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "source_hashes": source_hashes,
                        "segment_text_sha256": text_sha,
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
                f"Encoded E540 segment={position + 1}/{SEGMENT_COUNT} "
                f"rows={stop}/{PILOT_ROWS}.",
                flush=True,
            )
    del matrix
    completed = np.load(paths["segment_cache"], mmap_mode="r")
    if completed.shape != expected_shape or not np.isfinite(completed).all():
        raise ValueError("E540 completed cache failed shape/finiteness audit.")
    if not np.allclose(
        np.linalg.norm(completed, axis=2), 1.0, atol=2e-5, rtol=0
    ):
        raise ValueError("E540 completed segment embeddings are not normalized.")
    elapsed = time.perf_counter() - started
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "shape": list(expected_shape),
        "segment_count": SEGMENT_COUNT,
        "segment_line_cap": SEGMENT_LINE_CAP,
        "line_word_cap": LINE_WORD_CAP,
        "batch_size": BATCH_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "pooling": "SentenceTransformers CLS pooling plus L2 normalization",
        "segment_text_sha256": text_sha,
        "cache_sha256": _sha256(paths["segment_cache"]),
        "cache_bytes": paths["segment_cache"].stat().st_size,
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
    return {
        "cache_path": str(paths["segment_cache"]),
        "metadata_path": str(paths["metadata"]),
        "metadata": metadata,
        "reused": False,
    }


def semantic_attention(
    segments: np.ndarray,
    objective: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        segments.ndim != 3
        or segments.shape[1:] != (SEGMENT_COUNT, EMBEDDING_DIMENSION)
        or objective.shape != (segments.shape[0], EMBEDDING_DIMENSION)
    ):
        raise ValueError("E540 attention inputs have invalid shapes.")
    similarity = np.einsum("nkd,nd->nk", segments, objective).astype(np.float64)
    logits = similarity * ATTENTION_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    pooled = np.einsum("nk,nkd->nd", weights, segments).astype(np.float32)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("E540 attention pooling is undefined.")
    pooled /= norms
    if (
        not np.isfinite(pooled).all()
        or not np.isfinite(weights).all()
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)
        or not np.allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=2e-5)
    ):
        raise ValueError("E540 attention output failed audit.")
    return pooled, weights, similarity.astype(np.float32)


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
    if not paths["metadata"].exists() or not paths["segment_cache"].exists():
        raise FileNotFoundError("Build the frozen E540 cache before validation.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("source_hashes") != source_hashes
        or metadata.get("cache_sha256") != _sha256(paths["segment_cache"])
    ):
        raise ValueError("E540 cache metadata binding is stale.")

    full = pd.read_parquet(paths["modeling"]).reset_index(drop=True)
    base_features_full, base_similarity_full = prepare_bge_base_features(
        discover_project_paths(project_root).cache_dir, full
    )
    _, dense_full, dense_names = prepare_semantic_features(
        discover_project_paths(project_root).cache_dir, full
    )
    objective = np.load(paths["base_objective"], mmap_mode="r")[indices]
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

    segments = np.load(paths["segment_cache"], mmap_mode="r")
    attended, attention_weights, segment_similarity = semantic_attention(
        segments, objective
    )
    candidate_features, attended_similarity = interaction_features(attended, objective)
    candidate_dense = dense_full[indices].copy()
    candidate_dense[:, -1:] = attended_similarity
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
                    "pred_attention": candidate_prediction,
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_semantic_attention")
    run_dir = discover_project_paths(project_root).experiments_dir / "runs" / run_id
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
        "candidate": "E540_multi_instance_semantic_attention",
        "model": MODEL_NAME,
        "model_license": MODEL_LICENSE,
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "segment_count": SEGMENT_COUNT,
        "segment_line_cap": SEGMENT_LINE_CAP,
        "line_word_cap": LINE_WORD_CAP,
        "attention_temperature": ATTENTION_TEMPERATURE,
        "blend_weights": list(BLEND_WEIGHTS),
        "fixed_regularization_c": REGULARIZATION_C,
        "dense_features": dense_names,
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
        "attention_weight_mean_by_position": attention_weights.mean(axis=0).tolist(),
        "attention_argmax_counts": np.bincount(
            attention_weights.argmax(axis=1), minlength=SEGMENT_COUNT
        ).tolist(),
        "segment_similarity_mean_by_position": segment_similarity.mean(
            axis=0
        ).tolist(),
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
    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passes_screen": passes,
        "report_sha256": _sha256(report_path),
        "artifact_sha256": {
            "metrics": _sha256(metrics_path),
            "predictions": _sha256(predictions_path),
            "fold_metrics": _sha256(fold_path),
            "bootstrap": _sha256(bootstrap_path),
        },
        "report": report,
    }
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E540 multi-instance semantic-attention screen."
    )
    parser.add_argument(
        "command", choices=("benchmark", "build", "validate", "pipeline")
    )
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
