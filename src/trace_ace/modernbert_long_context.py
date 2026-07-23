from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Iterable
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


MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_DIRECTORY = "ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MODEL_SAFETENSORS_SHA256 = (
    "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9"
)
PILOT_INDEX_SHA256 = (
    "4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba"
)
MODELING_BASE_SHA256 = (
    "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
)
SESSION_TEXTS_SHA256 = (
    "da6067f22f15ee0d593fa2489eb8acba4b3037c911a054be2aaf1c5e01932e22"
)
BASELINE_OOF_SHA256 = (
    "1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6"
)
BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
PILOT_ROWS = 4096
BENCHMARK_ROWS = 16
CONTEXT_LENGTH_CANDIDATES = (8192, 4096, 2048)
MAX_PROJECTED_HOURS = 8.0
MAX_RSS_BYTES = 8 * 1024**3
CPU_THREADS = 6
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
HYPOTHESIS_TEMPLATE = (
    "The student demonstrates mastery of this learning objective: {objective}."
)


def assert_e420_runtime() -> dict[str, str]:
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
            f"E420 runtime differs from the frozen .venv contract: {observed}"
        )
    return observed


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


def _normalized_objective(value: object) -> str:
    return " ".join(str(value).split())


def mastery_hypothesis(objective: object) -> str:
    return HYPOTHESIS_TEMPLATE.format(objective=_normalized_objective(objective))


def _source_paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    return {
        "model": paths.root
        / "assets"
        / "pretrained"
        / MODEL_DIRECTORY
        / "model.safetensors",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "sessions": paths.cache_dir / "session_texts.parquet",
        "baseline": paths.experiments_dir
        / "runs"
        / BASELINE_RUN_ID
        / "oof_predictions.parquet",
    }


def verify_e420_sources(project_root: str | Path) -> dict[str, str]:
    source_paths = _source_paths(project_root)
    expected = {
        "model": MODEL_SAFETENSORS_SHA256,
        "indices": PILOT_INDEX_SHA256,
        "modeling": MODELING_BASE_SHA256,
        "sessions": SESSION_TEXTS_SHA256,
        "baseline": BASELINE_OOF_SHA256,
    }
    for key, expected_sha in expected.items():
        _verify_file(source_paths[key], expected_sha, f"E420 {key}")
    return expected


def _load_pilot_frame(
    project_root: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    paths = discover_project_paths(project_root)
    verify_e420_sources(project_root)
    indices = np.load(paths.cache_dir / "qwen3_pilot_indices.npy")
    if indices.shape != (PILOT_ROWS,) or indices.dtype.kind not in "iu":
        raise ValueError("E420 pilot indices have the wrong shape or dtype.")
    if len(np.unique(indices)) != PILOT_ROWS or not np.all(np.diff(indices) > 0):
        raise ValueError("E420 pilot indices must be unique and strictly increasing.")

    modeling = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    sessions = pd.read_parquet(
        paths.cache_dir / "session_texts.parquet",
        columns=["session_id", "full_text"],
    )
    if sessions["session_id"].duplicated().any():
        raise ValueError("Session text cache contains duplicate session IDs.")
    selected = modeling.iloc[indices].copy()
    selected["_pilot_order"] = np.arange(PILOT_ROWS, dtype=np.int32)
    selected = selected.merge(
        sessions,
        on="session_id",
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_pilot_order", kind="mergesort")
    selected = selected.reset_index(drop=True)
    if selected["full_text"].isna().any() or selected["full_text"].eq("").any():
        raise ValueError("E420 pilot contains a missing full transcript.")

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
        raise ValueError("E420 baseline rows do not align with modeling data.")
    folds = baseline.iloc[indices]["fold"].to_numpy(dtype=np.int8)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("E420 pilot must cover all five frozen folds.")
    if set(selected["target"].astype(int)) != {0, 1}:
        raise ValueError("E420 pilot must contain both outcome labels.")
    return selected, indices.astype(np.int64), folds


def _configure_torch() -> None:
    import torch

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_encoder(project_root: str | Path):
    from transformers import AutoModel, AutoTokenizer

    paths = discover_project_paths(project_root)
    model_path = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    if int(model.config.hidden_size) != 768:
        raise ValueError("E420 expected ModernBERT hidden size 768.")
    if int(model.config.max_position_embeddings) != 8192:
        raise ValueError("E420 expected ModernBERT context length 8192.")
    return tokenizer, model


def _token_lengths(tokenizer, frame: pd.DataFrame) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(frame), 64):
        end = min(len(frame), start + 64)
        batch = frame.iloc[start:end]
        encoded = tokenizer(
            batch["full_text"].astype(str).tolist(),
            [mastery_hypothesis(value) for value in batch["learning_objective"]],
            padding=False,
            truncation="only_first",
            max_length=8192,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
    result = np.asarray(lengths, dtype=np.int32)
    if result.shape != (len(frame),) or (result <= 0).any() or (result > 8192).any():
        raise ValueError("E420 token-length audit failed.")
    return result


def select_benchmark_indices(token_lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(token_lengths)
    if values.ndim != 1 or len(values) < BENCHMARK_ROWS:
        raise ValueError("Not enough token lengths for the E420 benchmark.")
    order = np.argsort(values, kind="mergesort")
    positions = np.rint(np.linspace(0, len(order) - 1, BENCHMARK_ROWS)).astype(
        np.int64
    )
    selected = order[positions]
    if len(np.unique(selected)) != BENCHMARK_ROWS:
        raise RuntimeError("E420 benchmark quantiles did not select unique rows.")
    return selected


def _pool_hidden(hidden, attention_mask):
    import torch
    from torch.nn import functional as F

    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    first = F.normalize(hidden[:, 0, :], p=2, dim=1)
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    mean = F.normalize(mean, p=2, dim=1)
    pooled = torch.cat([first, mean], dim=1) / math.sqrt(2.0)
    if pooled.shape[1] != 1536 or not bool(torch.isfinite(pooled).all()):
        raise RuntimeError("E420 pooled representation failed audit.")
    return pooled


def _encode_one(tokenizer, model, transcript: str, objective: object, max_length: int):
    import torch

    encoded = tokenizer(
        [str(transcript)],
        [mastery_hypothesis(objective)],
        padding="max_length",
        truncation="only_first",
        max_length=max_length,
        return_tensors="pt",
    )
    if "token_type_ids" in encoded:
        raise ValueError("ModernBERT E420 unexpectedly emitted token_type_ids.")
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        pooled = _pool_hidden(hidden, encoded["attention_mask"])
    result = pooled[0].cpu().numpy().astype(np.float32)
    if result.shape != (1536,) or not np.isfinite(result).all():
        raise RuntimeError("E420 encoded row failed shape/finiteness audit.")
    return result, int(encoded["attention_mask"].sum())


def select_context_length(benchmark_rows: list[dict[str, object]]) -> int:
    by_length = {int(row["max_length"]): row for row in benchmark_rows}
    if set(by_length) != set(CONTEXT_LENGTH_CANDIDATES):
        raise ValueError("E420 benchmark does not cover every frozen context length.")
    for max_length in CONTEXT_LENGTH_CANDIDATES:
        row = by_length[max_length]
        if (
            float(row["projected_hours_for_4096"]) <= MAX_PROJECTED_HOURS
            and int(row["peak_rss_bytes"]) <= MAX_RSS_BYTES
            and bool(row["completed"])
        ):
            return max_length
    raise RuntimeError("No frozen E420 context length satisfies runtime/RSS limits.")


def benchmark_modernbert_context(
    project_root: str | Path,
) -> dict[str, object]:
    import torch
    import transformers

    runtime = assert_e420_runtime()
    paths = discover_project_paths(project_root)
    frame, _, _ = _load_pilot_frame(project_root)
    _configure_torch()
    tokenizer, model = _load_encoder(project_root)
    token_lengths = _token_lengths(tokenizer, frame)
    benchmark_indices = select_benchmark_indices(token_lengths)
    rows: list[dict[str, object]] = []
    for max_length in CONTEXT_LENGTH_CANDIDATES:
        started = time.perf_counter()
        peak_rss = _current_rss_bytes()
        completed = True
        failure = ""
        try:
            for row_index in benchmark_indices:
                row = frame.iloc[int(row_index)]
                _encode_one(
                    tokenizer,
                    model,
                    str(row["full_text"]),
                    row["learning_objective"],
                    max_length,
                )
                peak_rss = max(peak_rss, _current_rss_bytes())
                if peak_rss > MAX_RSS_BYTES:
                    raise MemoryError(
                        f"E420 RSS {peak_rss} exceeds {MAX_RSS_BYTES} bytes."
                    )
        except (MemoryError, RuntimeError) as exc:
            completed = False
            failure = str(exc)
        elapsed = time.perf_counter() - started
        projected = elapsed / BENCHMARK_ROWS * PILOT_ROWS / 3600.0
        rows.append(
            {
                "max_length": max_length,
                "benchmark_rows": BENCHMARK_ROWS,
                "elapsed_seconds": elapsed,
                "seconds_per_row": elapsed / BENCHMARK_ROWS,
                "projected_hours_for_4096": projected,
                "peak_rss_bytes": peak_rss,
                "completed": completed,
                "failure": failure,
            }
        )
        print(
            f"E420 benchmark max_length={max_length} rows={BENCHMARK_ROWS} "
            f"seconds={elapsed:.2f} projected_hours={projected:.2f} "
            f"rss_gib={peak_rss / 1024**3:.2f}",
            flush=True,
        )
    selected_length = select_context_length(rows)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E420_modernbert_long_context",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "pilot_rows": PILOT_ROWS,
        "benchmark_selection": "16 deterministic token-length quantiles",
        "benchmark_indices": benchmark_indices.tolist(),
        "benchmark_token_lengths": token_lengths[benchmark_indices].tolist(),
        "context_length_candidates": list(CONTEXT_LENGTH_CANDIDATES),
        "max_projected_hours": MAX_PROJECTED_HOURS,
        "max_rss_bytes": MAX_RSS_BYTES,
        "results": rows,
        "selected_max_length": selected_length,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "runtime_contract": runtime,
        "V_final_accessed": False,
    }
    benchmark_path = paths.cache_dir / "modernbert_e420_benchmark.json"
    benchmark_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(paths) -> dict[str, object]:
    benchmark_path = paths.cache_dir / "modernbert_e420_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError("Run the frozen E420 benchmark before cache construction.")
    result = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selected = int(result["selected_max_length"])
    if selected != select_context_length(list(result["results"])):
        raise ValueError("E420 benchmark selection no longer matches its frozen rule.")
    if result.get("model_revision") != MODEL_REVISION:
        raise ValueError("E420 benchmark belongs to a different model revision.")
    return result


def build_modernbert_pilot_cache(
    project_root: str | Path,
    flush_every: int = 16,
) -> dict[str, object]:
    runtime = assert_e420_runtime()
    paths = discover_project_paths(project_root)
    frame, indices, _ = _load_pilot_frame(project_root)
    benchmark = _load_benchmark(paths)
    max_length = int(benchmark["selected_max_length"])
    _configure_torch()
    tokenizer, model = _load_encoder(project_root)

    cache_path = paths.cache_dir / f"modernbert_e420_joint_{max_length}.npy"
    progress_path = (
        paths.cache_dir / f"modernbert_e420_joint_{max_length}.progress.json"
    )
    metadata_path = (
        paths.cache_dir / f"modernbert_e420_joint_{max_length}.metadata.json"
    )
    expected_shape = (PILOT_ROWS, 1536)
    if cache_path.exists():
        values = np.lib.format.open_memmap(cache_path, mode="r+")
        if values.shape != expected_shape or values.dtype != np.float32:
            raise ValueError("Existing E420 cache has the wrong shape or dtype.")
    else:
        values = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected_progress = {
            "model_revision": MODEL_REVISION,
            "max_length": max_length,
            "pilot_index_sha256": PILOT_INDEX_SHA256,
        }
        for key, expected_value in expected_progress.items():
            if progress.get(key) != expected_value:
                raise ValueError(f"Stale E420 progress field: {key}")
        completed = int(progress.get("completed_rows", 0))
        if not 0 <= completed <= PILOT_ROWS:
            raise ValueError("E420 progress row count is invalid.")

    token_counts = np.zeros(PILOT_ROWS, dtype=np.int32)
    started = time.perf_counter()
    peak_rss = _current_rss_bytes()
    for row_number in range(completed, PILOT_ROWS):
        row = frame.iloc[row_number]
        encoded, token_count = _encode_one(
            tokenizer,
            model,
            str(row["full_text"]),
            row["learning_objective"],
            max_length,
        )
        values[row_number] = encoded
        token_counts[row_number] = token_count
        peak_rss = max(peak_rss, _current_rss_bytes())
        if peak_rss > MAX_RSS_BYTES:
            raise MemoryError(f"E420 RSS {peak_rss} exceeds {MAX_RSS_BYTES} bytes.")
        finished = row_number + 1
        if finished % flush_every == 0 or finished == PILOT_ROWS:
            values.flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "candidate": "E420_modernbert_long_context",
                        "model_revision": MODEL_REVISION,
                        "max_length": max_length,
                        "pilot_index_sha256": PILOT_INDEX_SHA256,
                        "completed_rows": finished,
                        "total_rows": PILOT_ROWS,
                        "peak_rss_bytes": peak_rss,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if finished % 64 == 0 or finished == PILOT_ROWS:
            elapsed = time.perf_counter() - started
            rate = (finished - completed) / max(elapsed, 1e-9)
            print(
                f"E420 encoded={finished}/{PILOT_ROWS} "
                f"rows_per_second={rate:.4f} rss_gib={peak_rss / 1024**3:.2f}",
                flush=True,
            )
    del values

    completed_values = np.load(cache_path, mmap_mode="r")
    if (
        completed_values.shape != expected_shape
        or not np.isfinite(completed_values).all()
    ):
        raise ValueError("Completed E420 cache failed shape/finiteness audit.")
    norms = np.linalg.norm(completed_values, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5, rtol=0.0):
        raise ValueError("Completed E420 pooled vectors are not unit-normalized.")
    if completed == 0:
        token_count_summary = {
            "minimum": int(token_counts.min()),
            "mean": float(token_counts.mean()),
            "maximum": int(token_counts.max()),
            "at_limit": int((token_counts == max_length).sum()),
        }
    else:
        token_count_summary = {
            "minimum": None,
            "mean": None,
            "maximum": None,
            "at_limit": None,
            "note": "resumed run; token counts are not reconstructed for prior rows",
        }
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E420_modernbert_long_context",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": PILOT_INDEX_SHA256,
        "pilot_response_sha256": hashlib.sha256(
            "\n".join(frame["response_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "pilot_indices_min": int(indices.min()),
        "pilot_indices_max": int(indices.max()),
        "max_length": max_length,
        "pooling": "concat(l2(first_token),l2(masked_mean))/sqrt(2)",
        "truncation": "only_first",
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "cache_file": cache_path.name,
        "cache_sha256": _sha256(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "token_counts": token_count_summary,
        "peak_rss_bytes": peak_rss,
        "runtime_contract": runtime,
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress_path.unlink(missing_ok=True)
    return {
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


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
        indices = rng.integers(0, len(grouped), size=len(grouped))
        values[replicate] = sums[indices].sum() / counts[indices].sum()
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


def validate_modernbert_pilot(project_root: str | Path) -> dict[str, object]:
    runtime = assert_e420_runtime()
    paths = discover_project_paths(project_root)
    frame, indices, folds = _load_pilot_frame(project_root)
    benchmark = _load_benchmark(paths)
    max_length = int(benchmark["selected_max_length"])
    cache_path = paths.cache_dir / f"modernbert_e420_joint_{max_length}.npy"
    metadata_path = (
        paths.cache_dir / f"modernbert_e420_joint_{max_length}.metadata.json"
    )
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Build and audit the E420 pilot cache before validation.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_sha256") != _sha256(cache_path):
        raise ValueError("E420 pilot cache hash differs from frozen metadata.")
    modern_features = np.load(cache_path)
    if modern_features.shape != (PILOT_ROWS, 1536):
        raise ValueError("E420 ModernBERT feature shape is invalid.")

    full_frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features_full, base_similarity_full = prepare_bge_base_features(
        paths.cache_dir, full_frame
    )
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full_frame)
    base_features = base_features_full[indices]
    dense = dense_full[indices].copy()
    dense[:, -1:] = base_similarity_full[indices]

    baseline_prediction = semantic_logistic_oof(
        frame,
        folds,
        base_features,
        dense,
        regularization_c=FIXED_REGULARIZATION_C,
        n_splits=5,
    )
    candidate_prediction = semantic_logistic_oof(
        frame,
        folds,
        modern_features,
        dense,
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_modernbert_long_context")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(
        [
            {"model": "bge_base_fair_comparator", **baseline_metrics},
            {"model": "E420_modernbert_long_context", **candidate_metrics},
        ]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    prediction_frame = pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "session_id": frame["session_id"],
            "learning_objective_id": frame["learning_objective_id"],
            "fold": folds,
            "target": target,
            "pred_bge_base": baseline_prediction,
            "pred_modernbert": candidate_prediction,
        }
    )
    prediction_frame.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    bootstrap_frame = pd.DataFrame([bootstrap])
    bootstrap_frame.to_csv(
        run_dir / "bootstrap.csv", index=False, lineterminator="\n"
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E420_modernbert_long_context",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "pilot_index_sha256": PILOT_INDEX_SHA256,
        "max_length": max_length,
        "pooling": metadata["pooling"],
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
        "source_hashes": verify_e420_sources(project_root),
        "runtime_contract": runtime,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered E420 ModernBERT long-context screen."
    )
    parser.add_argument(
        "command", choices=("benchmark", "build", "validate", "pipeline")
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--flush-every", type=int, default=16)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "benchmark":
        result = benchmark_modernbert_context(args.project_root)
    elif args.command == "build":
        result = build_modernbert_pilot_cache(
            args.project_root, flush_every=args.flush_every
        )
    elif args.command == "validate":
        result = validate_modernbert_pilot(args.project_root)
    else:
        benchmark_modernbert_context(args.project_root)
        build_modernbert_pilot_cache(
            args.project_root, flush_every=args.flush_every
        )
        result = validate_modernbert_pilot(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
