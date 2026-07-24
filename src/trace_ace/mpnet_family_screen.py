from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.bge_large_capacity import (
    BATCH_SIZE,
    BENCHMARK_ROWS,
    BOOTSTRAP_SEED,
    CPU_THREADS,
    FIXED_REGULARIZATION_C,
    MAX_PROJECTED_HOURS,
    MAX_RSS_BYTES,
    PILOT_INDEX_SHA256,
    PILOT_PROTOCOL,
    PILOT_ROWS,
    _paired_session_bootstrap,
    _sha256,
    assert_e450_runtime,
    select_benchmark_indices,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_cache import compact_objective_context
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)
from trace_ace.source_robust_validation import _current_rss_bytes


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_DIRECTORY = "all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MODEL_SHA256 = "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658"
CONFIG_SHA256 = "d46a3e04ded82bba22528424480697d394eeda6a27484e08c5bb2bdf5906cfa0"
POOLING_SHA256 = "a37f83ada23e7887be6b88f4998927dbeac0038af301553c7cd5461413bf1a56"
MODELING_SHA256 = "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
CONTEXT_SHA256 = "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6"
BASELINE_SHA256 = "1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6"
BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
DIMENSION = 768
MAX_LENGTH = 256


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "pooling": model / "1_Pooling" / "config.json",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "contexts": paths.cache_dir / "response_objective_context.parquet",
        "baseline": paths.experiments_dir / "runs" / BASELINE_RUN_ID / "oof_predictions.parquet",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    expected = {
        "model": MODEL_SHA256,
        "config": CONFIG_SHA256,
        "pooling": POOLING_SHA256,
        "indices": PILOT_INDEX_SHA256,
        "modeling": MODELING_SHA256,
        "contexts": CONTEXT_SHA256,
        "baseline": BASELINE_SHA256,
    }
    for key, value in expected.items():
        path = _paths(project_root)[key]
        if not path.exists() or _sha256(path) != value:
            raise ValueError(f"E470 source drift: {key}")
    return expected


def _load(project_root: str | Path):
    paths = discover_project_paths(project_root)
    verify_sources(project_root)
    indices = np.load(paths.cache_dir / "qwen3_pilot_indices.npy")
    modeling = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    baseline = pd.read_parquet(_paths(project_root)["baseline"])
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(drop=True)
    if indices.shape != (PILOT_ROWS,) or list(modeling.response_id) != list(contexts.response_id):
        raise ValueError("E470 pilot alignment changed.")
    if list(modeling.response_id) != list(baseline.response_id):
        raise ValueError("E470 baseline alignment changed.")
    frame = modeling.iloc[indices].reset_index(drop=True)
    folds = baseline.iloc[indices].fold.to_numpy(dtype=np.int8)
    texts = [compact_objective_context(x) for x in contexts.iloc[indices].objective_context]
    return frame, indices.astype(np.int64), folds, texts


def _model(project_root: str | Path):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = discover_project_paths(project_root).root
    model = SentenceTransformer(str(root / "assets" / "pretrained" / MODEL_DIRECTORY), device="cpu")
    model.max_seq_length = MAX_LENGTH
    if model.get_embedding_dimension() != DIMENSION:
        raise ValueError("E470 embedding dimension changed.")
    pooling = json.loads(_paths(project_root)["pooling"].read_text(encoding="utf-8"))
    if not pooling["pooling_mode_mean_tokens"] or pooling["pooling_mode_cls_token"]:
        raise ValueError("E470 mean-pooling contract changed.")
    return model


def _encode(model, texts: list[str]) -> np.ndarray:
    values = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if values.shape != (len(texts), DIMENSION) or not np.isfinite(values).all():
        raise ValueError("E470 encoder output is invalid.")
    return values


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_e450_runtime()
    paths = discover_project_paths(project_root)
    output = paths.cache_dir / "mpnet_e470_benchmark.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    frame, _, _, texts = _load(project_root)
    model = _model(project_root)
    lengths = np.asarray(
        [len(row) for row in model.tokenizer(texts, padding=False, truncation=False)["input_ids"]],
        dtype=np.int32,
    )
    selected = select_benchmark_indices(lengths)
    started = time.perf_counter()
    _encode(model, [texts[i] for i in selected])
    elapsed = time.perf_counter() - started
    rss = _current_rss_bytes()
    rows = PILOT_ROWS + int(frame.learning_objective.nunique(dropna=False))
    hours = elapsed / BENCHMARK_ROWS * rows / 3600
    result = {
        "candidate": "E470_mpnet_family",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "benchmark_rows": BENCHMARK_ROWS,
        "batch_size": BATCH_SIZE,
        "cpu_threads": CPU_THREADS,
        "elapsed_seconds": elapsed,
        "seconds_per_row": elapsed / BENCHMARK_ROWS,
        "projected_rows": rows,
        "projected_hours": hours,
        "peak_rss_bytes": rss,
        "proceed": bool(hours < MAX_PROJECTED_HOURS and rss < MAX_RSS_BYTES),
        "selected_indices": selected.tolist(),
        "selected_token_lengths": lengths[selected].tolist(),
        "runtime_contract": runtime,
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build(project_root: str | Path, flush_every: int = 32) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    bench = benchmark(project_root)
    if not bench["proceed"]:
        raise RuntimeError("E470 operational benchmark failed.")
    frame, indices, _, texts = _load(project_root)
    model = _model(project_root)
    context_path = paths.cache_dir / "mpnet_e470_context_256.npy"
    objective_path = paths.cache_dir / "mpnet_e470_objective_256.npy"
    progress_path = paths.cache_dir / "mpnet_e470.progress.json"
    shape = (PILOT_ROWS, DIMENSION)
    values = np.lib.format.open_memmap(
        context_path, mode="r+" if context_path.exists() else "w+", dtype=np.float32, shape=shape
    )
    completed = 0
    if progress_path.exists():
        completed = int(json.loads(progress_path.read_text())["completed_rows"])
    started = time.perf_counter()
    peak = _current_rss_bytes()
    for start in range(completed, PILOT_ROWS, flush_every):
        end = min(PILOT_ROWS, start + flush_every)
        values[start:end] = _encode(model, texts[start:end])
        values.flush()
        peak = max(peak, _current_rss_bytes())
        progress_path.write_text(json.dumps({"completed_rows": end, "revision": MODEL_REVISION}))
        print(f"Encoded E470 contexts for {end}/{PILOT_ROWS}.", flush=True)
    objectives = frame.learning_objective.fillna("").astype(str)
    if not objective_path.exists():
        unique = objectives.drop_duplicates().tolist()
        encoded = _encode(model, unique)
        lookup = dict(zip(unique, encoded, strict=True))
        np.save(objective_path, np.vstack([lookup[x] for x in objectives]).astype(np.float32))
    del values
    for path in (context_path, objective_path):
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("E470 completed cache audit failed.")
        if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=2e-5, rtol=0):
            raise ValueError("E470 completed cache is not normalized.")
    metadata = {
        "candidate": "E470_mpnet_family",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "pilot_index_sha256": PILOT_INDEX_SHA256,
        "context_sha256": _sha256(context_path),
        "objective_sha256": _sha256(objective_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak,
        "pilot_indices_min": int(indices.min()),
        "pilot_indices_max": int(indices.max()),
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    metadata_path = paths.cache_dir / "mpnet_e470.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    progress_path.unlink(missing_ok=True)
    return metadata


def _interaction(context: np.ndarray, objective: np.ndarray):
    features = np.column_stack(
        [context * 0.7, objective * 0.7, context * objective * 6.0, np.abs(context - objective) * 0.7]
    ).astype(np.float32)
    return features, np.sum(context * objective, axis=1, keepdims=True).astype(np.float32)


def validate(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame, indices, folds, _ = _load(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    base_x, base_similarity = prepare_bge_base_features(paths.cache_dir, full)
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    context = np.load(paths.cache_dir / "mpnet_e470_context_256.npy")
    objective = np.load(paths.cache_dir / "mpnet_e470_objective_256.npy")
    candidate_x, candidate_similarity = _interaction(context, objective)
    dense_base = dense_full[indices].copy()
    dense_base[:, -1:] = base_similarity[indices]
    dense_candidate = dense_full[indices].copy()
    dense_candidate[:, -1:] = candidate_similarity
    pred_base = semantic_logistic_oof(
        frame, folds, base_x[indices], dense_base, FIXED_REGULARIZATION_C, 5
    )
    pred_candidate = semantic_logistic_oof(
        frame, folds, candidate_x, dense_candidate, FIXED_REGULARIZATION_C, 5
    )
    target = frame.target.to_numpy(dtype=np.int8)
    base_metrics = binary_metrics(target, pred_base)
    candidate_metrics = binary_metrics(target, pred_candidate)
    delta = {k: candidate_metrics[k] - base_metrics[k] for k in candidate_metrics}
    boot = _paired_session_bootstrap(frame, pred_base, pred_candidate)
    loss_path = (
        delta["log_loss"] <= -0.0015 and delta["roc_auc"] >= 0
        and delta["brier_score"] <= 0 and delta["ece_10"] <= 0
    )
    auc_path = (
        delta["roc_auc"] >= 0.005 and delta["log_loss"] <= 0
        and delta["brier_score"] <= 0 and delta["ece_10"] <= 0
    )
    passed = bool((loss_path or auc_path) and boot["support_positive_log_loss_gain"] >= 0.9)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_mpnet_family")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    pd.DataFrame([
        {"model": "bge_base_fair_comparator", **base_metrics},
        {"model": "E470_mpnet_family", **candidate_metrics},
    ]).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame({
        "response_id": frame.response_id, "session_id": frame.session_id, "fold": folds,
        "target": target, "pred_bge_base": pred_base, "pred_mpnet": pred_candidate,
    }).to_parquet(run_dir / "oof_predictions.parquet", index=False)
    report = {
        "run_id": run_id,
        "candidate": "E470_mpnet_family",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "baseline_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_baseline": delta,
        "paired_session_bootstrap": boot,
        "screen_clauses": {"loss_path": loss_path, "auc_path": auc_path},
        "passes_screen": passed,
        "dense_features": dense_names,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("benchmark", "build", "validate", "pipeline"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "benchmark":
        result = benchmark(args.project_root)
    elif args.command == "build":
        result = build(args.project_root)
    elif args.command == "validate":
        result = validate(args.project_root)
    else:
        benchmark(args.project_root)
        build(args.project_root)
        result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
