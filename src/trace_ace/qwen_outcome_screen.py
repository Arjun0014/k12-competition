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
    FIXED_REGULARIZATION_C,
    MAX_PROJECTED_HOURS,
    MAX_RSS_BYTES,
    PILOT_INDEX_SHA256,
    PILOT_PROTOCOL,
    PILOT_ROWS,
    _paired_session_bootstrap,
    _sha256,
    assert_e450_runtime,
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


MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_DIRECTORY = "Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
MODELING_SHA256 = "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
CONTEXT_SHA256 = "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6"
BASELINE_SHA256 = "1693717193bae4bed75765d48b6c85de8f240eba8f5ac92b0724f499e7fe7ba6"
BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
SYSTEM = "You assess whether a student has mastered a K-12 learning objective after tutoring."
USER_TEMPLATE = (
    "Learning objective: {objective}\nTutoring evidence:\n{context}\n\n"
    "Will the student answer the next assessment question correctly? Answer Yes or No."
)
MARKER = "ZXQW_EVIDENCE_MARKER_7391"
MAX_LENGTH = 384
HIDDEN = 1024
YES_ID = 9454
NO_ID = 2753
BENCHMARK_ROWS = 16
CPU_THREADS = 6


def _source_paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    model = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    return {
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "tokenizer": model / "tokenizer.json",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "contexts": paths.cache_dir / "response_objective_context.parquet",
        "baseline": paths.experiments_dir / "runs" / BASELINE_RUN_ID / "oof_predictions.parquet",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    expected = {
        "model": MODEL_SHA256,
        "config": CONFIG_SHA256,
        "tokenizer": TOKENIZER_SHA256,
        "indices": PILOT_INDEX_SHA256,
        "modeling": MODELING_SHA256,
        "contexts": CONTEXT_SHA256,
        "baseline": BASELINE_SHA256,
    }
    for key, digest in expected.items():
        if _sha256(_source_paths(project_root)[key]) != digest:
            raise ValueError(f"E490 source drift: {key}")
    return expected


def _load(project_root: str | Path):
    paths = discover_project_paths(project_root)
    verify_sources(project_root)
    indices = np.load(paths.cache_dir / "qwen3_pilot_indices.npy")
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    baseline = pd.read_parquet(_source_paths(project_root)["baseline"])
    baseline = baseline.loc[baseline.protocol.eq(PILOT_PROTOCOL)].reset_index(drop=True)
    if indices.shape != (PILOT_ROWS,) or list(frame.response_id) != list(contexts.response_id):
        raise ValueError("E490 pilot alignment changed.")
    if list(frame.response_id) != list(baseline.response_id):
        raise ValueError("E490 baseline alignment changed.")
    selected = frame.iloc[indices].reset_index(drop=True)
    evidence = [compact_objective_context(x) for x in contexts.iloc[indices].objective_context]
    folds = baseline.iloc[indices].fold.to_numpy(dtype=np.int8)
    return selected, indices.astype(np.int64), folds, evidence


def _load_model(project_root: str | Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_dir = discover_project_paths(project_root).root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    if model.config.hidden_size != HIDDEN:
        raise ValueError("E490 hidden size changed.")
    if tokenizer.encode("Yes", add_special_tokens=False) != [YES_ID]:
        raise ValueError("E490 Yes token changed.")
    if tokenizer.encode("No", add_special_tokens=False) != [NO_ID]:
        raise ValueError("E490 No token changed.")
    return tokenizer, model


def prompt_token_ids(tokenizer, objective: str, context: str) -> list[int]:
    user = USER_TEMPLATE.format(objective=" ".join(str(objective).split()), context=MARKER)
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count(MARKER) != 1:
        raise ValueError("E490 chat template did not preserve evidence marker.")
    left, right = rendered.split(MARKER)
    left_ids = tokenizer.encode(left, add_special_tokens=False)
    right_ids = tokenizer.encode(right, add_special_tokens=False)
    budget = MAX_LENGTH - len(left_ids) - len(right_ids)
    if budget <= 0:
        raise ValueError("E490 fixed prompt exceeds maximum length.")
    context_ids = tokenizer.encode(str(context), add_special_tokens=False)[:budget]
    result = [*left_ids, *context_ids, *right_ids]
    if len(result) > MAX_LENGTH:
        raise RuntimeError("E490 evidence-only truncation failed.")
    return result


def _encode_rows(tokenizer, model, objectives: list[str], evidence: list[str]) -> np.ndarray:
    import torch

    output = np.empty((len(objectives), HIDDEN + 1), dtype=np.float32)
    with torch.inference_mode():
        for index, (objective, context) in enumerate(zip(objectives, evidence, strict=True)):
            ids = prompt_token_ids(tokenizer, objective, context)
            input_ids = torch.tensor([ids], dtype=torch.long)
            result = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[-1][0, -1].float()
            hidden /= torch.linalg.vector_norm(hidden)
            margin = result.logits[0, -1, YES_ID] - result.logits[0, -1, NO_ID]
            output[index, :HIDDEN] = hidden.cpu().numpy()
            output[index, HIDDEN] = float(margin)
    if not np.isfinite(output).all():
        raise ValueError("E490 representation contains non-finite values.")
    return output


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_e450_runtime()
    paths = discover_project_paths(project_root)
    output_path = paths.cache_dir / "qwen_outcome_e490_benchmark.json"
    if output_path.exists():
        return json.loads(output_path.read_text())
    frame, _, _, evidence = _load(project_root)
    tokenizer, model = _load_model(project_root)
    objectives = frame.learning_objective.fillna("").astype(str).tolist()
    lengths = np.asarray(
        [len(prompt_token_ids(tokenizer, objective, text)) for objective, text in zip(objectives, evidence, strict=True)]
    )
    order = np.argsort(lengths, kind="mergesort")
    positions = np.rint(np.linspace(0, PILOT_ROWS - 1, BENCHMARK_ROWS)).astype(int)
    selected = order[positions]
    started = time.perf_counter()
    _encode_rows(
        tokenizer, model, [objectives[i] for i in selected], [evidence[i] for i in selected]
    )
    elapsed = time.perf_counter() - started
    hours = elapsed / BENCHMARK_ROWS * PILOT_ROWS / 3600
    rss = _current_rss_bytes()
    result = {
        "candidate": "E490_qwen_outcome",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "elapsed_seconds": elapsed,
        "seconds_per_row": elapsed / BENCHMARK_ROWS,
        "projected_hours": hours,
        "peak_rss_bytes": rss,
        "selected_positions": selected.tolist(),
        "selected_prompt_lengths": lengths[selected].tolist(),
        "proceed": bool(hours < MAX_PROJECTED_HOURS and rss < MAX_RSS_BYTES),
        "runtime_contract": runtime,
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build(project_root: str | Path, flush_every: int = 8) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    if not benchmark(project_root)["proceed"]:
        raise RuntimeError("E490 operational benchmark failed.")
    frame, _, _, evidence = _load(project_root)
    objectives = frame.learning_objective.fillna("").astype(str).tolist()
    tokenizer, model = _load_model(project_root)
    cache_path = paths.cache_dir / "qwen_outcome_e490.npy"
    progress_path = paths.cache_dir / "qwen_outcome_e490.progress.json"
    values = np.lib.format.open_memmap(
        cache_path,
        mode="r+" if cache_path.exists() else "w+",
        dtype=np.float32,
        shape=(PILOT_ROWS, HIDDEN + 1),
    )
    completed = 0
    if progress_path.exists():
        completed = int(json.loads(progress_path.read_text())["completed_rows"])
    started = time.perf_counter()
    peak = _current_rss_bytes()
    for start in range(completed, PILOT_ROWS, flush_every):
        end = min(PILOT_ROWS, start + flush_every)
        values[start:end] = _encode_rows(
            tokenizer, model, objectives[start:end], evidence[start:end]
        )
        values.flush()
        peak = max(peak, _current_rss_bytes())
        progress_path.write_text(json.dumps({"completed_rows": end, "revision": MODEL_REVISION}))
        print(f"Encoded E490 prompts for {end}/{PILOT_ROWS}.", flush=True)
    del values
    cache = np.load(cache_path, mmap_mode="r")
    if not np.isfinite(cache).all():
        raise ValueError("E490 completed cache contains non-finite values.")
    if not np.allclose(np.linalg.norm(cache[:, :HIDDEN], axis=1), 1, atol=2e-5, rtol=0):
        raise ValueError("E490 hidden states are not normalized.")
    metadata = {
        "candidate": "E490_qwen_outcome",
        "cache_sha256": _sha256(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak,
        "margin_min": float(cache[:, HIDDEN].min()),
        "margin_mean": float(cache[:, HIDDEN].mean()),
        "margin_max": float(cache[:, HIDDEN].max()),
        "source_hashes": verify_sources(project_root),
        "V_final_accessed": False,
    }
    metadata_path = paths.cache_dir / "qwen_outcome_e490.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    progress_path.unlink(missing_ok=True)
    return metadata


def validate(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame, indices, folds, _ = _load(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    base_x, base_similarity = prepare_bge_base_features(paths.cache_dir, full)
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    cache = np.load(paths.cache_dir / "qwen_outcome_e490.npy")
    dense_base = dense_full[indices].copy()
    dense_base[:, -1:] = base_similarity[indices]
    dense_candidate = np.column_stack([dense_full[indices], cache[:, HIDDEN]])
    pred_base = semantic_logistic_oof(frame, folds, base_x[indices], dense_base, 0.1, 5)
    pred_candidate = semantic_logistic_oof(
        frame, folds, cache[:, :HIDDEN], dense_candidate, FIXED_REGULARIZATION_C, 5
    )
    target = frame.target.to_numpy(dtype=np.int8)
    base_metrics = binary_metrics(target, pred_base)
    candidate_metrics = binary_metrics(target, pred_candidate)
    delta = {key: candidate_metrics[key] - base_metrics[key] for key in candidate_metrics}
    boot = _paired_session_bootstrap(frame, pred_base, pred_candidate)
    loss_path = delta["log_loss"] <= -0.0015 and all(
        [delta["roc_auc"] >= 0, delta["brier_score"] <= 0, delta["ece_10"] <= 0]
    )
    auc_path = delta["roc_auc"] >= 0.005 and all(
        [delta["log_loss"] <= 0, delta["brier_score"] <= 0, delta["ece_10"] <= 0]
    )
    passed = bool((loss_path or auc_path) and boot["support_positive_log_loss_gain"] >= 0.9)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_qwen_outcome")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    pd.DataFrame([
        {"model": "bge_base_fair_comparator", **base_metrics},
        {"model": "E490_qwen_outcome", **candidate_metrics},
    ]).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "response_id": frame.response_id,
            "session_id": frame.session_id,
            "learning_objective_id": frame.learning_objective_id,
            "fold": folds,
            "target": target,
            "pred_bge_base": pred_base,
            "pred_qwen_outcome": pred_candidate,
            "qwen_yes_minus_no_logit": cache[:, HIDDEN],
        }
    ).to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame([boot]).to_csv(run_dir / "bootstrap.csv", index=False)
    report = {
        "run_id": run_id,
        "candidate": "E490_qwen_outcome",
        "baseline_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_baseline": delta,
        "paired_session_bootstrap": boot,
        "screen_clauses": {"loss_path": loss_path, "auc_path": auc_path},
        "passes_screen": passed,
        "dense_features": [*dense_names, "qwen_yes_minus_no_logit"],
        "cache_sha256": _sha256(paths.cache_dir / "qwen_outcome_e490.npy"),
        "source_hashes": verify_sources(project_root),
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
