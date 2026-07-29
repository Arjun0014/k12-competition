from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.bge_base_multiview_screen import (
    PILOT_ROWS,
    SOURCE_HASHES,
    _current_rss_bytes,
    assert_runtime,
    verify_sources,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.io import discover_project_paths
from trace_ace.mathdial_contrastive_alignment import (
    EVAL_BATCH_SIZE,
    MODEL_SHA256,
    PROTOCOL_ID as EXTERNAL_PROTOCOL_ID,
    QUERY_MAX_LENGTH,
    QUERY_PREFIX,
)
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_cache import compact_objective_context
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E820_mathdial_contrastive_competition_screen_v1"
EXTERNAL_RUN_ID = "20260729T190028Z_mathdial_contrastive_alignment"
EXTERNAL_DELTA_SHA256 = (
    "c7caf144e91db5aedd427ab27c860cd5ae091cbb1e466621b0b563f88e6b9248"
)
EXTERNAL_REPORT_SHA256 = (
    "d3f6f8c5bb5ec832c959db91d2777f93b58330b8fd129afe5c49db78b3d02fad"
)
DELTA_PREFIX = EXTERNAL_DELTA_SHA256[:12]
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
REGULARIZATION_C = 0.1
BOOTSTRAP_REPLICATES = 5_000
SEED = 20260730
DOCUMENT_MAX_LENGTH = 256
BATCH_SIZE = EVAL_BATCH_SIZE
PROBABILITY_CLIP = 1e-6
MAX_PROJECTED_SECONDS = 3600
MAX_RSS_BYTES = 8 * 1024**3
V05_WEIGHTS = {
    "pred_full": 0.25,
    "pred_role": 0.25,
    "pred_bge_base": 0.50,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    run_dir = paths.experiments_dir / "runs" / EXTERNAL_RUN_ID
    stem = f"e820_mathdial_contrastive_{DELTA_PREFIX}"
    return {
        "model": (
            paths.root
            / "assets"
            / "pretrained"
            / "bge-base-en-v1.5"
        ),
        "delta": run_dir / "e820_bge_base_delta.safetensors",
        "external_report": run_dir / "report.json",
        "indices": paths.cache_dir / "qwen3_pilot_indices.npy",
        "modeling": paths.cache_dir / "modeling_base.parquet",
        "contexts": paths.cache_dir / "response_objective_context.parquet",
        "base_context": paths.cache_dir / "bge_base_context_256.npy",
        "context_cache": paths.cache_dir / f"{stem}_context_256.npy",
        "objective_cache": paths.cache_dir / f"{stem}_objective_96.npy",
        "progress": paths.cache_dir / f"{stem}.progress.json",
        "metadata": paths.cache_dir / f"{stem}.metadata.json",
        "target_free_report": (
            paths.cache_dir / f"{stem}.target_free_report.json"
        ),
        "benchmark": paths.cache_dir / f"{stem}.benchmark.json",
        "binding": (
            paths.root / "E820_COMPETITION_CACHE_BINDING_2026-07-30.json"
        ),
        "experiments": paths.experiments_dir,
    }


def verify_external_gate(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    if _sha256(paths["delta"]) != EXTERNAL_DELTA_SHA256:
        raise ValueError("E820 external delta SHA-256 changed.")
    if _sha256(paths["external_report"]) != EXTERNAL_REPORT_SHA256:
        raise ValueError("E820 external report SHA-256 changed.")
    report = json.loads(paths["external_report"].read_text(encoding="utf-8"))
    required = {
        "protocol_id": EXTERNAL_PROTOCOL_ID,
        "passes_target_free_gate": True,
        "competition_outcomes_accessed": False,
        "v_seen_accessed": False,
        "v_objective_accessed": False,
        "v_style_accessed": False,
        "v_joint_accessed": False,
        "v_final_accessed": False,
    }
    observed = {name: report.get(name) for name in required}
    if observed != required:
        raise ValueError(f"E820 external gate contract changed: {observed}")
    return report


def load_target_free_inputs(
    project_root: str | Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    paths = _paths(project_root)
    verify_sources(project_root)
    indices = np.load(paths["indices"])
    if (
        indices.shape != (PILOT_ROWS,)
        or indices.dtype.kind not in "iu"
        or len(np.unique(indices)) != PILOT_ROWS
        or not np.all(np.diff(indices) > 0)
    ):
        raise ValueError("E820 pilot indices changed.")
    modeling = pd.read_parquet(
        paths["modeling"],
        columns=["response_id", "learning_objective"],
    ).reset_index(drop=True)
    contexts = pd.read_parquet(
        paths["contexts"],
        columns=["response_id", "objective_context"],
    ).reset_index(drop=True)
    if list(modeling["response_id"]) != list(contexts["response_id"]):
        raise ValueError("E820 modeling/context rows do not align.")
    frame = modeling.iloc[indices].reset_index(drop=True)
    frame["document"] = [
        compact_objective_context(value)
        for value in contexts.iloc[indices]["objective_context"].astype(str)
    ]
    frame["query"] = (
        QUERY_PREFIX + frame["learning_objective"].fillna("").astype(str)
    )
    if frame[["document", "query"]].eq("").any().any():
        raise ValueError("E820 competition text contains an empty view.")
    return frame, indices.astype(np.int64)


def _configure_torch() -> None:
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_encoder(project_root: str | Path, apply_delta: bool):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    paths = _paths(project_root)
    if (
        _sha256(paths["model"] / "model.safetensors")
        != MODEL_SHA256
    ):
        raise ValueError("E820 packaged BGE-base model changed.")
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model"], local_files_only=True
    )
    model = AutoModel.from_pretrained(
        paths["model"], local_files_only=True
    )
    if apply_delta:
        delta = load_file(str(paths["delta"]))
        state = model.state_dict()
        if not delta or not set(delta).issubset(state):
            raise ValueError("E820 delta keys do not match BGE-base.")
        with torch.no_grad():
            for name, value in delta.items():
                if state[name].shape != value.shape:
                    raise ValueError(f"E820 delta shape changed for {name}.")
                state[name].add_(value)
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer


def _encode(
    model,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer(
                list(texts[start : start + BATCH_SIZE]),
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            output = model(**encoded).last_hidden_state[:, 0]
            output = functional.normalize(output, p=2, dim=1)
            batches.append(output.cpu().numpy().astype(np.float32))
    result = np.concatenate(batches, axis=0)
    if result.shape != (len(texts), 768) or not np.isfinite(result).all():
        raise RuntimeError("E820 competition encoder output is invalid.")
    return result


def base_parity(
    project_root: str | Path,
    rows: int = 32,
) -> dict[str, float | int]:
    paths = _paths(project_root)
    frame, indices = load_target_free_inputs(project_root)
    _configure_torch()
    model, tokenizer = _load_encoder(project_root, apply_delta=False)
    encoded = _encode(
        model,
        tokenizer,
        frame["document"].iloc[:rows].astype(str).tolist(),
        DOCUMENT_MAX_LENGTH,
    )
    reference = np.load(paths["base_context"], mmap_mode="r")[indices[:rows]]
    cosine = np.sum(encoded * reference, axis=1)
    return {
        "rows": int(rows),
        "max_absolute_difference": float(
            np.max(np.abs(encoded - reference))
        ),
        "minimum_cosine": float(cosine.min()),
    }


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    external = verify_external_gate(project_root)
    frame, _ = load_target_free_inputs(project_root)
    parity = base_parity(project_root)
    if (
        parity["max_absolute_difference"] > 2e-5
        or parity["minimum_cosine"] < 0.99999
    ):
        raise RuntimeError(f"E820 base encoder parity failed: {parity}")
    _configure_torch()
    model, tokenizer = _load_encoder(project_root, apply_delta=True)
    rows = min(64, len(frame))
    started = time.perf_counter()
    _encode(
        model,
        tokenizer,
        frame["document"].iloc[:rows].astype(str).tolist(),
        DOCUMENT_MAX_LENGTH,
    )
    document_seconds = time.perf_counter() - started
    unique_queries = frame["query"].drop_duplicates().astype(str).tolist()
    query_rows = min(64, len(unique_queries))
    started = time.perf_counter()
    _encode(
        model,
        tokenizer,
        unique_queries[:query_rows],
        QUERY_MAX_LENGTH,
    )
    query_seconds = time.perf_counter() - started
    projected = (
        document_seconds / rows * len(frame)
        + query_seconds / query_rows * len(unique_queries)
    )
    peak = _current_rss_bytes()
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "external_report_sha256": EXTERNAL_REPORT_SHA256,
        "external_delta_sha256": EXTERNAL_DELTA_SHA256,
        "external_gate_passed": external["passes_target_free_gate"],
        "pilot_rows": len(frame),
        "unique_queries": len(unique_queries),
        "sample_document_rows": rows,
        "sample_query_rows": query_rows,
        "document_seconds": document_seconds,
        "query_seconds": query_seconds,
        "projected_total_seconds": projected,
        "base_encoder_parity": parity,
        "peak_rss_bytes": peak,
        "resource_gate": {
            "max_projected_seconds": MAX_PROJECTED_SECONDS,
            "max_rss_bytes": MAX_RSS_BYTES,
            "passes_duration": projected < MAX_PROJECTED_SECONDS,
            "passes_memory": peak < MAX_RSS_BYTES,
        },
        "proceed": bool(
            projected < MAX_PROJECTED_SECONDS and peak < MAX_RSS_BYTES
        ),
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    path = _paths(project_root)["benchmark"]
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["benchmark_path"] = str(path)
    result["benchmark_sha256"] = _sha256(path)
    return result


def build_cache(
    project_root: str | Path,
    flush_every: int = 256,
) -> dict[str, object]:
    paths = _paths(project_root)
    runtime = assert_runtime()
    external = verify_external_gate(project_root)
    benchmark_report = benchmark(project_root)
    if not benchmark_report["proceed"]:
        raise RuntimeError("E820 competition cache resource gate failed.")
    frame, indices = load_target_free_inputs(project_root)
    _configure_torch()
    model, tokenizer = _load_encoder(project_root, apply_delta=True)
    context = np.lib.format.open_memmap(
        paths["context_cache"],
        mode="r+" if paths["context_cache"].exists() else "w+",
        dtype=np.float32,
        shape=(PILOT_ROWS, 768),
    )
    completed = 0
    if paths["progress"].exists():
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if (
            progress.get("protocol_id") != PROTOCOL_ID
            or progress.get("delta_sha256") != EXTERNAL_DELTA_SHA256
        ):
            raise ValueError("E820 cache progress belongs to another lineage.")
        completed = int(progress["completed_context_rows"])
    started = time.perf_counter()
    peak = _current_rss_bytes()
    documents = frame["document"].astype(str).tolist()
    for start in range(completed, PILOT_ROWS, int(flush_every)):
        stop = min(PILOT_ROWS, start + int(flush_every))
        context[start:stop] = _encode(
            model,
            tokenizer,
            documents[start:stop],
            DOCUMENT_MAX_LENGTH,
        )
        context.flush()
        peak = max(peak, _current_rss_bytes())
        paths["progress"].write_text(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "delta_sha256": EXTERNAL_DELTA_SHA256,
                    "completed_context_rows": stop,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Encoded E820 competition contexts {stop}/{PILOT_ROWS}.",
            flush=True,
        )
    del context

    unique_queries = frame[["query"]].drop_duplicates().reset_index(drop=True)
    unique_embedding = _encode(
        model,
        tokenizer,
        unique_queries["query"].astype(str).tolist(),
        QUERY_MAX_LENGTH,
    )
    lookup = {
        text: embedding
        for text, embedding in zip(
            unique_queries["query"], unique_embedding, strict=True
        )
    }
    objective = np.vstack(
        [lookup[text] for text in frame["query"]]
    ).astype(np.float32)
    np.save(paths["objective_cache"], objective)
    paths["progress"].unlink(missing_ok=True)

    context_values = np.load(paths["context_cache"], mmap_mode="r")
    objective_values = np.load(paths["objective_cache"], mmap_mode="r")
    for name, values in (
        ("context", context_values),
        ("objective", objective_values),
    ):
        if (
            values.shape != (PILOT_ROWS, 768)
            or not np.isfinite(values).all()
            or not np.allclose(
                np.linalg.norm(values, axis=1),
                1.0,
                atol=2e-5,
                rtol=0,
            )
        ):
            raise RuntimeError(f"E820 {name} cache failed audit.")
    elapsed = time.perf_counter() - started
    metadata: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_run_id": EXTERNAL_RUN_ID,
        "external_report_sha256": EXTERNAL_REPORT_SHA256,
        "external_delta_sha256": EXTERNAL_DELTA_SHA256,
        "base_model_sha256": MODEL_SHA256,
        "pilot_indices_sha256": SOURCE_HASHES["indices"],
        "pilot_rows": PILOT_ROWS,
        "pilot_response_sha256": hashlib.sha256(
            "\n".join(frame["response_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "unique_objective_queries": len(unique_queries),
        "query_prefix": QUERY_PREFIX,
        "context_max_length": DOCUMENT_MAX_LENGTH,
        "objective_max_length": QUERY_MAX_LENGTH,
        "pooling": "normalized CLS",
        "cache_hashes": {
            "context": _sha256(paths["context_cache"]),
            "objective": _sha256(paths["objective_cache"]),
        },
        "cache_bytes": {
            "context": paths["context_cache"].stat().st_size,
            "objective": paths["objective_cache"].stat().st_size,
        },
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak,
        "runtime": runtime,
        "base_encoder_parity": benchmark_report["base_encoder_parity"],
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "passes_target_free_cache_gate": True,
        "external_gate_passed": external["passes_target_free_gate"],
        "metadata_sha256": _sha256(paths["metadata"]),
        "cache_hashes": metadata["cache_hashes"],
        "benchmark_sha256": benchmark_report["benchmark_sha256"],
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["target_free_report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "metadata": metadata,
        "target_free_report": report,
        "metadata_sha256": _sha256(paths["metadata"]),
        "target_free_report_sha256": _sha256(
            paths["target_free_report"]
        ),
    }


def verify_committed_binding(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    if not paths["binding"].is_file():
        raise FileNotFoundError(
            "Committed E820 competition cache binding is required."
        )
    binding = json.loads(paths["binding"].read_text(encoding="utf-8"))
    required = {
        "protocol_id": PROTOCOL_ID,
        "external_delta_sha256": EXTERNAL_DELTA_SHA256,
        "external_report_sha256": EXTERNAL_REPORT_SHA256,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    observed = {name: binding.get(name) for name in required}
    if observed != required:
        raise ValueError(f"E820 committed binding changed: {observed}")
    actual = {
        "context_cache_sha256": _sha256(paths["context_cache"]),
        "objective_cache_sha256": _sha256(paths["objective_cache"]),
        "metadata_sha256": _sha256(paths["metadata"]),
        "target_free_report_sha256": _sha256(paths["target_free_report"]),
    }
    expected = {name: binding.get(name) for name in actual}
    if actual != expected:
        raise ValueError(
            f"E820 bound competition cache hashes changed: {actual}"
        )
    report = json.loads(
        paths["target_free_report"].read_text(encoding="utf-8")
    )
    if (
        report.get("passes_target_free_cache_gate") is not True
        or report.get("competition_outcomes_accessed") is not False
    ):
        raise ValueError("E820 target-free cache report is ineligible.")
    return binding


def _candidate_features(
    context: np.ndarray, objective: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        context.shape != (PILOT_ROWS, 768)
        or objective.shape != context.shape
    ):
        raise ValueError("E820 candidate cache shape changed.")
    features = np.column_stack(
        [
            context * np.float32(0.7),
            objective * np.float32(0.7),
            context * objective * np.float32(6.0),
            np.abs(context - objective) * np.float32(0.7),
        ]
    )
    similarity = np.sum(context * objective, axis=1, keepdims=True)
    return features, similarity


def _raw_v05(component: pd.DataFrame) -> np.ndarray:
    return np.clip(
        sum(
            weight * component[name].to_numpy(dtype=np.float64)
            for name, weight in V05_WEIGHTS.items()
        ),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )


def _prediction_rows(
    component: pd.DataFrame,
    candidate_prediction: np.ndarray,
    base_pilot_prediction: np.ndarray,
    weights: Sequence[float],
) -> pd.DataFrame:
    baseline = _raw_v05(component)
    metadata = component[
        [
            "environment",
            "response_id",
            "session_id",
            "learning_objective_id",
            "semantic_family",
            "fold",
            "evaluation_eligible",
            "target",
        ]
    ].copy()
    rows: list[pd.DataFrame] = []
    for weight in weights:
        output = metadata.copy()
        output["candidate_weight"] = float(weight)
        output["pred_v05_raw"] = baseline
        output["pred_bge_base_pilot"] = base_pilot_prediction
        output["pred_e820_adapted"] = candidate_prediction
        output["prediction"] = np.clip(
            (1.0 - float(weight)) * baseline
            + float(weight) * candidate_prediction,
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def _metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (environment, weight, fold), group in predictions.groupby(
        ["environment", "candidate_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        candidate = binary_metrics(target, scored["prediction"])
        row: dict[str, object] = {
            "environment": str(environment),
            "candidate_weight": float(weight),
            "fold": int(fold),
            "rows": int(len(scored)),
        }
        for name, value in candidate.items():
            row[name] = value
            row[f"baseline_{name}"] = baseline[name]
            row[f"delta_{name}_vs_v05_raw"] = value - baseline[name]
        rows.append(row)
    folds = pd.DataFrame(rows)
    metric_names = (
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "prediction_mean",
    )
    columns = [
        *metric_names,
        *[f"baseline_{name}" for name in metric_names],
        *[f"delta_{name}_vs_v05_raw" for name in metric_names],
    ]
    environments = (
        folds.groupby(
            ["environment", "candidate_weight"],
            as_index=False,
            sort=True,
        )[columns]
        .mean()
        .sort_values(
            ["environment", "candidate_weight"], kind="stable"
        )
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def _select(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    environments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    bootstraps: dict[str, object] = {}
    for index, weight in enumerate(BLEND_WEIGHTS):
        metric = environments.loc[
            environments["candidate_weight"].eq(weight)
        ]
        selected = predictions.loc[
            predictions["candidate_weight"].eq(weight)
        ]
        session = _macro_cluster_bootstrap(
            selected,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="session_id",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + index,
        )
        family = _macro_cluster_bootstrap(
            selected,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="semantic_family",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + 100 + index,
        )
        bootstraps[f"{weight:.2f}"] = {
            "session": session,
            "semantic_family": family,
        }
        loss_delta = metric[
            "delta_log_loss_vs_v05_raw"
        ].to_numpy(dtype=np.float64)
        weight_folds = folds.loc[folds["candidate_weight"].eq(weight)]
        rows.append(
            {
                "candidate_weight": weight,
                "mean_log_loss": float(metric["log_loss"].mean()),
                "mean_log_loss_gain_vs_v05_raw": float(-loss_delta.mean()),
                "improved_environments": int(np.sum(loss_delta < 0)),
                "worst_environment_delta_log_loss_vs_v05_raw": float(
                    loss_delta.max()
                ),
                "worst_fold_delta_log_loss_vs_v05_raw": float(
                    weight_folds[
                        "delta_log_loss_vs_v05_raw"
                    ].max()
                ),
                "mean_delta_roc_auc_vs_v05_raw": float(
                    metric["delta_roc_auc_vs_v05_raw"].mean()
                ),
                "mean_delta_brier_score_vs_v05_raw": float(
                    metric["delta_brier_score_vs_v05_raw"].mean()
                ),
                "mean_delta_ece_10_vs_v05_raw": float(
                    metric["delta_ece_10_vs_v05_raw"].mean()
                ),
                "session_bootstrap_support": session[
                    "support_positive_gain"
                ],
                "semantic_family_bootstrap_support": family[
                    "support_positive_gain"
                ],
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "candidate_weight"], kind="stable"
    )
    selected = selection.iloc[0].to_dict()
    clauses = {
        "mean_log_loss_gain_at_least_0_0016": (
            selected["mean_log_loss_gain_vs_v05_raw"] >= 0.0016
        ),
        "all_three_environments_improve": (
            selected["improved_environments"]
            == len(SELECTION_ENVIRONMENTS)
        ),
        "no_environment_log_loss_regression": (
            selected[
                "worst_environment_delta_log_loss_vs_v05_raw"
            ]
            <= 0
        ),
        "worst_fold_regression_at_most_0_0005": (
            selected["worst_fold_delta_log_loss_vs_v05_raw"] <= 0.0005
        ),
        "macro_auroc_non_regression": (
            selected["mean_delta_roc_auc_vs_v05_raw"] >= 0
        ),
        "macro_brier_non_regression": (
            selected["mean_delta_brier_score_vs_v05_raw"] <= 0
        ),
        "macro_ece_non_regression": (
            selected["mean_delta_ece_10_vs_v05_raw"] <= 0
        ),
        "session_bootstrap_support_at_least_0_95": (
            selected["session_bootstrap_support"] >= 0.95
        ),
        "semantic_family_bootstrap_support_at_least_0_95": (
            selected["semantic_family_bootstrap_support"] >= 0.95
        ),
    }
    key = f"{selected['candidate_weight']:.2f}"
    return selection.reset_index(drop=True), {
        "selected_weight": float(selected["candidate_weight"]),
        "selected_row": selected,
        "clauses": clauses,
        "passes_selection_gate": bool(all(clauses.values())),
        "selected_bootstrap": bootstraps[key],
        "all_preregistered_bootstraps": bootstraps,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def _environment_predictions(
    project_root: str | Path,
    environment: str,
    indices: np.ndarray,
    frame: pd.DataFrame,
    candidate_features: np.ndarray,
    candidate_similarity: np.ndarray,
    base_features: np.ndarray,
    base_similarity: np.ndarray,
    dense_full: np.ndarray,
    weights: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    component_full = load_component_oof(project_root, environment)
    component = component_full.iloc[indices].reset_index(drop=True)
    if list(component["response_id"].astype(str)) != list(
        frame["response_id"].astype(str)
    ):
        raise ValueError(f"E820 response order differs in {environment}.")
    folds = component["fold"].to_numpy(dtype=np.int8)
    candidate_dense = dense_full[indices].copy()
    candidate_dense[:, -1:] = candidate_similarity
    base_dense = dense_full[indices].copy()
    base_dense[:, -1:] = base_similarity
    candidate_prediction = semantic_logistic_oof(
        frame,
        folds,
        candidate_features,
        candidate_dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    base_prediction = semantic_logistic_oof(
        frame,
        folds,
        base_features,
        base_dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    return (
        _prediction_rows(
            component,
            candidate_prediction,
            base_prediction,
            weights,
        ),
        {
            "environment": environment,
            "pilot_rows": len(component),
            "fold_counts": {
                str(key): int(value)
                for key, value in component["fold"].value_counts(
                    sort=False
                ).sort_index().items()
            },
            "evaluation_eligible_rows": int(
                component["evaluation_eligible"].astype(bool).sum()
            ),
            "candidate_standalone": binary_metrics(
                component["target"].to_numpy(dtype=np.int8),
                candidate_prediction,
            ),
            "base_pilot_standalone": binary_metrics(
                component["target"].to_numpy(dtype=np.int8),
                base_prediction,
            ),
        },
    )


def _confirm_joint(
    project_root: str | Path,
    indices: np.ndarray,
    frame: pd.DataFrame,
    candidate_features: np.ndarray,
    candidate_similarity: np.ndarray,
    base_features: np.ndarray,
    base_similarity: np.ndarray,
    dense_full: np.ndarray,
    selected_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions, diagnostics = _environment_predictions(
        project_root,
        CONFIRMATION_ENVIRONMENT,
        indices,
        frame,
        candidate_features,
        candidate_similarity,
        base_features,
        base_similarity,
        dense_full,
        [selected_weight],
    )
    folds, environments = _metric_tables(predictions)
    row = environments.iloc[0]
    session = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v05_raw",
        group_column="session_id",
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1_000,
    )
    family = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v05_raw",
        group_column="semantic_family",
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1_100,
    )
    clauses = {
        "joint_log_loss_gain_at_least_0_0010": (
            -float(row["delta_log_loss_vs_v05_raw"]) >= 0.001
        ),
        "joint_worst_fold_regression_at_most_0_0005": (
            float(folds["delta_log_loss_vs_v05_raw"].max()) <= 0.0005
        ),
        "joint_auroc_non_regression": (
            float(row["delta_roc_auc_vs_v05_raw"]) >= 0
        ),
        "joint_brier_non_regression": (
            float(row["delta_brier_score_vs_v05_raw"]) <= 0
        ),
        "joint_ece_non_regression": (
            float(row["delta_ece_10_vs_v05_raw"]) <= 0
        ),
        "joint_session_bootstrap_support_at_least_0_95": (
            session["support_positive_gain"] >= 0.95
        ),
        "joint_semantic_family_bootstrap_support_at_least_0_95": (
            family["support_positive_gain"] >= 0.95
        ),
    }
    return predictions, folds, {
        "environment_metrics": row.to_dict(),
        "diagnostics": diagnostics,
        "session_bootstrap": session,
        "semantic_family_bootstrap": family,
        "clauses": clauses,
        "passes_confirmation_gate": bool(all(clauses.values())),
        "V_joint_accessed": True,
        "V_final_accessed": False,
    }


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    binding = verify_committed_binding(project_root)
    frame_target_free, indices = load_target_free_inputs(project_root)
    paths = _paths(project_root)
    modeling = pd.read_parquet(paths["modeling"]).reset_index(drop=True)
    frame = modeling.iloc[indices].reset_index(drop=True)
    if list(frame["response_id"]) != list(frame_target_free["response_id"]):
        raise ValueError("E820 outcome frame differs from target-free lineage.")
    context = np.load(paths["context_cache"]).astype(np.float32)
    objective = np.load(paths["objective_cache"]).astype(np.float32)
    candidate_features, candidate_similarity = _candidate_features(
        context, objective
    )
    base_features_full, base_similarity_full = prepare_bge_base_features(
        paths["modeling"].parent, modeling
    )
    base_features = base_features_full[indices]
    base_similarity = base_similarity_full[indices]
    _, dense_full, dense_names = prepare_semantic_features(
        paths["modeling"].parent, modeling
    )

    prediction_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}
    for environment in SELECTION_ENVIRONMENTS:
        predictions, detail = _environment_predictions(
            project_root,
            environment,
            indices,
            frame,
            candidate_features,
            candidate_similarity,
            base_features,
            base_similarity,
            dense_full,
            BLEND_WEIGHTS,
        )
        prediction_frames.append(predictions)
        diagnostics[environment] = detail
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics, environment_metrics = _metric_tables(predictions)
    selection, decision = _select(
        predictions, fold_metrics, environment_metrics
    )

    joint_predictions: pd.DataFrame | None = None
    joint_folds: pd.DataFrame | None = None
    joint: dict[str, object] = {
        "reason": "Selection gate failed; V_joint remained sealed.",
        "passes_confirmation_gate": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    if decision["passes_selection_gate"]:
        joint_predictions, joint_folds, joint = _confirm_joint(
            project_root,
            indices,
            frame,
            candidate_features,
            candidate_similarity,
            base_features,
            base_similarity,
            dense_full,
            float(decision["selected_weight"]),
        )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime(
        "%Y%m%dT%H%M%SZ_mathdial_contrastive_competition"
    )
    run_dir = paths["experiments"] / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, pd.DataFrame | dict[str, object]] = {
        "development_predictions.parquet": predictions,
        "development_fold_metrics.csv": fold_metrics,
        "development_environment_metrics.csv": environment_metrics,
        "development_selection.csv": selection,
        "development_bootstraps.json": decision[
            "all_preregistered_bootstraps"
        ],
    }
    if joint_predictions is not None and joint_folds is not None:
        artifacts.update(
            {
                "joint_confirmation_predictions.parquet": joint_predictions,
                "joint_confirmation_fold_metrics.csv": joint_folds,
                "joint_confirmation.json": joint,
            }
        )
    artifact_hashes: dict[str, str] = {}
    for name, value in artifacts.items():
        path = run_dir / name
        if name.endswith(".parquet"):
            assert isinstance(value, pd.DataFrame)
            value.to_parquet(path, index=False)
        elif name.endswith(".csv"):
            assert isinstance(value, pd.DataFrame)
            value.to_csv(path, index=False, lineterminator="\n")
        else:
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        artifact_hashes[name] = _sha256(path)

    accepted = bool(
        decision["passes_selection_gate"]
        and joint["passes_confirmation_gate"]
    )
    gain = float(
        decision["selected_row"]["mean_log_loss_gain_vs_v05_raw"]
    )
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E820_mathdial_contrastive_alignment",
        "decision": "accept" if accepted else "reject",
        "passes_all_frozen_gates": accepted,
        "external_run_id": EXTERNAL_RUN_ID,
        "external_report_sha256": EXTERNAL_REPORT_SHA256,
        "external_delta_sha256": EXTERNAL_DELTA_SHA256,
        "cache_binding": binding,
        "lineage": {
            "pilot_rows": PILOT_ROWS,
            "pilot_indices_sha256": SOURCE_HASHES["indices"],
            "query_prefix": QUERY_PREFIX,
            "context_max_length": DOCUMENT_MAX_LENGTH,
            "query_max_length": QUERY_MAX_LENGTH,
            "feature_geometry": (
                "0.7 context, 0.7 objective, 6.0 product, "
                "0.7 absolute difference"
            ),
            "dense_features": dense_names,
            "estimator": "fold-local LogisticRegression",
            "regularization_c": REGULARIZATION_C,
            "seed": SEED,
            "raw_v05_weights": V05_WEIGHTS,
            "blend_weights": list(BLEND_WEIGHTS),
            "selection": (
                "lowest equal-environment equal-fold macro log loss; "
                "ties use smaller weight"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_groups": ["session_id", "semantic_family"],
        },
        "environment_diagnostics": diagnostics,
        "selection_gate": decision,
        "joint_confirmation": joint,
        "local_mean_log_loss_gain": gain,
        "projected_public_log_loss": 0.6054 - gain,
        "runtime": runtime,
        "validation_runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": artifact_hashes,
        "competition_outcomes_accessed": True,
        "V_seen_accessed": True,
        "V_objective_accessed": True,
        "V_style_accessed": True,
        "V_joint_accessed": bool(joint["V_joint_accessed"]),
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "decision": report["decision"],
        "passes_all_frozen_gates": accepted,
        "selection_gate": decision,
        "joint_confirmation": joint,
        "report_sha256": _sha256(report_path),
        "artifact_sha256": artifact_hashes,
        "validation_runtime_seconds": report[
            "validation_runtime_seconds"
        ],
        "V_joint_accessed": report["V_joint_accessed"],
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen E820 competition cache and validation."
    )
    parser.add_argument(
        "stage", choices=("benchmark", "build-cache", "validate")
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build-cache":
        result = build_cache(args.project_root)
    else:
        result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
