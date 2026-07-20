from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_cache import compact_objective_context
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_DIRECTORY = "Qwen3-Embedding-0.6B"
BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
PILOT_ROWS = 4096
MAX_SEQUENCE_LENGTH = 128
OUTPUT_DIMENSION = 256
RANDOM_STATE = 20260717
TASK_INSTRUCTION = (
    "Given a K-12 learning objective, retrieve tutoring transcript evidence "
    "relevant to assessing student mastery"
)
DEFAULT_C_VALUES = (0.01, 0.03, 0.1)


def select_pilot_indices(folds: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(folds) != len(target):
        raise ValueError("Pilot fold and target rows must align.")
    if len(folds) < PILOT_ROWS:
        raise ValueError("Not enough rows for the registered Qwen pilot.")
    strata = np.asarray(
        [f"{int(fold)}_{int(label)}" for fold, label in zip(folds, target)]
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=PILOT_ROWS, random_state=RANDOM_STATE
    )
    selected, _ = next(splitter.split(np.zeros(len(strata)), strata))
    return np.sort(selected.astype(np.int64))


def _query_text(objective: str) -> str:
    return f"Instruct: {TASK_INSTRUCTION}\nQuery: {' '.join(str(objective).split())}"


def build_qwen_pilot_cache(
    project_root: str | Path,
    batch_size: int = 16,
    chunk_size: int = 128,
) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / BASELINE_RUN_ID / "oof_predictions.parquet"
    )
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(drop=True)
    if list(frame["response_id"]) != list(contexts["response_id"]):
        raise ValueError("Qwen pilot context rows do not align with modeling data.")
    if list(frame["response_id"]) != list(baseline["response_id"]):
        raise ValueError("Qwen pilot baseline rows do not align with modeling data.")
    indices = select_pilot_indices(
        baseline["fold"].to_numpy(dtype=np.int8), frame["target"].to_numpy(dtype=np.int8)
    )
    indices_path = paths.cache_dir / "qwen3_pilot_indices.npy"
    if indices_path.exists():
        cached_indices = np.load(indices_path)
        if not np.array_equal(cached_indices, indices):
            raise ValueError("Existing Qwen pilot selection does not match the frozen sample.")
    else:
        np.save(indices_path, indices)

    model_dir = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    model = SentenceTransformer(
        str(model_dir), device="cpu", truncate_dim=OUTPUT_DIMENSION
    )
    model.float()
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    if int(model.get_embedding_dimension()) != OUTPUT_DIMENSION:
        raise ValueError("Qwen pilot encoder returned an unexpected output dimension.")

    selected_contexts = [
        compact_objective_context(value)
        for value in contexts.iloc[indices]["objective_context"]
    ]
    selected_objectives = frame.iloc[indices]["learning_objective"].fillna("").astype(str)
    context_path = paths.cache_dir / "qwen3_pilot_context_128_d256.npy"
    objective_path = paths.cache_dir / "qwen3_pilot_objective_128_d256.npy"
    progress_path = paths.cache_dir / "qwen3_pilot_context_128_d256.progress.json"
    metadata_path = paths.cache_dir / "qwen3_pilot_128_d256.metadata.json"
    expected_shape = (PILOT_ROWS, OUTPUT_DIMENSION)
    if context_path.exists():
        context_values = np.lib.format.open_memmap(context_path, mode="r+")
        if context_values.shape != expected_shape:
            raise ValueError("Existing Qwen pilot context cache has the wrong shape.")
    else:
        context_values = np.lib.format.open_memmap(
            context_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("model") != MODEL_NAME:
            raise ValueError("Qwen pilot progress belongs to a different model.")
        completed = int(progress.get("completed_rows", 0))
    for start in range(completed, PILOT_ROWS, chunk_size):
        end = min(PILOT_ROWS, start + chunk_size)
        encoded = model.encode(
            selected_contexts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        context_values[start:end] = encoded
        context_values.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "completed_rows": end,
                    "total_rows": PILOT_ROWS,
                    "model": MODEL_NAME,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                    "output_dimension": OUTPUT_DIMENSION,
                    "task_instruction": TASK_INSTRUCTION,
                    "random_state": RANDOM_STATE,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Encoded Qwen pilot contexts for {end}/{PILOT_ROWS} rows.", flush=True)

    if not objective_path.exists():
        unique = selected_objectives.drop_duplicates().tolist()
        query_embeddings = model.encode(
            [_query_text(value) for value in unique],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        lookup = dict(zip(unique, query_embeddings))
        objective_values = np.vstack([lookup[value] for value in selected_objectives])
        np.save(objective_path, objective_values.astype(np.float32))

    del context_values
    context = np.load(context_path, mmap_mode="r")
    objective = np.load(objective_path, mmap_mode="r")
    for name, values in (("context", context), ("objective", objective)):
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Completed Qwen pilot {name} cache failed audit.")
    metadata_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "model_directory": MODEL_DIRECTORY,
                "license": "Apache-2.0",
                "pilot_protocol": PILOT_PROTOCOL,
                "pilot_rows": PILOT_ROWS,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "output_dimension": OUTPUT_DIMENSION,
                "task_instruction": TASK_INSTRUCTION,
                "random_state": RANDOM_STATE,
                "indices_path": indices_path.name,
                "context_path": context_path.name,
                "objective_path": objective_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress_path.unlink(missing_ok=True)
    return {
        "pilot_rows": PILOT_ROWS,
        "context_path": str(context_path),
        "objective_path": str(objective_path),
        "metadata_path": str(metadata_path),
    }


def _interaction(
    context: np.ndarray, objective: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def validate_qwen_pilot(
    project_root: str | Path,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    indices = np.load(paths.cache_dir / "qwen3_pilot_indices.npy")
    frame_full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    frame = frame_full.iloc[indices].reset_index(drop=True)
    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / BASELINE_RUN_ID / "oof_predictions.parquet"
    )
    baseline = baseline.loc[baseline["protocol"].eq(PILOT_PROTOCOL)].reset_index(drop=True)
    folds = baseline.iloc[indices]["fold"].to_numpy(dtype=np.int8)
    target = frame["target"].to_numpy(dtype=np.float64)

    small_variants, dense_full, dense_names = prepare_semantic_features(
        paths.cache_dir, frame_full
    )
    small_features = small_variants["semantic_interaction"][indices]
    small_similarity = dense_full[indices, -1:]
    base_features_full, base_similarity_full = prepare_bge_base_features(
        paths.cache_dir, frame_full
    )
    base_features = base_features_full[indices]
    base_similarity = base_similarity_full[indices]
    qwen_context = np.load(paths.cache_dir / "qwen3_pilot_context_128_d256.npy")
    qwen_objective = np.load(paths.cache_dir / "qwen3_pilot_objective_128_d256.npy")
    qwen_features, qwen_similarity = _interaction(qwen_context, qwen_objective)
    dense_base = dense_full[indices].copy()

    variants = {
        "bge_small": (small_features, small_similarity),
        "bge_base": (base_features, base_similarity),
        "qwen3_0.6b_d256": (qwen_features, qwen_similarity),
    }
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for encoder_name, (features, similarity) in variants.items():
        dense = dense_base.copy()
        dense[:, -1:] = similarity
        for c_value in c_values:
            prediction = semantic_logistic_oof(
                frame,
                folds,
                features,
                dense,
                regularization_c=float(c_value),
                n_splits=int(folds.max()) + 1,
            )
            metrics = binary_metrics(target, prediction)
            rows.append(
                {
                    "encoder": encoder_name,
                    "regularization_c": float(c_value),
                    **metrics,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "encoder": encoder_name,
                        "regularization_c": float(c_value),
                        "response_id": frame["response_id"],
                        "fold": folds,
                        "target": target,
                        "prediction": prediction,
                    }
                )
            )
    metrics = pd.DataFrame(rows).sort_values(
        ["log_loss", "roc_auc"], ascending=[True, False]
    )
    best = metrics.sort_values(["encoder", "log_loss", "roc_auc"]).groupby(
        "encoder", as_index=False
    ).first()
    qwen = best.loc[best["encoder"].eq("qwen3_0.6b_d256")].iloc[0]
    base = best.loc[best["encoder"].eq("bge_base")].iloc[0]
    decision = {
        "qwen_delta_log_loss_vs_bge_base": float(qwen["log_loss"] - base["log_loss"]),
        "qwen_delta_roc_auc_vs_bge_base": float(qwen["roc_auc"] - base["roc_auc"]),
    }
    decision["continue_full_cache"] = bool(
        (
            decision["qwen_delta_log_loss_vs_bge_base"] <= -0.0015
            and decision["qwen_delta_roc_auc_vs_bge_base"] >= 0.0
        )
        or (
            decision["qwen_delta_roc_auc_vs_bge_base"] >= 0.005
            and decision["qwen_delta_log_loss_vs_bge_base"] <= 0.0
        )
    )
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_qwen_pilot")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    best.to_csv(run_dir / "best_by_encoder.csv", index=False, lineterminator="\n")
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "selection": "fixed fold/label-balanced subsample",
        "model": MODEL_NAME,
        "license": "Apache-2.0",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "output_dimension": OUTPUT_DIMENSION,
        "task_instruction": TASK_INSTRUCTION,
        "regularization_c": list(c_values),
        "dense_features": dense_names,
        "decision": decision,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "metrics": metrics, "decision": decision}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or validate the frozen Qwen pilot.")
    parser.add_argument("command", choices=["build", "validate"])
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "build":
        result = build_qwen_pilot_cache(
            args.project_root, batch_size=args.batch_size, chunk_size=args.chunk_size
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = validate_qwen_pilot(args.project_root)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
        print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
