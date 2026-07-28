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
    _encode,
    assert_runtime,
)
from trace_ace.io import discover_project_paths
from trace_ace.simulator_outcome_transfer import (
    _load_encoder_left,
    _sha256,
)
from trace_ace.simulator_quality_transfer import (
    _group_bootstrap,
    _regression_metrics,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E690_essayjudge_quality_v1"
SOURCE_COMMIT = "e5ee947e97231d03d1341092e187ef0384da219e"
SOURCE_SHA256 = (
    "7998f78165f3b354f42333ad25dc75ee6a33bfa3aa6ffb8c07650fc53c29dea5"
)
LICENSE_DECLARATION = "competition-platform-catalog:Apache-2.0"
EXPECTED_ROWS = 1_054
EXPECTED_QUESTIONS = 125
EXPECTED_COMPONENTS = 123
FOLDS = 5
RIDGE_ALPHA = 10.0
MAX_PROJECTED_HOURS = 1.0
TASK_LINE = "Task: represent the overall quality of this student response."
TRAIT_COLUMNS = (
    "argument_clarity(ground_truth)",
    "justifying_persuasiveness(ground_truth)",
    "organizational_structure(ground_truth)",
    "coherence(ground_truth)",
    "essay_length(ground_truth)",
    "grammatical_accuracy(ground_truth)",
    "grammatical_diversity(ground_truth)",
    "lexical_accuracy(ground_truth)",
    "lexical_diversity(ground_truth)",
    "punctuation_accuracy(ground_truth)",
)
CANONICAL_CONTENT_SHA256 = (
    "9a88708f39ba9230a6315a088e5693b55111d7e51c9664732ab4ba49da00ef3b"
)
CANONICAL_PARQUET_SHA256 = (
    "a5f57aef306add56855d3411710552f5e4d48776c92c3ed3e3e5f07907cd7fad"
)
GATE_THRESHOLDS = {
    "pearson": 0.35,
    "spearman": 0.35,
    "rmse": 0.10,
    "rmse_gain": 0.005,
    "mae_gain": 0.005,
    "positive_fold_gains": 5,
    "component_bootstrap_support": 0.95,
    "type_bootstrap_support": 0.95,
}


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    return {
        "source": paths.root / "Datasets" / "EssayJudge" / "data.csv",
        "canonical": paths.cache_dir / "essayjudge_e690_canonical.parquet",
        "canonical_metadata": (
            paths.cache_dir / "essayjudge_e690_canonical.metadata.json"
        ),
        "benchmark": paths.cache_dir / "essayjudge_e690_benchmark.json",
        "embeddings": paths.cache_dir / "essayjudge_e690_embeddings.npy",
        "embedding_metadata": (
            paths.cache_dir / "essayjudge_e690_embeddings.metadata.json"
        ),
        "progress": paths.cache_dir / "essayjudge_e690_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def verify_source(project_root: str | Path) -> None:
    source = _paths(project_root)["source"]
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError("E690 source CSV SHA-256 changed.")


def load_source(project_root: str | Path) -> pd.DataFrame:
    verify_source(project_root)
    frame = pd.read_csv(_paths(project_root)["source"], encoding="gb18030")
    required = {"Question", "Essay", "Type", *TRAIT_COLUMNS}
    if len(frame) != EXPECTED_ROWS or not required.issubset(frame.columns):
        raise ValueError("E690 source shape or schema changed.")
    if frame[list(required)].isna().any().any():
        raise ValueError("E690 source contains missing required values.")
    if int(frame["Question"].nunique()) != EXPECTED_QUESTIONS:
        raise ValueError("E690 prompt count changed.")
    return frame


def canonical_text(question: object, essay: object) -> str:
    prompt = " ".join(str(question).split())
    response = " ".join(str(essay).split())
    if not prompt or not response:
        raise ValueError("E690 prompt or response is empty.")
    return (
        f"Prompt: {prompt}\n"
        f"Student response: {response}\n"
        f"{TASK_LINE}"
    )


def prompt_component_map(
    frame: pd.DataFrame, expected_components: int = EXPECTED_COMPONENTS
) -> dict[str, str]:
    questions = sorted(frame["Question"].astype(str).unique())
    parent = {question: question for question in questions}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            parent[larger] = smaller

    for _, duplicate_rows in frame.groupby("Essay", sort=False):
        linked = sorted(duplicate_rows["Question"].astype(str).unique())
        for question in linked[1:]:
            union(linked[0], question)
    result = {question: find(question) for question in questions}
    if len(set(result.values())) != expected_components:
        raise ValueError("E690 prompt-component count changed.")
    return result


def fold_for_component(component_id: str) -> int:
    digest = hashlib.sha256(
        f"E690|prompt_component|{component_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % FOLDS


def _fold_masks(
    frame: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = frame["fold"].eq(fold).to_numpy()
    train = ~valid
    return train, valid


def _audit_folds(frame: pd.DataFrame) -> list[dict[str, int]]:
    audits: list[dict[str, int]] = []
    for fold in range(FOLDS):
        train, valid = _fold_masks(frame, fold)
        train_components = set(frame.loc[train, "component_id"])
        valid_components = set(frame.loc[valid, "component_id"])
        train_essays = set(frame.loc[train, "essay_sha256"])
        valid_essays = set(frame.loc[valid, "essay_sha256"])
        if (
            train_components & valid_components
            or train_essays & valid_essays
            or not train.any()
            or not valid.any()
        ):
            raise ValueError(f"E690 fold {fold} leaks a component or essay.")
        audits.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "train_components": len(train_components),
                "validation_components": len(valid_components),
                "component_overlap": 0,
                "essay_overlap": 0,
            }
        )
    return audits


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "source_index",
        "example_id",
        "component_id",
        "essay_sha256",
        "fold",
        "graph_type",
        "target",
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


def build_canonical(project_root: str | Path) -> dict[str, object]:
    source = load_source(project_root)
    paths = _paths(project_root)
    components = prompt_component_map(source)
    trait_values = source[list(TRAIT_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(trait_values).all() or np.any(
        (trait_values < 0.0) | (trait_values > 5.0)
    ):
        raise ValueError("E690 trait values changed.")
    records: list[dict[str, object]] = []
    for source_index, row in source.iterrows():
        question = str(row["Question"])
        essay = str(row["Essay"])
        component_id = components[question]
        essay_sha256 = hashlib.sha256(essay.encode("utf-8")).hexdigest()
        identity = hashlib.sha256(
            f"E690|{source_index}|{question}|{essay}".encode("utf-8")
        ).hexdigest()
        target = float(
            np.mean([float(row[column]) for column in TRAIT_COLUMNS]) / 5.0
        )
        records.append(
            {
                "source_index": int(source_index),
                "example_id": identity,
                "component_id": component_id,
                "essay_sha256": essay_sha256,
                "fold": fold_for_component(component_id),
                "graph_type": str(row["Type"]),
                "target": target,
                "text": canonical_text(question, essay),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_ROWS or frame["example_id"].duplicated().any():
        raise ValueError("E690 canonical identity audit failed.")
    fold_audit = _audit_folds(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_declaration": LICENSE_DECLARATION,
        "rows": len(frame),
        "questions": int(source["Question"].nunique()),
        "components": int(frame["component_id"].nunique()),
        "target_mean": float(frame["target"].mean()),
        "target_std": float(frame["target"].std(ddof=0)),
        "target_min": float(frame["target"].min()),
        "target_max": float(frame["target"].max()),
        "fold_audit": fold_audit,
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "score_or_target_input": False,
        "graph_or_image_input": False,
        "chart_type_input": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["canonical_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_canonical(project_root: str | Path) -> dict[str, object]:
    verify_source(project_root)
    if not CANONICAL_CONTENT_SHA256 or not CANONICAL_PARQUET_SHA256:
        raise RuntimeError("E690 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E690 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E690 canonical ordered-content SHA-256 changed.")
    _audit_folds(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"], columns=["example_id", "text"])
    order = frame["example_id"].map(
        lambda value: hashlib.sha256(
            f"E690-benchmark|{value}".encode("utf-8")
        ).hexdigest()
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
        "candidate_target_or_metric_accessed": False,
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
        raise ValueError("E690 benchmark did not authorize cache construction.")
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
            print(f"e690_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0)
    ):
        raise ValueError("E690 embedding cache audit failed.")
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


def external_gate(
    metrics: dict[str, float],
    rmse_gain: float,
    mae_gain: float,
    fold_gains: Sequence[float],
    component_support: float,
    type_support: float,
) -> dict[str, bool]:
    return {
        "pearson": metrics["pearson"] >= GATE_THRESHOLDS["pearson"],
        "spearman": metrics["spearman"] >= GATE_THRESHOLDS["spearman"],
        "rmse": metrics["rmse"] <= GATE_THRESHOLDS["rmse"],
        "rmse_gain": rmse_gain >= GATE_THRESHOLDS["rmse_gain"],
        "mae_gain": mae_gain >= GATE_THRESHOLDS["mae_gain"],
        "positive_fold_gains": (
            sum(gain > 0 for gain in fold_gains)
            >= GATE_THRESHOLDS["positive_fold_gains"]
        ),
        "component_bootstrap_support": (
            component_support
            >= GATE_THRESHOLDS["component_bootstrap_support"]
        ),
        "type_bootstrap_support": (
            type_support >= GATE_THRESHOLDS["type_bootstrap_support"]
        ),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import Ridge

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(
        paths["embedding_metadata"].read_text(encoding="utf-8")
    )
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E690 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    if embeddings.shape != (len(frame), EMBEDDING_DIMENSION):
        raise ValueError("E690 embedding cache shape changed.")
    target = frame["target"].to_numpy(dtype=np.float64)
    prediction = np.full(len(frame), np.nan)
    prior = np.full(len(frame), np.nan)
    fold_rows: list[dict[str, object]] = []
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    for fold in range(FOLDS):
        train, valid = _fold_masks(frame, fold)
        model = Ridge(
            alpha=RIDGE_ALPHA,
            fit_intercept=True,
            solver="lsqr",
            max_iter=1000,
            tol=1e-6,
        )
        model.fit(embeddings[train], target[train])
        prediction[valid] = np.clip(model.predict(embeddings[valid]), 0.0, 1.0)
        prior[valid] = float(target[train].mean())
        candidate_metrics = _regression_metrics(target[valid], prediction[valid])
        prior_rmse = float(np.mean((target[valid] - prior[valid]) ** 2) ** 0.5)
        prior_mae = float(np.mean(np.abs(target[valid] - prior[valid])))
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "candidate": candidate_metrics,
                "prior_rmse": prior_rmse,
                "prior_mae": prior_mae,
                "rmse_gain": prior_rmse - candidate_metrics["rmse"],
            }
        )
        coefficients.append(np.asarray(model.coef_, dtype=np.float64))
        intercepts.append(float(model.intercept_))
    if not np.isfinite(prediction).all() or not np.isfinite(prior).all():
        raise ValueError("E690 OOF predictions are incomplete.")
    metrics = _regression_metrics(target, prediction)
    prior_rmse = float(np.mean((target - prior) ** 2) ** 0.5)
    prior_mae = float(np.mean(np.abs(target - prior)))
    rmse_gain = prior_rmse - metrics["rmse"]
    mae_gain = prior_mae - metrics["mae"]
    squared_error_gain = (target - prior) ** 2 - (target - prediction) ** 2
    component_bootstrap = _group_bootstrap(
        squared_error_gain, frame["component_id"], salt=6901
    )
    type_bootstrap = _group_bootstrap(
        squared_error_gain, frame["graph_type"], salt=6902
    )
    fold_gains = [float(row["rmse_gain"]) for row in fold_rows]
    clauses = external_gate(
        metrics,
        rmse_gain,
        mae_gain,
        fold_gains,
        float(component_bootstrap["support"]),
        float(type_bootstrap["support"]),
    )
    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_essayjudge_quality_transfer"
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = frame[
        [
            "source_index",
            "example_id",
            "component_id",
            "essay_sha256",
            "fold",
            "graph_type",
            "target",
        ]
    ].copy()
    predictions["prediction"] = prediction
    predictions["prior_prediction"] = prior
    predictions_path = run_dir / "oof_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    model_path = run_dir / "fold_models.npz"
    np.savez_compressed(
        model_path,
        coefficients=np.vstack(coefficients),
        intercepts=np.asarray(intercepts),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "rows": len(frame),
        "metrics": metrics,
        "prior_rmse": prior_rmse,
        "prior_mae": prior_mae,
        "rmse_gain_vs_prior": rmse_gain,
        "mae_gain_vs_prior": mae_gain,
        "folds": fold_rows,
        "component_bootstrap": component_bootstrap,
        "type_bootstrap": type_bootstrap,
        "thresholds": GATE_THRESHOLDS,
        "clauses": clauses,
        "passes_external_gate": all(clauses.values()),
        "runtime": runtime,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "license_declaration": LICENSE_DECLARATION,
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
        description="Run the frozen E690 EssayJudge response-quality transfer."
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
