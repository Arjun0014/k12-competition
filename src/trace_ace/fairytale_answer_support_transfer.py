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
from trace_ace.metrics import binary_metrics
from trace_ace.simulator_outcome_transfer import (
    _collapse,
    _load_encoder_left,
    _sha256,
)
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E700_fairytale_answer_support_v1"
SOURCE_COMMIT = "a24ddc17364666b7c13a425b9970c87368b03417"
LICENSE_SHA256 = (
    "335d2c093cbb191de9693d87c9935f832a2442a67615a6e47ac71f74ccb4bd5d"
)
SOURCE_HASHES = {
    "train": "b778e56bfc9cd9337eed02a8a0216eb7c5ff9deef96edc2fe49911924c9a79ab",
    "valid": "b2dfd4dc5e99cf903e412fa785357138f844202222771674c446a4e41acb9abf",
    "test": "34acae1b2b62469fb616ed55ea049c25185838f2388c60ef28c3f810e512a54b",
}
EXPECTED_SOURCE_ROWS = {"train": 8_548, "valid": 1_025, "test": 1_007}
EXPECTED_SOURCE_STORIES = {"train": 232, "valid": 23, "test": 23}
TRAIN_POSITIVES = 4_096
EXPECTED_CANONICAL_ROWS = 12_256
REGULARIZATION_C = 0.1
MAX_PROJECTED_HOURS = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
TASK_LINE = (
    "Task: determine whether the student response is supported by the context "
    "and answers the question."
)
CANONICAL_CONTENT_SHA256 = (
    "2d4fb1f185b3931e9e00020ed835660ead7bfd5b0315adf91a2522ebb061beb2"
)
CANONICAL_PARQUET_SHA256 = (
    "f64cc3c9c7ac6f744e17e318dd6a2bbe7222cc370c5117b88cc25748724c881a"
)
GATE_THRESHOLDS = {
    "roc_auc": 0.70,
    "macro_f1": 0.65,
    "log_loss_gain": 0.030,
    "brier_gain": 0.010,
    "ece_10": 0.10,
    "bootstrap_support": 0.95,
}


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "FairytaleQA"
    return {
        "source_root": source_root,
        "train": source_root / "train.csv",
        "valid": source_root / "valid.csv",
        "test": source_root / "test.csv",
        "canonical": paths.cache_dir / "fairytale_e700_canonical.parquet",
        "canonical_metadata": (
            paths.cache_dir / "fairytale_e700_canonical.metadata.json"
        ),
        "benchmark": paths.cache_dir / "fairytale_e700_benchmark.json",
        "embeddings": paths.cache_dir / "fairytale_e700_embeddings.npy",
        "embedding_metadata": (
            paths.cache_dir / "fairytale_e700_embeddings.metadata.json"
        ),
        "progress": paths.cache_dir / "fairytale_e700_embeddings.progress.json",
        "runs": paths.experiments_dir / "runs",
    }


def load_sources(project_root: str | Path) -> dict[str, pd.DataFrame]:
    paths = _paths(project_root)
    required = {
        "story_name",
        "story_section",
        "question",
        "answer1",
        "local_or_sum",
        "attribute",
        "ex_or_im",
    }
    result: dict[str, pd.DataFrame] = {}
    story_sets: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        if _sha256(paths[split]) != SOURCE_HASHES[split]:
            raise ValueError(f"E700 {split} source SHA-256 changed.")
        frame = pd.read_csv(paths[split]).reset_index(drop=True)
        if (
            len(frame) != EXPECTED_SOURCE_ROWS[split]
            or not required.issubset(frame.columns)
            or frame[list(required)].isna().any().any()
            or int(frame["story_name"].nunique()) != EXPECTED_SOURCE_STORIES[split]
        ):
            raise ValueError(f"E700 {split} source schema changed.")
        result[split] = frame
        story_sets[split] = set(frame["story_name"].astype(str))
    if (
        story_sets["train"] & story_sets["valid"]
        or story_sets["train"] & story_sets["test"]
        or story_sets["valid"] & story_sets["test"]
    ):
        raise ValueError("E700 official splits overlap by story.")
    return result


def canonical_text(
    context: object, question: object, candidate_answer: object
) -> str:
    context_text = _collapse(context)
    question_text = _collapse(question)
    answer_text = _collapse(candidate_answer)
    if not context_text or not question_text or not answer_text:
        raise ValueError("E700 canonical field is empty.")
    return (
        f"Context: {context_text}\n"
        f"Question: {question_text}\n"
        f"Student response: {answer_text}\n"
        f"{TASK_LINE}"
    )


def selected_train_indices(frame: pd.DataFrame) -> list[int]:
    order = [
        (
            hashlib.sha256(
                (
                    "E700|train|"
                    + str(row.story_name)
                    + "|"
                    + str(row.question)
                    + "|"
                    + str(row.answer1)
                ).encode("utf-8")
            ).hexdigest(),
            int(index),
        )
        for index, row in frame.iterrows()
    ]
    return [index for _, index in sorted(order)[:TRAIN_POSITIVES]]


def negative_source_indices(frame: pd.DataFrame, split: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for _, story_rows in frame.groupby("story_name", sort=False):
        indices = story_rows.index.to_list()
        if len(indices) < 2:
            raise ValueError("E700 story cannot provide a negative answer.")
        for position, source_index in enumerate(indices):
            digest = hashlib.sha256(
                f"E700|negative|{split}|{source_index}".encode("utf-8")
            ).hexdigest()
            offset = 1 + int(digest[:16], 16) % (len(indices) - 1)
            chosen: int | None = None
            positive = str(frame.at[source_index, "answer1"])
            for step in range(len(indices) - 1):
                candidate = indices[(position + offset + step) % len(indices)]
                if str(frame.at[candidate, "answer1"]) != positive:
                    chosen = int(candidate)
                    break
            if chosen is None:
                raise ValueError("E700 story has no distinct negative answer.")
            result[int(source_index)] = chosen
    return result


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "split",
        "source_index",
        "negative_source_index",
        "example_id",
        "story_name",
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


def _audit_canonical(frame: pd.DataFrame) -> dict[str, object]:
    if len(frame) != EXPECTED_CANONICAL_ROWS:
        raise ValueError("E700 canonical row count changed.")
    counts = (
        frame.groupby(["split", "target"]).size().astype(int).to_dict()
    )
    expected = {
        ("train", 0): TRAIN_POSITIVES,
        ("train", 1): TRAIN_POSITIVES,
        ("valid", 0): EXPECTED_SOURCE_ROWS["valid"],
        ("valid", 1): EXPECTED_SOURCE_ROWS["valid"],
        ("test", 0): EXPECTED_SOURCE_ROWS["test"],
        ("test", 1): EXPECTED_SOURCE_ROWS["test"],
    }
    if counts != expected or frame["example_id"].duplicated().any():
        raise ValueError("E700 canonical balance or identity changed.")
    story_sets = {
        split: set(frame.loc[frame["split"].eq(split), "story_name"])
        for split in ("train", "valid", "test")
    }
    if (
        story_sets["train"] & story_sets["valid"]
        or story_sets["train"] & story_sets["test"]
        or story_sets["valid"] & story_sets["test"]
    ):
        raise ValueError("E700 canonical splits overlap by story.")
    return {
        "split_target_counts": {
            f"{split}_{target}": count
            for (split, target), count in counts.items()
        },
        "split_stories": {
            split: len(stories) for split, stories in story_sets.items()
        },
        "story_overlap": 0,
    }


def build_canonical(project_root: str | Path) -> dict[str, object]:
    sources = load_sources(project_root)
    paths = _paths(project_root)
    records: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        source = sources[split]
        negatives = negative_source_indices(source, split)
        retained = (
            selected_train_indices(source)
            if split == "train"
            else source.index.to_list()
        )
        for source_index in retained:
            row = source.loc[source_index]
            negative_index = negatives[int(source_index)]
            for target, answer_index in (
                (1, int(source_index)),
                (0, negative_index),
            ):
                candidate = source.at[answer_index, "answer1"]
                identity = hashlib.sha256(
                    (
                        f"E700|{split}|{source_index}|{answer_index}|{target}|"
                        f"{row['story_name']}|{row['question']}|{candidate}"
                    ).encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "split": split,
                        "source_index": int(source_index),
                        "negative_source_index": (
                            -1 if target == 1 else negative_index
                        ),
                        "example_id": identity,
                        "story_name": str(row["story_name"]),
                        "target": target,
                        "text": canonical_text(
                            row["story_section"], row["question"], candidate
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    audit = _audit_canonical(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "source_hashes": SOURCE_HASHES,
        "license_sha256": LICENSE_SHA256,
        "rows": len(frame),
        "audit": audit,
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "metadata_label_input": False,
        "answer_support_label_input": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["canonical_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_canonical(project_root: str | Path) -> dict[str, object]:
    load_sources(project_root)
    if not CANONICAL_CONTENT_SHA256 or not CANONICAL_PARQUET_SHA256:
        raise RuntimeError("E700 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E700 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E700 canonical ordered-content SHA-256 changed.")
    _audit_canonical(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["canonical"], columns=["example_id", "text"])
    order = frame["example_id"].map(
        lambda value: hashlib.sha256(
            f"E700-benchmark|{value}".encode("utf-8")
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
        raise ValueError("E700 benchmark did not authorize cache construction.")
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
            print(f"e700_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=2e-5, rtol=0)
    ):
        raise ValueError("E700 embedding cache audit failed.")
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


def _story_bootstrap(
    gains: np.ndarray, stories: Sequence[object], salt: int
) -> dict[str, object]:
    values = np.asarray([str(value) for value in stories])
    unique = np.unique(values)
    sums = np.asarray(
        [gains[values == story].sum() for story in unique], dtype=np.float64
    )
    counts = np.asarray(
        [(values == story).sum() for story in unique], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + salt)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 64):
        stop = min(start + 64, BOOTSTRAP_REPLICATES)
        sample = rng.integers(
            0, len(unique), size=(stop - start, len(unique))
        )
        draws[start:stop] = (
            sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "stories": len(unique),
        "mean_log_loss_gain": float(draws.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "support": float(np.mean(draws > 0)),
    }


def external_gate(
    metrics: dict[str, float],
    prior_metrics: dict[str, float],
    bootstrap: dict[str, object],
) -> dict[str, bool]:
    return {
        "roc_auc": metrics["roc_auc"] >= GATE_THRESHOLDS["roc_auc"],
        "macro_f1": metrics["macro_f1"] >= GATE_THRESHOLDS["macro_f1"],
        "log_loss_gain": (
            prior_metrics["log_loss"] - metrics["log_loss"]
            >= GATE_THRESHOLDS["log_loss_gain"]
        ),
        "brier_gain": (
            prior_metrics["brier_score"] - metrics["brier_score"]
            >= GATE_THRESHOLDS["brier_gain"]
        ),
        "ece_10": metrics["ece_10"] <= GATE_THRESHOLDS["ece_10"],
        "bootstrap_support": (
            float(bootstrap["support"])
            >= GATE_THRESHOLDS["bootstrap_support"]
        ),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    metadata = json.loads(paths["embedding_metadata"].read_text(encoding="utf-8"))
    embedding_hash = _sha256(paths["embeddings"])
    if metadata.get("cache_sha256") != embedding_hash:
        raise ValueError("E700 embedding cache binding failed.")
    frame = pd.read_parquet(paths["canonical"]).reset_index(drop=True)
    embeddings = np.load(paths["embeddings"])
    if embeddings.shape != (len(frame), EMBEDDING_DIMENSION):
        raise ValueError("E700 embedding cache shape changed.")
    target = frame["target"].to_numpy(dtype=np.int8)
    train = frame["split"].eq("train").to_numpy()
    model = LogisticRegression(
        C=REGULARIZATION_C,
        solver="lbfgs",
        max_iter=1000,
        random_state=20260728,
    )
    model.fit(embeddings[train], target[train])
    split_results: dict[str, dict[str, object]] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split_index, split in enumerate(("valid", "test")):
        valid = frame["split"].eq(split).to_numpy()
        split_target = target[valid]
        probability = model.predict_proba(embeddings[valid])[:, 1]
        prior = np.full(valid.sum(), float(target[train].mean()))
        metrics = binary_metrics(split_target, probability)
        metrics["macro_f1"] = float(
            f1_score(split_target, probability >= 0.5, average="macro")
        )
        prior_metrics = binary_metrics(split_target, prior)
        target_float = split_target.astype(np.float64)
        candidate_loss = -(
            target_float * np.log(np.clip(probability, 1e-6, 1 - 1e-6))
            + (1 - target_float)
            * np.log1p(-np.clip(probability, 1e-6, 1 - 1e-6))
        )
        prior_loss = -(
            target_float * np.log(np.clip(prior, 1e-6, 1 - 1e-6))
            + (1 - target_float)
            * np.log1p(-np.clip(prior, 1e-6, 1 - 1e-6))
        )
        bootstrap = _story_bootstrap(
            prior_loss - candidate_loss,
            frame.loc[valid, "story_name"],
            salt=split_index + 1,
        )
        clauses = external_gate(metrics, prior_metrics, bootstrap)
        split_results[split] = {
            "rows": int(valid.sum()),
            "stories": int(frame.loc[valid, "story_name"].nunique()),
            "metrics": metrics,
            "prior_metrics": prior_metrics,
            "log_loss_gain": prior_metrics["log_loss"] - metrics["log_loss"],
            "brier_gain": prior_metrics["brier_score"]
            - metrics["brier_score"],
            "bootstrap": bootstrap,
            "clauses": clauses,
            "passes": all(clauses.values()),
        }
        predictions = frame.loc[
            valid,
            [
                "split",
                "source_index",
                "negative_source_index",
                "example_id",
                "story_name",
                "target",
            ],
        ].copy()
        predictions["probability"] = probability
        predictions["train_prior"] = prior
        prediction_frames.append(predictions)
    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ") + "_fairytale_answer_support"
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "external_predictions.parquet"
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        predictions_path, index=False
    )
    model_path = run_dir / "model.npz"
    np.savez_compressed(
        model_path,
        coefficient=model.coef_[0].astype(np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "rows": len(frame),
        "train_rows": int(train.sum()),
        "splits": split_results,
        "thresholds": GATE_THRESHOLDS,
        "passes_external_gate": all(
            value["passes"] for value in split_results.values()
        ),
        "runtime": runtime,
        "source_commit": SOURCE_COMMIT,
        "source_hashes": SOURCE_HASHES,
        "license_sha256": LICENSE_SHA256,
        "canonical_content_sha256": CANONICAL_CONTENT_SHA256,
        "canonical_parquet_sha256": CANONICAL_PARQUET_SHA256,
        "embedding_sha256": embedding_hash,
        "predictions_sha256": _sha256(predictions_path),
        "model_sha256": _sha256(model_path),
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
        description="Run the frozen E700 FairytaleQA answer-support transfer."
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
