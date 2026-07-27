from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E620_frozen_multilingual_mastery_v1"
MODEL_DIRECTORY = "multilingual-e5-small"
MAX_LENGTH = 384
BATCH_SIZE = 16
EMBEDDING_DIMENSION = 384
MAX_PROJECTED_HOURS = 4.0
MAX_RSS_BYTES = 8 * 1024**3
SOURCE_HASHES = {
    "model": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    "config": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "tokenizer": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "spm": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
}
MCD_CACHE_SHA256 = "4180bbb1829214c569f084e3387eb38352d5b241e03f217736305b0c97d4dc2f"
BOOTSTRAP_REPLICATES = 2_000


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
        "model_dir": model,
        "model": model / "model.safetensors",
        "config": model / "config.json",
        "tokenizer": model / "tokenizer.json",
        "spm": model / "sentencepiece.bpe.model",
        "mcd_cache": paths.cache_dir / "mcd_mastery_e610.parquet",
        "benchmark": paths.cache_dir / "frozen_multilingual_e620_benchmark.json",
        "external_embeddings": paths.cache_dir / "frozen_multilingual_e620_mcd.npy",
        "external_metadata": paths.cache_dir / "frozen_multilingual_e620_mcd.metadata.json",
        "progress": paths.cache_dir / "frozen_multilingual_e620_mcd.progress.json",
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _paths(project_root)
    for name, expected in SOURCE_HASHES.items():
        observed = _sha256(paths[name])
        if observed != expected:
            raise ValueError(f"E620 {name} SHA-256 changed: {observed}")
    if _sha256(paths["mcd_cache"]) != MCD_CACHE_SHA256:
        raise ValueError("E620 canonical MCD cache SHA-256 changed.")
    return dict(SOURCE_HASHES)


def mean_pool(hidden, attention_mask):
    import torch

    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    if not torch.isfinite(pooled).all():
        raise ValueError("E620 pooling produced non-finite values.")
    return pooled


def _load_model(project_root: str | Path):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    paths = _paths(project_root)
    tokenizer = AutoTokenizer.from_pretrained(paths["model_dir"])
    tokenizer.truncation_side = "left"
    model = AutoModel.from_pretrained(paths["model_dir"])
    model.eval()
    for value in model.parameters():
        value.requires_grad = False
    return tokenizer, model


def _fixed_benchmark_frame(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for split in ("train", "test"):
        subset = frame.loc[frame["split"].eq(split)].copy()
        subset["_order"] = subset["example_id"].map(
            lambda value: hashlib.sha256(
                f"E620|{value}".encode("utf-8")
            ).hexdigest()
        )
        pieces.append(subset.sort_values("_order", kind="mergesort").head(32))
    return pd.concat(pieces, ignore_index=True)


def benchmark(project_root: str | Path) -> dict[str, object]:
    import sklearn
    import torch
    import transformers

    source_hashes = verify_sources(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(paths["mcd_cache"])
    sample = _fixed_benchmark_frame(frame)
    tokenizer, model = _load_model(project_root)
    started = time.perf_counter()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sample), BATCH_SIZE):
            texts = [
                f"passage: {value}"
                for value in sample.iloc[start : start + BATCH_SIZE]["text"]
            ]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            output = model(**encoded)
            embeddings.append(
                mean_pool(output.last_hidden_state, encoded["attention_mask"])
                .cpu()
                .numpy()
            )
    elapsed = time.perf_counter() - started
    matrix = np.vstack(embeddings)
    if (
        matrix.shape != (64, EMBEDDING_DIMENSION)
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("E620 benchmark embedding audit failed.")
    projected_rows = 5_226 + 4_096
    projected_seconds = elapsed / len(sample) * projected_rows
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_rows": len(sample),
        "benchmark_batches": math.ceil(len(sample) / BATCH_SIZE),
        "elapsed_seconds": elapsed,
        "seconds_per_row": elapsed / len(sample),
        "projected_rows": projected_rows,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "source_sha256": source_hashes,
        "mcd_cache_sha256": MCD_CACHE_SHA256,
        "outcome_labels_accessed": False,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_external_cache(project_root: str | Path) -> dict[str, object]:
    import torch
    from numpy.lib.format import open_memmap

    source_hashes = verify_sources(project_root)
    paths = _paths(project_root)
    benchmark_result = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    if benchmark_result.get("proceed") is not True:
        raise ValueError("E620 benchmark did not authorize cache construction.")
    frame = pd.read_parquet(paths["mcd_cache"])
    tokenizer, model = _load_model(project_root)
    matrix = open_memmap(
        paths["external_embeddings"],
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), EMBEDDING_DIMENSION),
    )
    started = time.perf_counter()
    for start in range(0, len(frame), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(frame))
        encoded = tokenizer(
            [f"passage: {value}" for value in frame.iloc[start:stop]["text"]],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**encoded)
            pooled = mean_pool(
                output.last_hidden_state, encoded["attention_mask"]
            )
        matrix[start:stop] = pooled.cpu().numpy()
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
        if stop % 512 == 0 or stop == len(frame):
            print(f"e620_cache completed_rows={stop}/{len(frame)}", flush=True)
    del matrix
    loaded = np.load(paths["external_embeddings"])
    if (
        loaded.shape != (len(frame), EMBEDDING_DIMENSION)
        or not np.isfinite(loaded).all()
        or not np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("E620 external cache audit failed.")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "shape": list(loaded.shape),
        "elapsed_seconds": time.perf_counter() - started,
        "cache_sha256": _sha256(paths["external_embeddings"]),
        "source_sha256": source_hashes,
        "mcd_cache_sha256": MCD_CACHE_SHA256,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["external_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _multiclass_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    prediction = probability.argmax(axis=1)
    one_hot = np.eye(3, dtype=np.float64)[target]
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "class_f1": f1_score(target, prediction, average=None).tolist(),
        "macro_ovr_auc": float(
            roc_auc_score(target, probability, multi_class="ovr", average="macro")
        ),
        "quadratic_kappa": float(
            cohen_kappa_score(target, prediction, weights="quadratic")
        ),
        "log_loss": float(log_loss(target, probability, labels=[0, 1, 2])),
        "summed_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    paths = _paths(project_root)
    metadata = json.loads(paths["external_metadata"].read_text(encoding="utf-8"))
    if metadata.get("cache_sha256") != _sha256(paths["external_embeddings"]):
        raise ValueError("E620 external cache binding failed.")
    frame = pd.read_parquet(paths["mcd_cache"])
    embeddings = np.load(paths["external_embeddings"])
    train_mask = frame["split"].eq("train").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    target = frame["label"].to_numpy(dtype=np.int8)
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=400, random_state=20260727
    )
    model.fit(embeddings[train_mask], target[train_mask])
    probability = model.predict_proba(embeddings[test_mask])
    test_target = target[test_mask]
    metrics = _multiclass_metrics(test_target, probability)
    prior = np.bincount(target[train_mask], minlength=3).astype(np.float64)
    prior /= prior.sum()
    prior_probability = np.tile(prior, (test_mask.sum(), 1))
    prior_metrics = _multiclass_metrics(test_target, prior_probability)
    gains = -np.log(np.clip(prior_probability[np.arange(len(test_target)), test_target], 1e-9, 1)) + np.log(
        np.clip(probability[np.arange(len(test_target)), test_target], 1e-9, 1)
    )
    groups = frame.loc[test_mask, "source_group"].astype(str).to_numpy()
    unique = np.unique(groups)
    grouped = np.array([gains[groups == group].mean() for group in unique])
    rng = np.random.default_rng(20260727)
    draws = grouped[rng.integers(0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped)))].mean(axis=1)
    clauses = {
        "accuracy": metrics["accuracy"] >= 0.50,
        "macro_f1": metrics["macro_f1"] >= 0.42,
        "every_class_f1": min(metrics["class_f1"]) >= 0.30,
        "macro_ovr_auc": metrics["macro_ovr_auc"] >= 0.65,
        "quadratic_kappa": metrics["quadratic_kappa"] >= 0.25,
        "log_loss": metrics["log_loss"] <= 1.00,
        "summed_brier": metrics["summed_brier"] <= 0.60,
        "prior_loss_gain": prior_metrics["log_loss"] - metrics["log_loss"] >= 0.10,
        "bootstrap_support": float(np.mean(draws > 0)) >= 0.90,
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "metrics": metrics,
        "prior_metrics": prior_metrics,
        "log_loss_gain_vs_prior": prior_metrics["log_loss"] - metrics["log_loss"],
        "source_group_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "groups": len(unique),
            "support": float(np.mean(draws > 0)),
            "mean_gain": float(draws.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        },
        "clauses": clauses,
        "passes_external_gate": all(clauses.values()),
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report = paths["external_metadata"].with_name(
        "frozen_multilingual_e620_external_report.json"
    )
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_sha256"] = _sha256(report)
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the frozen E620 screen.")
    parser.add_argument("stage", choices=("benchmark", "build-cache", "validate"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build-cache":
        result = build_external_cache(args.project_root)
    else:
        result = validate_external(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
