from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.semantic_cache import compact_objective_context


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MODEL_DIRECTORY = "nli-deberta-v3-small"
MAX_SEQUENCE_LENGTH = 256
POOLED_DIMENSION = 768
LOGIT_DIMENSION = 3


def mastery_hypothesis(objective: str) -> str:
    cleaned = " ".join(str(objective).split()).strip().rstrip(".")
    return f"The student demonstrates mastery of this learning objective: {cleaned}."


def build_nli_cross_encoder_cache(
    project_root: str | Path,
    batch_size: int = 16,
    chunk_size: int = 256,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    paths = discover_project_paths(project_root)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet",
        columns=["response_id", "learning_objective"],
    )
    if list(contexts["response_id"]) != list(frame["response_id"]):
        raise ValueError("NLI cache rows do not match modeling data.")
    premises = [
        compact_objective_context(text) for text in contexts["objective_context"]
    ]
    hypotheses = [mastery_hypothesis(value) for value in frame["learning_objective"]]

    model_dir = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model.eval()
    if int(model.config.hidden_size) != POOLED_DIMENSION:
        raise ValueError("Unexpected NLI pooled dimension.")
    expected_labels = {0: "contradiction", 1: "entailment", 2: "neutral"}
    actual_labels = {
        int(key): str(value).lower() for key, value in model.config.id2label.items()
    }
    if actual_labels != expected_labels:
        raise ValueError(f"Unexpected NLI label mapping: {actual_labels}")

    pooled_path = paths.cache_dir / "nli_v3_small_mastery_pooled_256.npy"
    logits_path = paths.cache_dir / "nli_v3_small_mastery_logits_256.npy"
    progress_path = paths.cache_dir / "nli_v3_small_mastery_256.progress.json"
    metadata_path = paths.cache_dir / "nli_v3_small_mastery_256.metadata.json"
    pooled_shape = (len(frame), POOLED_DIMENSION)
    logits_shape = (len(frame), LOGIT_DIMENSION)
    if pooled_path.exists():
        pooled_values = np.lib.format.open_memmap(pooled_path, mode="r+")
        logits_values = np.lib.format.open_memmap(logits_path, mode="r+")
        if pooled_values.shape != pooled_shape or logits_values.shape != logits_shape:
            raise ValueError("Existing NLI cache has the wrong shape.")
    else:
        pooled_values = np.lib.format.open_memmap(
            pooled_path, mode="w+", dtype=np.float32, shape=pooled_shape
        )
        logits_values = np.lib.format.open_memmap(
            logits_path, mode="w+", dtype=np.float32, shape=logits_shape
        )

    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("model") != MODEL_NAME:
            raise ValueError("NLI progress file belongs to another model.")
        completed = int(progress.get("completed_rows", 0))
    for chunk_start in range(completed, len(frame), chunk_size):
        chunk_end = min(len(frame), chunk_start + chunk_size)
        chunk_pooled: list[np.ndarray] = []
        chunk_logits: list[np.ndarray] = []
        for batch_start in range(chunk_start, chunk_end, batch_size):
            batch_end = min(chunk_end, batch_start + batch_size)
            tokenized = tokenizer(
                premises[batch_start:batch_end],
                hypotheses[batch_start:batch_end],
                padding=True,
                truncation="only_first",
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
            )
            with torch.inference_mode():
                outputs = model(
                    **tokenized,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pooled = model.pooler(outputs.hidden_states[-1])
            chunk_pooled.append(pooled.cpu().numpy().astype(np.float32))
            chunk_logits.append(outputs.logits.cpu().numpy().astype(np.float32))
        pooled_values[chunk_start:chunk_end] = np.vstack(chunk_pooled)
        logits_values[chunk_start:chunk_end] = np.vstack(chunk_logits)
        pooled_values.flush()
        logits_values.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "completed_rows": chunk_end,
                    "total_rows": len(frame),
                    "model": MODEL_NAME,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                    "hypothesis_template": (
                        "The student demonstrates mastery of this learning objective: "
                        "<objective>."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Encoded NLI mastery pairs for {chunk_end}/{len(frame)} rows.", flush=True)

    del pooled_values, logits_values
    completed_pooled = np.load(pooled_path, mmap_mode="r")
    completed_logits = np.load(logits_path, mmap_mode="r")
    if not np.isfinite(completed_pooled).all() or not np.isfinite(completed_logits).all():
        raise ValueError("Completed NLI cache contains non-finite values.")
    metadata_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "model_directory": MODEL_DIRECTORY,
                "license": "Apache-2.0",
                "training_datasets": ["SNLI", "MultiNLI"],
                "response_rows": len(frame),
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "pooled_dimension": POOLED_DIMENSION,
                "logit_labels": expected_labels,
                "pooled_path": pooled_path.name,
                "logits_path": logits_path.name,
                "hypothesis_template": (
                    "The student demonstrates mastery of this learning objective: "
                    "<objective>."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress_path.unlink(missing_ok=True)
    return {
        "responses": len(frame),
        "pooled_path": str(pooled_path),
        "logits_path": str(logits_path),
        "metadata_path": str(metadata_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen DeBERTa NLI context-objective features."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_nli_cross_encoder_cache(
        args.project_root, batch_size=args.batch_size, chunk_size=args.chunk_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
