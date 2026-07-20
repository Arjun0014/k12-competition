from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.semantic_cache import compact_objective_context


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIRECTORY = "bge-small-en-v1.5"
SOURCE_MAX_LENGTH = 256
TARGET_MAX_LENGTH = 384
EMBEDDING_DIMENSION = 384


def build_long_context_cache(
    project_root: str | Path,
    batch_size: int = 64,
    chunk_size: int = 256,
) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    paths = discover_project_paths(project_root)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet", columns=["response_id"]
    )
    if list(contexts["response_id"]) != list(frame["response_id"]):
        raise ValueError("Long-context rows do not match modeling data.")
    compact_texts = [
        compact_objective_context(text) for text in contexts["objective_context"]
    ]

    model_dir = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    token_lengths: list[int] = []
    for start in range(0, len(compact_texts), 1024):
        tokenized = tokenizer(
            compact_texts[start : start + 1024],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        token_lengths.extend(len(ids) for ids in tokenized)
    token_lengths_array = np.asarray(token_lengths, dtype=np.int32)
    long_rows = np.flatnonzero(token_lengths_array > SOURCE_MAX_LENGTH)
    if (token_lengths_array > TARGET_MAX_LENGTH).any():
        raise ValueError("At least one compact context exceeds the target token budget.")

    source_path = paths.cache_dir / "bge_small_context_256.npy"
    target_path = paths.cache_dir / "bge_small_context_384.npy"
    progress_path = paths.cache_dir / "bge_small_context_384.progress.json"
    metadata_path = paths.cache_dir / "bge_small_context_384.metadata.json"
    expected_shape = (len(frame), EMBEDDING_DIMENSION)
    source = np.load(source_path, mmap_mode="r")
    if source.shape != expected_shape or not np.isfinite(source).all():
        raise ValueError("Source BGE-small cache failed audit.")
    if not target_path.exists():
        shutil.copy2(source_path, target_path)
    target = np.lib.format.open_memmap(target_path, mode="r+")
    if target.shape != expected_shape:
        raise ValueError("Existing long-context cache has the wrong shape.")

    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_long_rows", 0))
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = TARGET_MAX_LENGTH
    for chunk_start in range(completed, len(long_rows), chunk_size):
        chunk_end = min(len(long_rows), chunk_start + chunk_size)
        row_indices = long_rows[chunk_start:chunk_end]
        embeddings = model.encode(
            [compact_texts[index] for index in row_indices],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        target[row_indices] = embeddings
        target.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "completed_long_rows": chunk_end,
                    "total_long_rows": len(long_rows),
                    "response_rows": len(frame),
                    "model": MODEL_NAME,
                    "source_max_length": SOURCE_MAX_LENGTH,
                    "target_max_length": TARGET_MAX_LENGTH,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Re-encoded long contexts {chunk_end}/{len(long_rows)}.", flush=True)
    del target
    completed_values = np.load(target_path, mmap_mode="r")
    if completed_values.shape != expected_shape or not np.isfinite(completed_values).all():
        raise ValueError("Completed long-context cache failed audit.")
    unchanged_rows = np.flatnonzero(token_lengths_array <= SOURCE_MAX_LENGTH)
    if not np.array_equal(completed_values[unchanged_rows], source[unchanged_rows]):
        raise ValueError("Short rows changed in the surgical long-context cache.")
    metadata_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "license": "MIT",
                "response_rows": len(frame),
                "reencoded_rows": len(long_rows),
                "source_max_length": SOURCE_MAX_LENGTH,
                "target_max_length": TARGET_MAX_LENGTH,
                "token_length_mean": float(token_lengths_array.mean()),
                "token_length_p95": float(np.percentile(token_lengths_array, 95)),
                "token_length_max": int(token_lengths_array.max()),
                "output_path": target_path.name,
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
        "reencoded_rows": len(long_rows),
        "output_path": str(target_path),
        "metadata_path": str(metadata_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Surgically extend BGE-small compact contexts to 384 tokens."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_long_context_cache(
        args.project_root, batch_size=args.batch_size, chunk_size=args.chunk_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
