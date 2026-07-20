from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.semantic_cache import compact_objective_context


MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_SLUG = "bge_base"
MODEL_DIRECTORY = "bge-base-en-v1.5"
MAX_SEQUENCE_LENGTH = 256


def build_encoder_upgrade_cache(
    project_root: str | Path,
    batch_size: int = 32,
    chunk_size: int = 256,
) -> dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required to build embeddings.") from error

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
        raise ValueError("Encoder-cache response order does not match modeling data.")

    model_dir = paths.root / "assets" / "pretrained" / MODEL_DIRECTORY
    if not model_dir.exists():
        raise FileNotFoundError(f"Packaged encoder is missing: {model_dir}")
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    dimension = int(model.get_embedding_dimension())
    if dimension <= 0:
        raise ValueError("Encoder reported an invalid embedding dimension.")

    context_path = paths.cache_dir / f"{MODEL_SLUG}_context_256.npy"
    objective_path = paths.cache_dir / f"{MODEL_SLUG}_objective_256.npy"
    progress_path = paths.cache_dir / f"{MODEL_SLUG}_context_256.progress.json"
    metadata_path = paths.cache_dir / f"{MODEL_SLUG}_semantic_256.metadata.json"
    expected_shape = (len(frame), dimension)
    if context_path.exists():
        context_embeddings = np.lib.format.open_memmap(context_path, mode="r+")
        if context_embeddings.shape != expected_shape:
            raise ValueError("Existing upgraded context cache has the wrong shape.")
    else:
        context_embeddings = np.lib.format.open_memmap(
            context_path, mode="w+", dtype=np.float32, shape=expected_shape
        )

    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("model") != MODEL_NAME:
            raise ValueError("Encoder progress file belongs to a different model.")
        start = int(progress.get("completed_rows", 0))
    compact_contexts = [
        compact_objective_context(text) for text in contexts["objective_context"]
    ]
    for chunk_start in range(start, len(frame), chunk_size):
        chunk_end = min(len(frame), chunk_start + chunk_size)
        embeddings = model.encode(
            compact_contexts[chunk_start:chunk_end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        if embeddings.shape != (chunk_end - chunk_start, dimension):
            raise ValueError("Encoder returned an unexpected context shape.")
        context_embeddings[chunk_start:chunk_end] = embeddings
        context_embeddings.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "completed_rows": chunk_end,
                    "total_rows": len(frame),
                    "model": MODEL_NAME,
                    "embedding_dimension": dimension,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Encoded {MODEL_SLUG} contexts for {chunk_end}/{len(frame)} responses.",
            flush=True,
        )

    if not objective_path.exists():
        unique_objectives = frame[["learning_objective"]].drop_duplicates().reset_index(
            drop=True
        )
        unique_embeddings = model.encode(
            unique_objectives["learning_objective"].fillna("").tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        lookup = {
            text: embedding
            for text, embedding in zip(
                unique_objectives["learning_objective"], unique_embeddings
            )
        }
        objective_embeddings = np.vstack(
            [lookup[text] for text in frame["learning_objective"]]
        ).astype(np.float32)
        np.save(objective_path, objective_embeddings)

    del context_embeddings
    context_values = np.load(context_path, mmap_mode="r")
    objective_values = np.load(objective_path, mmap_mode="r")
    for name, values in (("context", context_values), ("objective", objective_values)):
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Completed {name} embedding cache failed audit.")
    metadata_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "model_directory": MODEL_DIRECTORY,
                "embedding_dimension": dimension,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "response_rows": len(frame),
                "context_path": context_path.name,
                "objective_path": objective_path.name,
                "license": "MIT",
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
        "embedding_dimension": dimension,
        "context_path": str(context_path),
        "objective_path": str(objective_path),
        "metadata_path": str(metadata_path),
        "model_dir": str(model_dir),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build BGE-base semantic caches.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_encoder_upgrade_cache(
        args.project_root, args.batch_size, args.chunk_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
