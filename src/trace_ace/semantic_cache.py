from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256


def compact_objective_context(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    objective = lines[0].replace("[OBJECTIVE]", "Objective:", 1)
    discussion = lines[1:]
    if len(discussion) <= 12:
        selected = discussion
    else:
        selected = [*discussion[-8:], *discussion[:4]]

    compact_lines: list[str] = []
    for line in selected:
        words = line.split()
        compact_lines.append(" ".join(words[:16]))
    ending_count = min(8, len(discussion))
    ending = compact_lines[:ending_count]
    opening = compact_lines[ending_count:]
    parts = [objective]
    if ending:
        parts.append("End of objective discussion: " + " ".join(ending))
    if opening:
        parts.append("Earlier objective discussion: " + " ".join(opening))
    return "\n".join(parts)


def build_semantic_cache(
    project_root: str | Path,
    batch_size: int = 64,
    chunk_size: int = 512,
) -> dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required to build semantic caches.") from error

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
        raise ValueError("Semantic-cache response order does not match modeling data.")

    model_dir = paths.root / "assets" / "pretrained" / "bge-small-en-v1.5"
    if not model_dir.exists():
        raise FileNotFoundError(f"Packaged BGE model is missing: {model_dir}")
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = MAX_SEQUENCE_LENGTH

    context_path = paths.cache_dir / "bge_small_context_256.npy"
    progress_path = paths.cache_dir / "bge_small_context_256.progress.json"
    expected_shape = (len(frame), EMBEDDING_DIMENSION)
    if context_path.exists():
        context_embeddings = np.lib.format.open_memmap(context_path, mode="r+")
        if context_embeddings.shape != expected_shape:
            raise ValueError("Existing BGE context cache has the wrong shape.")
    else:
        context_embeddings = np.lib.format.open_memmap(
            context_path, mode="w+", dtype=np.float32, shape=expected_shape
        )

    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start = int(progress.get("completed_rows", 0))
    compact_contexts = [compact_objective_context(text) for text in contexts["objective_context"]]
    for chunk_start in range(start, len(frame), chunk_size):
        chunk_end = min(len(frame), chunk_start + chunk_size)
        embeddings = model.encode(
            compact_contexts[chunk_start:chunk_end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        context_embeddings[chunk_start:chunk_end] = embeddings
        context_embeddings.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "completed_rows": chunk_end,
                    "total_rows": len(frame),
                    "model": MODEL_NAME,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Encoded semantic contexts for {chunk_end}/{len(frame)} responses.", flush=True)

    objective_path = paths.cache_dir / "bge_small_objective_256.npy"
    if not objective_path.exists():
        unique_objectives = frame[["learning_objective"]].drop_duplicates().reset_index(drop=True)
        unique_embeddings = model.encode(
            unique_objectives["learning_objective"].fillna("").tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        lookup = {
            text: embedding
            for text, embedding in zip(unique_objectives["learning_objective"], unique_embeddings)
        }
        objective_embeddings = np.vstack(
            [lookup[text] for text in frame["learning_objective"]]
        ).astype(np.float32)
        np.save(objective_path, objective_embeddings)
    progress_path.unlink(missing_ok=True)
    return {
        "responses": len(frame),
        "context_path": str(context_path),
        "objective_path": str(objective_path),
        "model_dir": str(model_dir),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build BGE semantic embedding caches.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_semantic_cache(args.project_root, args.batch_size, args.chunk_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
