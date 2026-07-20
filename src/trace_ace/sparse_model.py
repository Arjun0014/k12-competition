from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from trace_ace.baselines import _logistic
from trace_ace.config import load_config
from trace_ace.foundation import sha256_file
from trace_ace.io import discover_project_paths


def train_sparse_model(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    text_config = config["text_baseline"]
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    session_texts = pd.read_parquet(paths.cache_dir / "session_texts.parquet").set_index(
        "session_id"
    )
    session_ids = pd.Index(frame["session_id"].drop_duplicates())
    missing = session_ids.difference(session_texts.index)
    if len(missing):
        raise ValueError(f"Missing cached text for {len(missing)} training sessions.")

    transcript_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=int(text_config["word_min_df"]),
        max_features=int(text_config["word_max_features"]),
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    unique_transcript_matrix = transcript_vectorizer.fit_transform(
        session_texts.loc[session_ids, "full_text"]
    )
    session_lookup = pd.Series(np.arange(len(session_ids)), index=session_ids)
    response_session_rows = frame["session_id"].map(session_lookup).to_numpy(dtype=np.int64)
    response_transcript_matrix = unique_transcript_matrix[response_session_rows]

    objective_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=min(20_000, int(text_config["word_max_features"])),
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    objective_matrix = objective_vectorizer.fit_transform(frame["learning_objective"])
    training_matrix = hstack(
        [response_transcript_matrix, objective_matrix], format="csr", dtype=np.float32
    )
    model = _logistic(
        c=float(text_config["logistic_c"]), max_iter=int(text_config["max_iter"])
    )
    model.fit(training_matrix, frame["target"].to_numpy())

    with (paths.cache_dir / "foundation_report.json").open("r", encoding="utf-8") as handle:
        foundation = json.load(handle)
    artifact = {
        "artifact_version": "1.0.0",
        "model_type": "full_session_word_tfidf_logistic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_python_version": platform.python_version(),
        "build_sklearn_version": sklearn.__version__,
        "build_numpy_version": np.__version__,
        "build_joblib_version": joblib.__version__,
        "source_manifest_sha256": foundation["source_manifest_sha256"],
        "probability_clip": float(config["probability_clip"]),
        "transcript_vectorizer": transcript_vectorizer,
        "objective_vectorizer": objective_vectorizer,
        "model": model,
        "training_rows": int(len(frame)),
        "training_sessions": int(frame["session_id"].nunique()),
        "transcript_features": int(unique_transcript_matrix.shape[1]),
        "objective_features": int(objective_matrix.shape[1]),
        "cv_reference_log_loss": 0.5321893419768516,
    }

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = paths.models_dir / "word_full_tfidf.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_hash = sha256_file(artifact_path)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"transcript_vectorizer", "objective_vectorizer", "model"}
    }
    metadata.update(
        {
            "artifact_path": artifact_path.name,
            "artifact_bytes": artifact_path.stat().st_size,
            "artifact_sha256": artifact_hash,
            "training_matrix_shape": [int(value) for value in training_matrix.shape],
        }
    )
    metadata_path = paths.models_dir / "word_full_tfidf.metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    submission_assets = paths.root / "submission_src" / "assets"
    submission_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, submission_assets / artifact_path.name)
    shutil.copy2(metadata_path, submission_assets / metadata_path.name)
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the promoted full-data sparse text model.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    metadata = train_sparse_model(args.project_root)
    print(
        f"Trained {metadata['model_type']}: {metadata['training_matrix_shape']} matrix, "
        f"artifact={metadata['artifact_bytes']} bytes, sha256={metadata['artifact_sha256']}."
    )


if __name__ == "__main__":
    main()
