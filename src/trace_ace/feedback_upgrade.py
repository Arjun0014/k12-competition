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
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from trace_ace.foundation import sha256_file
from trace_ace.io import discover_project_paths
from trace_ace.multiview_cache import VIEW_SCHEMA_VERSION, build_multiview_texts
from trace_ace.ordered_features import (
    ORDERED_FEATURE_NAMES,
    ORDERED_FEATURE_SCHEMA,
    build_ordered_feature_cache,
)


FEEDBACK_WORD_FEATURES = 2**17
FEEDBACK_DENSE_WEIGHT = 0.12
FEEDBACK_ALPHA = 1e-4
FEEDBACK_BLEND_WEIGHT = 0.20
FEEDBACK_RANDOM_STATE = 20260716


def feedback_word_hasher() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=FEEDBACK_WORD_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def fit_feedback_model(
    word_matrix: sparse.csr_matrix,
    ordered_values: np.ndarray,
    target: np.ndarray,
) -> tuple[SGDClassifier, StandardScaler, tuple[int, int]]:
    if word_matrix.shape[0] != len(target) or ordered_values.shape[0] != len(target):
        raise ValueError("Feedback features and targets must contain the same rows.")
    if ordered_values.shape[1] != len(ORDERED_FEATURE_NAMES):
        raise ValueError("Ordered feature width does not match the registered schema.")
    if not np.isfinite(ordered_values).all():
        raise ValueError("Ordered feedback features contain non-finite values.")

    scaler = StandardScaler()
    dense_scaled = scaler.fit_transform(ordered_values).astype(np.float32)
    matrix = sparse.hstack(
        [
            word_matrix,
            sparse.csr_matrix(dense_scaled * np.float32(FEEDBACK_DENSE_WEIGHT)),
        ],
        format="csr",
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=FEEDBACK_ALPHA,
        max_iter=100,
        tol=1e-4,
        shuffle=True,
        random_state=FEEDBACK_RANDOM_STATE,
        average=True,
    )
    model.fit(matrix, target)
    return model, scaler, (int(matrix.shape[0]), int(matrix.shape[1]))


def train_feedback_upgrade(
    project_root: str | Path,
    base_artifact_name: str = "final_ensemble_v02.joblib",
    output_artifact_name: str = "final_ensemble_v03_feedback.joblib",
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    base_artifact_path = paths.models_dir / base_artifact_name
    if not base_artifact_path.exists():
        raise FileNotFoundError(f"Missing base ensemble artifact: {base_artifact_path}")

    build_multiview_texts(paths.root)
    build_ordered_feature_cache(paths.root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy(dtype=np.int8)

    feedback = pd.read_parquet(
        paths.cache_dir / "response_multiview_texts.parquet",
        columns=["response_id", "feedback_evidence", "view_schema_version"],
    )
    ordered = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    expected_ids = list(frame["response_id"])
    if list(feedback["response_id"]) != expected_ids:
        raise ValueError("Feedback evidence is not aligned with modeling rows.")
    if list(ordered["response_id"]) != expected_ids:
        raise ValueError("Ordered features are not aligned with modeling rows.")
    if set(feedback["view_schema_version"]) != {VIEW_SCHEMA_VERSION}:
        raise ValueError("Feedback evidence uses an unexpected view schema.")
    if set(ordered["ordered_feature_schema"]) != {ORDERED_FEATURE_SCHEMA}:
        raise ValueError("Ordered features use an unexpected feature schema.")

    word_matrix = feedback_word_hasher().transform(
        feedback["feedback_evidence"].fillna("").astype(str)
    ).tocsr()
    ordered_values = ordered[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    feedback_model, feedback_scaler, training_shape = fit_feedback_model(
        word_matrix, ordered_values, target
    )

    artifact = dict(joblib.load(base_artifact_path))
    base_artifact_version = str(artifact.get("artifact_version", "unknown"))
    base_model_type = str(artifact.get("model_type", "unknown"))
    artifact.update(
        {
            "artifact_version": "3.0.0-feedback",
            "model_type": "objective_shift_robust_ensemble_with_ordered_feedback",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "build_python_version": platform.python_version(),
            "build_sklearn_version": sklearn.__version__,
            "build_numpy_version": np.__version__,
            "build_joblib_version": joblib.__version__,
            "base_artifact_name": base_artifact_name,
            "base_artifact_sha256": sha256_file(base_artifact_path),
            "base_artifact_version": base_artifact_version,
            "base_model_type": base_model_type,
            "feedback_model": feedback_model,
            "feedback_scaler": feedback_scaler,
            "feedback_word_features": FEEDBACK_WORD_FEATURES,
            "feedback_ngram_range": [1, 2],
            "feedback_dense_weight": FEEDBACK_DENSE_WEIGHT,
            "feedback_alpha": FEEDBACK_ALPHA,
            "feedback_random_state": FEEDBACK_RANDOM_STATE,
            "feedback_blend_weight": FEEDBACK_BLEND_WEIGHT,
            "feedback_view_schema": VIEW_SCHEMA_VERSION,
            "feedback_ordered_feature_schema": ORDERED_FEATURE_SCHEMA,
            "feedback_ordered_feature_names": list(ORDERED_FEATURE_NAMES),
        }
    )

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = paths.models_dir / output_artifact_name
    joblib.dump(artifact, artifact_path, compress=3)
    excluded = {
        "full_model",
        "role_model",
        "role_scaler",
        "semantic_model",
        "semantic_scaler",
        "feedback_model",
        "feedback_scaler",
        "objective_counts",
    }
    metadata = {key: value for key, value in artifact.items() if key not in excluded}
    metadata.update(
        {
            "artifact_path": artifact_path.name,
            "artifact_bytes": artifact_path.stat().st_size,
            "artifact_sha256": sha256_file(artifact_path),
            "feedback_training_shape": list(training_shape),
            "feedback_training_positive_rate": float(target.mean()),
        }
    )
    metadata_path = artifact_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    submission_assets = paths.root / "submission_src" / "assets"
    submission_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, submission_assets / artifact_path.name)
    shutil.copy2(metadata_path, submission_assets / metadata_path.name)
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the promoted answer-feedback upgrade over a frozen ensemble."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--base-artifact", default="final_ensemble_v02.joblib")
    parser.add_argument("--output-artifact", default="final_ensemble_v03_feedback.joblib")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = train_feedback_upgrade(
        args.project_root,
        base_artifact_name=args.base_artifact,
        output_artifact_name=args.output_artifact,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
