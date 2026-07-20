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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from trace_ace.foundation import sha256_file
from trace_ace.hard_validation import _response_transcript_matrix, prepare_hash_matrices
from trace_ace.io import discover_project_paths
from trace_ace.role_hard_validation import (
    ROLE_FEATURES,
    _objective_hash,
    _response_rows,
    _row_cosine,
    _unit_weighted_hstack,
    prepare_behavior_features,
    prepare_role_hashes,
)
from trace_ace.semantic_hard_validation import prepare_semantic_features


FINAL_WEIGHTS = {
    "full_transcript": 0.25,
    "role_objective_dense": 0.25,
    "semantic_interaction_dense": 0.50,
}


def _sgd(alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=100,
        tol=1e-4,
        shuffle=True,
        random_state=20260716,
        average=True,
    )


def train_final_ensemble(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy()

    full_session, _ = prepare_hash_matrices(frame, paths.cache_dir)
    full_response = _response_transcript_matrix(frame, paths.cache_dir, full_session)
    full_model = _sgd(alpha=3e-5)
    full_model.fit(full_response, target)

    session_order, session_roles = prepare_role_hashes(paths.cache_dir)
    response_rows = _response_rows(frame, session_order)
    student = session_roles["student"][response_rows]
    tutor = session_roles["tutor"][response_rows]
    opening = session_roles["opening"][response_rows]
    closing_student = session_roles["closing_student"][response_rows]
    closing_tutor = session_roles["closing_tutor"][response_rows]
    objective_64k = _objective_hash(frame, ROLE_FEATURES)
    objective_32k = _objective_hash(frame, 2**15)
    role_sparse = _unit_weighted_hstack(
        [(student, 0.75), (tutor, 0.55), (objective_64k, 1.0)]
    )
    behavior, behavior_names = prepare_behavior_features(
        frame, paths.cache_dir, session_order, response_rows
    )
    role_alignment = np.column_stack(
        [
            _row_cosine(student, objective_64k),
            _row_cosine(tutor, objective_64k),
            _row_cosine(opening, objective_64k),
            _row_cosine(closing_student, objective_32k),
            _row_cosine(closing_tutor, objective_32k),
        ]
    )
    role_dense = np.column_stack([behavior, role_alignment])
    role_scaler = StandardScaler()
    role_dense_scaled = role_scaler.fit_transform(role_dense).astype(np.float32)
    role_matrix = sparse.hstack(
        [role_sparse, sparse.csr_matrix(role_dense_scaled * np.float32(0.2))],
        format="csr",
    )
    role_model = _sgd(alpha=3e-5)
    role_model.fit(role_matrix, target)

    semantic_variants, semantic_dense, semantic_dense_names = prepare_semantic_features(
        paths.cache_dir, frame
    )
    semantic_features = semantic_variants["semantic_interaction"]
    semantic_scaler = StandardScaler()
    semantic_dense_scaled = semantic_scaler.fit_transform(semantic_dense).astype(np.float32)
    semantic_matrix = np.column_stack(
        [semantic_features, semantic_dense_scaled * np.float32(0.08)]
    )
    semantic_model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=400,
        tol=1e-5,
        random_state=20260716,
    )
    semantic_model.fit(semantic_matrix, target)

    artifact = {
        "artifact_version": "2.0.0",
        "model_type": "objective_shift_robust_sparse_semantic_ensemble",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_python_version": platform.python_version(),
        "build_sklearn_version": sklearn.__version__,
        "build_numpy_version": np.__version__,
        "build_joblib_version": joblib.__version__,
        "training_rows": int(len(frame)),
        "training_sessions": int(frame["session_id"].nunique()),
        "global_prior": float(target.mean()),
        "probability_clip": 1e-6,
        "ensemble_weights": FINAL_WEIGHTS,
        "full_hash_features": 2**17,
        "role_hash_features": ROLE_FEATURES,
        "closing_hash_features": 2**15,
        "full_model": full_model,
        "role_model": role_model,
        "role_scaler": role_scaler,
        "role_behavior_names": behavior_names,
        "role_alignment_names": [
            "objective_student_cosine",
            "objective_tutor_cosine",
            "objective_opening_cosine",
            "objective_closing_student_cosine",
            "objective_closing_tutor_cosine",
        ],
        "semantic_model": semantic_model,
        "semantic_scaler": semantic_scaler,
        "semantic_dense_names": semantic_dense_names,
        "semantic_encoder": "BAAI/bge-small-en-v1.5",
        "semantic_max_sequence_length": 256,
        "objective_counts": frame["learning_objective_id"].value_counts().to_dict(),
        "hard_cv_reference": {
            "protocol": "learning_objective-disjoint with validation-session purge",
            "log_loss": 0.5845817,
            "roc_auc": 0.6344370,
        },
    }

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = paths.models_dir / "final_ensemble_v02.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "full_model",
            "role_model",
            "role_scaler",
            "semantic_model",
            "semantic_scaler",
            "objective_counts",
        }
    }
    metadata.update(
        {
            "artifact_path": artifact_path.name,
            "artifact_bytes": artifact_path.stat().st_size,
            "artifact_sha256": sha256_file(artifact_path),
            "full_training_shape": [int(value) for value in full_response.shape],
            "role_training_shape": [int(value) for value in role_matrix.shape],
            "semantic_training_shape": [int(value) for value in semantic_matrix.shape],
        }
    )
    metadata_path = paths.models_dir / "final_ensemble_v02.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    submission_assets = paths.root / "submission_src" / "assets"
    submission_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, submission_assets / artifact_path.name)
    shutil.copy2(metadata_path, submission_assets / metadata_path.name)
    encoder_source = paths.root / "assets" / "pretrained" / "bge-small-en-v1.5"
    encoder_target = submission_assets / "bge-small-en-v1.5"
    if encoder_target.exists():
        shutil.rmtree(encoder_target)
    shutil.copytree(encoder_source, encoder_target)
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the promoted v0.2 ensemble.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    metadata = train_final_ensemble(args.project_root)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
