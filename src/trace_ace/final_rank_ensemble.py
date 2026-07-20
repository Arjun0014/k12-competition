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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.feedback_hash_validation import prepare_feedback_hashes
from trace_ace.foundation import sha256_file
from trace_ace.io import discover_project_paths
from trace_ace.nbsvm_validation import nb_log_count_ratio
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
from trace_ace.semantic_hard_validation import prepare_semantic_features


ARTIFACT_NAME = "final_ensemble_v04_rank.joblib"
SUPERVISED_RUN_ID = "20260717T103832Z_final_supervised_train"


def train_final_rank_ensemble(
    project_root: str | Path,
    supervised_run_id: str = SUPERVISED_RUN_ID,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy(dtype=np.int8)

    base_artifact_path = paths.models_dir / "final_ensemble_v02.joblib"
    if not base_artifact_path.exists():
        raise FileNotFoundError(f"Missing v0.2 base artifact: {base_artifact_path}")
    artifact = dict(joblib.load(base_artifact_path))

    bge_features, bge_similarity = prepare_bge_base_features(paths.cache_dir, frame)
    _, semantic_dense, semantic_dense_names = prepare_semantic_features(
        paths.cache_dir, frame
    )
    semantic_dense = semantic_dense.copy()
    semantic_dense[:, -1:] = bge_similarity
    semantic_scaler = StandardScaler()
    semantic_scaled = semantic_scaler.fit_transform(semantic_dense).astype(np.float32)
    semantic_matrix = np.column_stack(
        [bge_features, semantic_scaled * np.float32(0.08)]
    )
    semantic_model = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=400,
        tol=1e-5,
        random_state=20260716,
    )
    semantic_model.fit(semantic_matrix, target)

    feedback_word, _, _ = prepare_feedback_hashes(
        paths.cache_dir, frame["response_id"]
    )
    ordered = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    if list(ordered["response_id"]) != list(frame["response_id"]):
        raise ValueError("Ordered feedback rows do not align with final training data.")
    ordered_values = ordered[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    nb_ratio = nb_log_count_ratio(feedback_word, target)
    nb_word = feedback_word.multiply(nb_ratio).tocsr()
    nb_scaler = StandardScaler()
    nb_dense = nb_scaler.fit_transform(ordered_values).astype(np.float32)
    nb_matrix = sparse.hstack(
        [nb_word, sparse.csr_matrix(nb_dense * np.float32(0.12))], format="csr"
    )
    nb_model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        dual=True,
        max_iter=400,
        tol=1e-5,
        random_state=20260717,
    )
    nb_model.fit(nb_matrix, target)

    supervised_dir = paths.experiments_dir / "runs" / supervised_run_id
    supervised_delta = supervised_dir / "supervised_delta.safetensors"
    supervised_report_path = supervised_dir / "report.json"
    if not supervised_delta.exists() or not supervised_report_path.exists():
        raise FileNotFoundError("Final supervised checkpoint is incomplete.")
    supervised_report = json.loads(supervised_report_path.read_text(encoding="utf-8"))
    if supervised_report.get("delta_sha256") != sha256_file(supervised_delta):
        raise RuntimeError("Final supervised delta digest mismatch.")

    timestamp = datetime.now(timezone.utc)
    artifact.update(
        {
            "artifact_version": "4.0.0-rank",
            "model_type": "bge_base_nbsvm_supervised_rank_ensemble",
            "created_at_utc": timestamp.isoformat(),
            "build_python_version": platform.python_version(),
            "build_sklearn_version": sklearn.__version__,
            "build_numpy_version": np.__version__,
            "build_joblib_version": joblib.__version__,
            "base_artifact_name": base_artifact_path.name,
            "base_artifact_sha256": sha256_file(base_artifact_path),
            "semantic_model": semantic_model,
            "semantic_scaler": semantic_scaler,
            "semantic_dense_names": semantic_dense_names,
            "semantic_encoder": "BAAI/bge-base-en-v1.5",
            "semantic_max_sequence_length": 256,
            "ensemble_weights": {
                "full_transcript": 0.25,
                "role_objective_dense": 0.25,
                "semantic_interaction_dense": 0.50,
            },
            "nbsvm_model": nb_model,
            "nbsvm_scaler": nb_scaler,
            "nbsvm_log_count_ratio": nb_ratio,
            "nbsvm_word_features": int(feedback_word.shape[1]),
            "nbsvm_dense_weight": 0.12,
            "nbsvm_regularization_c": 1.0,
            "nbsvm_ordered_feature_names": list(ORDERED_FEATURE_NAMES),
            "nbsvm_component_weight": 0.30,
            "supervised_model": "BAAI/bge-small-en-v1.5",
            "supervised_delta_file": "supervised_delta.safetensors",
            "supervised_delta_sha256": supervised_report["delta_sha256"],
            "supervised_max_length": 192,
            "supervised_component_weight": 0.30,
            "supervised_seed": 20260717,
            "final_component_weights": {
                "full_transcript": 0.1225,
                "role_objective_dense": 0.1225,
                "bge_base_semantic": 0.245,
                "nbsvm_feedback": 0.21,
                "supervised_bge_small": 0.30,
            },
            "validation_reference": {
                "combined_candidate_run": "20260717T023245Z_combined_candidate_validation",
                "supervised_fold_0": "20260717T082126Z_supervised_encoder_screen",
                "supervised_fold_3": "20260717T063544Z_supervised_encoder_screen",
            },
        }
    )

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = paths.models_dir / ARTIFACT_NAME
    joblib.dump(artifact, artifact_path, compress=3)
    excluded = {
        "full_model",
        "role_model",
        "role_scaler",
        "semantic_model",
        "semantic_scaler",
        "nbsvm_model",
        "nbsvm_scaler",
        "nbsvm_log_count_ratio",
        "objective_counts",
    }
    metadata = {key: value for key, value in artifact.items() if key not in excluded}
    metadata.update(
        {
            "artifact_path": artifact_path.name,
            "artifact_bytes": artifact_path.stat().st_size,
            "artifact_sha256": sha256_file(artifact_path),
            "semantic_training_shape": list(semantic_matrix.shape),
            "nbsvm_training_shape": list(nb_matrix.shape),
            "training_positive_rate": float(target.mean()),
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
    shutil.copy2(supervised_delta, submission_assets / supervised_delta.name)
    for encoder_name in ("bge-base-en-v1.5", "bge-small-en-v1.5"):
        source = paths.root / "assets" / "pretrained" / encoder_name
        target_dir = submission_assets / encoder_name
        if not target_dir.exists():
            shutil.copytree(source, target_dir)
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the locked final rank ensemble.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--supervised-run-id", default=SUPERVISED_RUN_ID)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = train_final_rank_ensemble(
        args.project_root, supervised_run_id=args.supervised_run_id
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
