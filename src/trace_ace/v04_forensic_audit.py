from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import types
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.feedback_hash_validation import prepare_feedback_hashes
from trace_ace.foundation import sha256_file
from trace_ace.hard_validation import _response_transcript_matrix, prepare_hash_matrices
from trace_ace.io import discover_project_paths
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
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
from trace_ace.supervised_encoder_screen import (
    MAX_LENGTH as TRAINING_SUPERVISED_MAX_LENGTH,
    _tokenize_pairs as training_tokenize_pairs,
    mastery_hypothesis as training_mastery_hypothesis,
)


PACKAGE_RELATIVE_PATH = Path("submission_builds/final_ensemble_v04_rank.zip")
SMOKE_DATA_RELATIVE_PATH = Path("tmp/final_ensemble_v04_rank_submission/data")
EXPECTED_PACKAGE_SHA256 = "aeff8f0ea73a02ba5c3f41f69a91e1d835dafde33e185d462ff86badcecb8387"
EXPECTED_WIDTHS = {
    "full_transcript": 131_072,
    "role_objective_dense": 196_637,
    "bge_base_semantic": 3_107,
    "nbsvm_feedback": 131_123,
    "supervised_bge_small": 192,
    "final_v04_output": 1,
}
GATE_A_TOLERANCE = 1e-6
PREPROCESSING_TOLERANCE = 2e-6
PILOT_PROTOCOL = "semantic_k50_s0"
FINAL_SUPERVISED_TRAIN_RUN_ID = "20260717T103832Z_final_supervised_train"
GATE_A_COMPONENTS = {
    "full": "full_transcript",
    "role": "role_objective_dense",
    "bge_base": "bge_base_semantic",
    "nbsvm": "nbsvm_feedback",
    "supervised": "supervised_bge_small",
    "final": "final_v04_output",
}
GATE_A_SAMPLE_ROWS = 100


@dataclass(frozen=True)
class AuditArtifacts:
    package: Path
    smoke_features: Path
    smoke_transcripts: Path
    model: Path
    delta: Path
    package_main: Path


class _CaptureEstimator:
    """Delegates inference while retaining the exact matrix passed by packaged code."""

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator
        self.matrix: sparse.csr_matrix | np.ndarray | None = None
        self.probability: np.ndarray | None = None

    def predict_proba(self, matrix):  # noqa: ANN001 - sklearn accepts sparse or dense
        self.matrix = matrix.copy()
        probability = self.estimator.predict_proba(matrix)
        self.probability = np.asarray(probability[:, 1], dtype=np.float64)
        return probability


class _CaptureTokenizer:
    """Delegates tokenization while capturing the exact packaged call contract."""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        self.contexts: list[str] = []
        self.hypotheses: list[str] = []
        self.keyword_arguments: list[dict[str, object]] = []
        self.input_id_batches: list[np.ndarray] = []

    def __call__(self, contexts, hypotheses, **kwargs):  # noqa: ANN001
        context_values = [str(value) for value in contexts]
        hypothesis_values = [str(value) for value in hypotheses]
        encoded = self.tokenizer(context_values, hypothesis_values, **kwargs)
        self.contexts.extend(context_values)
        self.hypotheses.extend(hypothesis_values)
        self.keyword_arguments.append(dict(kwargs))
        self.input_id_batches.append(
            encoded["input_ids"].detach().cpu().numpy().astype(np.int64, copy=True)
        )
        return encoded

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_v04_forensic_audit")


def _hash_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _hash_texts(values: Sequence[str]) -> str:
    def chunks() -> Iterable[bytes]:
        for value in values:
            encoded = str(value).encode("utf-8")
            yield len(encoded).to_bytes(8, "little")
            yield encoded

    return _hash_bytes(chunks())


def _hash_dense(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return _hash_bytes(
        [
            json.dumps(list(array.shape)).encode("ascii"),
            str(array.dtype).encode("ascii"),
            array.tobytes(order="C"),
        ]
    )


def _hash_sparse(values: sparse.spmatrix) -> str:
    matrix = values.tocsr(copy=True)
    matrix.sort_indices()
    return _hash_bytes(
        [
            json.dumps(list(matrix.shape)).encode("ascii"),
            str(matrix.dtype).encode("ascii"),
            np.ascontiguousarray(matrix.indptr).tobytes(),
            np.ascontiguousarray(matrix.indices).tobytes(),
            np.ascontiguousarray(matrix.data).tobytes(),
        ]
    )


def _matrix_hash(values: sparse.spmatrix | np.ndarray) -> str:
    return _hash_sparse(values) if sparse.issparse(values) else _hash_dense(np.asarray(values))


def _matrix_max_abs_difference(
    left: sparse.spmatrix | np.ndarray,
    right: sparse.spmatrix | np.ndarray,
) -> tuple[float, int]:
    if left.shape != right.shape:
        return float("inf"), -1
    if sparse.issparse(left) or sparse.issparse(right):
        difference = sparse.csr_matrix(left) - sparse.csr_matrix(right)
        if difference.nnz == 0:
            return 0.0, 0
        absolute = np.abs(difference.data)
        return float(absolute.max(initial=0.0)), int(np.count_nonzero(absolute))
    difference = np.abs(np.asarray(left) - np.asarray(right))
    return float(difference.max(initial=0.0)), int(np.count_nonzero(difference))


def _text_check(name: str, offline: Sequence[str], runtime: Sequence[str]) -> dict[str, object]:
    mismatch_count = sum(left != right for left, right in zip(offline, runtime))
    mismatch_count += abs(len(offline) - len(runtime))
    return {
        "check": name,
        "comparison_type": "text",
        "offline_shape": f"[{len(offline)}]",
        "runtime_shape": f"[{len(runtime)}]",
        "offline_sha256": _hash_texts(offline),
        "runtime_sha256": _hash_texts(runtime),
        "max_abs_difference": 0.0 if mismatch_count == 0 else None,
        "mismatch_count": int(mismatch_count),
        "exact": mismatch_count == 0,
    }


def _matrix_check(
    name: str,
    offline: sparse.spmatrix | np.ndarray,
    runtime: sparse.spmatrix | np.ndarray,
) -> dict[str, object]:
    maximum, mismatches = _matrix_max_abs_difference(offline, runtime)
    return {
        "check": name,
        "comparison_type": "matrix",
        "offline_shape": json.dumps(list(offline.shape)),
        "runtime_shape": json.dumps(list(runtime.shape)),
        "offline_sha256": _matrix_hash(offline),
        "runtime_sha256": _matrix_hash(runtime),
        "max_abs_difference": maximum if math.isfinite(maximum) else None,
        "mismatch_count": mismatches,
        "exact": bool(maximum == 0.0 and mismatches == 0),
    }


def _zip_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    with archive.open(member) as handle:
        return _hash_bytes(iter(lambda: handle.read(1024 * 1024), b""))


def _verify_and_extract_package(
    project_root: Path,
    temporary_root: Path,
) -> tuple[AuditArtifacts, dict[str, object]]:
    package = project_root / PACKAGE_RELATIVE_PATH
    smoke_data = project_root / SMOKE_DATA_RELATIVE_PATH
    if not package.is_file():
        raise FileNotFoundError(f"Missing v0.4 package: {package}")
    if not smoke_data.is_dir():
        raise FileNotFoundError(f"Missing preserved smoke data: {smoke_data}")

    package_sha = sha256_file(package)
    if package_sha != EXPECTED_PACKAGE_SHA256:
        raise RuntimeError(
            f"v0.4 package digest mismatch: expected {EXPECTED_PACKAGE_SHA256}, found {package_sha}"
        )

    extraction = temporary_root / "package"
    extraction.mkdir(parents=True, exist_ok=True)
    required = {
        "main.py",
        "assets/final_ensemble_v04_rank.joblib",
        "assets/final_ensemble_v04_rank.metadata.json",
        "assets/supervised_delta.safetensors",
    }
    local_asset_root = project_root / "assets" / "pretrained"
    asset_checks: list[dict[str, object]] = []
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"v0.4 package is missing required members: {sorted(missing)}")
        for member in sorted(required):
            archive.extract(member, extraction)

        metadata = json.loads(
            archive.read("assets/final_ensemble_v04_rank.metadata.json").decode("utf-8")
        )
        artifact_sha = _zip_member_sha256(archive, "assets/final_ensemble_v04_rank.joblib")
        delta_sha = _zip_member_sha256(archive, "assets/supervised_delta.safetensors")
        if artifact_sha != metadata["artifact_sha256"]:
            raise RuntimeError("Packaged v0.4 joblib does not match its metadata digest.")
        if delta_sha != metadata["supervised_delta_sha256"]:
            raise RuntimeError("Packaged supervised delta does not match its metadata digest.")

        for slug in ("bge-base-en-v1.5", "bge-small-en-v1.5"):
            prefix = f"assets/{slug}/"
            members = sorted(
                name for name in names if name.startswith(prefix) and not name.endswith("/")
            )
            if not members:
                raise RuntimeError(f"Package has no files for {slug}.")
            for member in members:
                relative = Path(member.removeprefix(prefix))
                local = local_asset_root / slug / relative
                if not local.is_file():
                    raise FileNotFoundError(f"Local parity asset is missing: {local}")
                packaged_sha = _zip_member_sha256(archive, member)
                local_sha = sha256_file(local)
                asset_checks.append(
                    {
                        "member": member,
                        "packaged_sha256": packaged_sha,
                        "local_sha256": local_sha,
                        "match": packaged_sha == local_sha,
                    }
                )
    mismatched_assets = [row["member"] for row in asset_checks if not row["match"]]
    if mismatched_assets:
        raise RuntimeError(f"Packaged/local encoder assets differ: {mismatched_assets}")

    artifacts = AuditArtifacts(
        package=package,
        smoke_features=smoke_data / "test_features.csv",
        smoke_transcripts=smoke_data / "test_transcripts",
        model=extraction / "assets" / "final_ensemble_v04_rank.joblib",
        delta=extraction / "assets" / "supervised_delta.safetensors",
        package_main=extraction / "main.py",
    )
    verification = {
        "package_sha256": package_sha,
        "package_sha256_expected": EXPECTED_PACKAGE_SHA256,
        "package_sha256_match": True,
        "main_py_sha256": sha256_file(artifacts.package_main),
        "artifact_sha256": sha256_file(artifacts.model),
        "artifact_sha256_expected": metadata["artifact_sha256"],
        "supervised_delta_sha256": sha256_file(artifacts.delta),
        "supervised_delta_sha256_expected": metadata["supervised_delta_sha256"],
        "encoder_asset_files_verified": len(asset_checks),
        "encoder_asset_hashes_match": True,
        "encoder_asset_checks": asset_checks,
    }
    local_model = project_root / "models" / "final_ensemble_v04_rank.joblib"
    local_delta = project_root / "submission_src" / "assets" / "supervised_delta.safetensors"
    if not local_model.is_file() or not local_delta.is_file():
        raise FileNotFoundError(
            "Local v0.4 joblib or supervised delta is unavailable for byte-identity audit."
        )
    local_model_sha = sha256_file(local_model)
    local_delta_sha = sha256_file(local_delta)
    verification.update(
        {
            "local_artifact_path": str(local_model.relative_to(project_root)),
            "local_artifact_sha256": local_model_sha,
            "packaged_local_artifact_match": local_model_sha == artifact_sha,
            "local_supervised_delta_path": str(local_delta.relative_to(project_root)),
            "local_supervised_delta_sha256": local_delta_sha,
            "packaged_local_supervised_delta_match": local_delta_sha == delta_sha,
        }
    )
    replay_report_path = project_root / "submission_builds" / "final_ensemble_v04_rank.verification.json"
    replay_output = project_root / "tmp" / "final_ensemble_v04_rank_replay" / "submission.csv"
    smoke_output = project_root / "tmp" / "final_ensemble_v04_rank_submission" / "submission.csv"
    if not replay_report_path.is_file() or not replay_output.is_file() or not smoke_output.is_file():
        raise FileNotFoundError("Preserved exact-package replay evidence is incomplete.")
    replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    replay_bytes_sha = sha256_file(replay_output)
    smoke_bytes_sha = sha256_file(smoke_output)
    replay_integrity = bool(
        replay_report.get("package_sha256") == EXPECTED_PACKAGE_SHA256
        and replay_report.get("package_sha256") == package_sha
        and replay_report.get("replay_count") == 2
        and replay_report.get("output_bytes_identical") is True
        and replay_report.get("max_probability_difference") == 0.0
        and replay_output.read_bytes() == smoke_output.read_bytes()
    )
    verification.update(
        {
            "replay_verification_path": str(replay_report_path.relative_to(project_root)),
            "replay_count": replay_report.get("replay_count"),
            "replay_output_bytes_identical": replay_report.get("output_bytes_identical"),
            "replay_package_sha256": replay_report.get("package_sha256"),
            "replay_output_sha256": replay_bytes_sha,
            "smoke_output_sha256": smoke_bytes_sha,
            "replay_smoke_output_bytes_match": replay_bytes_sha == smoke_bytes_sha,
            "replay_integrity_verified": replay_integrity,
        }
    )
    if not replay_integrity:
        raise RuntimeError("Preserved exact-package replay evidence failed integrity checks.")
    return artifacts, verification


def import_packaged_main(path: Path) -> types.ModuleType:
    """Import the exact ``main.py`` extracted from the submitted ZIP."""
    module_name = f"trace_ace_packaged_v04_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import packaged runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample_frame(project_root: Path, smoke_features_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    smoke = pd.read_csv(
        smoke_features_path,
        dtype={
            "response_id": "string",
            "session_id": "string",
            "learning_objective_id": "string",
            "learning_objective": "string",
        },
    )
    if len(smoke) != 100 or smoke["response_id"].duplicated().any():
        raise ValueError("The preserved smoke set must contain 100 unique response rows.")
    modeling = pd.read_parquet(project_root / "data_cache" / "modeling_base.parquet").reset_index(
        drop=True
    )
    position = pd.Series(np.arange(len(modeling)), index=modeling["response_id"])
    rows = smoke["response_id"].map(position)
    if rows.isna().any():
        missing = smoke.loc[rows.isna(), "response_id"].tolist()
        raise ValueError(f"Smoke rows do not map to training cache: {missing}")
    positions = rows.to_numpy(dtype=np.int64)
    cached = modeling.iloc[positions].reset_index(drop=True)
    for column in ("response_id", "session_id", "learning_objective_id", "learning_objective"):
        if not cached[column].astype("string").equals(smoke[column].astype("string")):
            raise ValueError(f"Smoke feature column differs from training cache: {column}")
    return smoke, positions


def _runtime_records(module: types.ModuleType, features: pd.DataFrame, transcript_dir: Path):
    sessions = pd.Index(features["session_id"].drop_duplicates())
    records = [module._read_transcript(transcript_dir / f"{session_id}.csv") for session_id in sessions]
    lookup = pd.Series(np.arange(len(sessions)), index=sessions)
    response_rows = features["session_id"].map(lookup).to_numpy(dtype=np.int64)
    return sessions, records, response_rows


def _offline_component_features(
    project_root: Path,
    sample_positions: np.ndarray,
    sample_frame: pd.DataFrame,
    artifact: Mapping[str, object],
) -> dict[str, sparse.csr_matrix | np.ndarray]:
    cache_dir = project_root / "data_cache"
    full_session, _ = prepare_hash_matrices(
        pd.read_parquet(cache_dir / "modeling_base.parquet").reset_index(drop=True), cache_dir
    )
    full_all = _response_transcript_matrix(
        pd.read_parquet(cache_dir / "modeling_base.parquet").reset_index(drop=True),
        cache_dir,
        full_session,
    )
    full = full_all[sample_positions]

    modeling = pd.read_parquet(cache_dir / "modeling_base.parquet").reset_index(drop=True)
    session_order, roles = prepare_role_hashes(cache_dir)
    response_rows_all = _response_rows(modeling, session_order)
    response_rows = response_rows_all[sample_positions]
    student = roles["student"][response_rows]
    tutor = roles["tutor"][response_rows]
    opening = roles["opening"][response_rows]
    closing_student = roles["closing_student"][response_rows]
    closing_tutor = roles["closing_tutor"][response_rows]
    objective_64k = _objective_hash(sample_frame, ROLE_FEATURES)
    objective_32k = _objective_hash(sample_frame, 2**15)
    role_sparse = _unit_weighted_hstack(
        [(student, 0.75), (tutor, 0.55), (objective_64k, 1.0)]
    )
    behavior_all, _ = prepare_behavior_features(
        modeling, cache_dir, session_order, response_rows_all
    )
    behavior = behavior_all[sample_positions]
    alignment = np.column_stack(
        [
            _row_cosine(student, objective_64k),
            _row_cosine(tutor, objective_64k),
            _row_cosine(opening, objective_64k),
            _row_cosine(closing_student, objective_32k),
            _row_cosine(closing_tutor, objective_32k),
        ]
    )
    role_dense = artifact["role_scaler"].transform(np.column_stack([behavior, alignment]))
    role = sparse.hstack(
        [role_sparse, sparse.csr_matrix(role_dense.astype(np.float32) * np.float32(0.2))],
        format="csr",
    )

    bge_features, bge_similarity = prepare_bge_base_features(cache_dir, modeling)
    _, semantic_dense, _ = prepare_semantic_features(cache_dir, modeling)
    semantic_dense = semantic_dense.copy()
    semantic_dense[:, -1:] = bge_similarity
    semantic_scaled = artifact["semantic_scaler"].transform(
        semantic_dense[sample_positions]
    ).astype(np.float32)
    semantic = np.column_stack(
        [bge_features[sample_positions], semantic_scaled * np.float32(0.08)]
    )

    feedback_word, _, _ = prepare_feedback_hashes(cache_dir, modeling["response_id"])
    ordered = pd.read_parquet(cache_dir / "response_ordered_features.parquet")
    if list(ordered["response_id"]) != list(modeling["response_id"]):
        raise ValueError("Ordered cache row order does not match modeling data.")
    ordered_values = ordered.loc[sample_positions, ORDERED_FEATURE_NAMES].to_numpy(
        dtype=np.float64
    )
    ratio = np.asarray(artifact["nbsvm_log_count_ratio"], dtype=np.float32)
    nb_word = feedback_word[sample_positions].multiply(ratio).tocsr()
    nb_dense = artifact["nbsvm_scaler"].transform(ordered_values).astype(np.float32)
    nbsvm = sparse.hstack(
        [nb_word, sparse.csr_matrix(nb_dense * np.float32(artifact["nbsvm_dense_weight"]))],
        format="csr",
    )
    return {
        "full_transcript": full,
        "role_objective_dense": role,
        "bge_base_semantic": semantic,
        "nbsvm_feedback": nbsvm,
    }


def _runtime_component_features(
    module: types.ModuleType,
    features: pd.DataFrame,
    transcript_dir: Path,
    encoder_dir: Path,
    artifact: Mapping[str, object],
) -> tuple[dict[str, sparse.csr_matrix | np.ndarray], dict[str, np.ndarray]]:
    captured: dict[str, _CaptureEstimator] = {}
    runtime_artifact = dict(artifact)
    for component, key in (
        ("full_transcript", "full_model"),
        ("role_objective_dense", "role_model"),
        ("bge_base_semantic", "semantic_model"),
    ):
        proxy = _CaptureEstimator(artifact[key])
        captured[component] = proxy
        runtime_artifact[key] = proxy
    module.predict_final_ensemble(features, runtime_artifact, transcript_dir, encoder_dir)

    nb_proxy = _CaptureEstimator(artifact["nbsvm_model"])
    nb_artifact = dict(artifact)
    nb_artifact["nbsvm_model"] = nb_proxy
    module.predict_nbsvm_feedback(features, nb_artifact, transcript_dir)
    captured["nbsvm_feedback"] = nb_proxy

    matrices: dict[str, sparse.csr_matrix | np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for component, proxy in captured.items():
        if proxy.matrix is None or proxy.probability is None:
            raise RuntimeError(f"Packaged runtime did not execute {component}.")
        matrices[component] = proxy.matrix
        probabilities[component] = proxy.probability
    return matrices, probabilities


def _infer_supervised_pairs(
    tokenizer,
    model,
    device: str,
    contexts: Sequence[str],
    objectives: Sequence[str],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    keys, tensors = training_tokenize_pairs(
        tokenizer,
        list(contexts),
        list(objectives),
        int(max_length),
    )
    encoded = dict(zip(keys, tensors))
    token_matrix = encoded["input_ids"].cpu().numpy().astype(np.int64)
    batches: list[np.ndarray] = []
    batch_size = 128 if device == "cuda" else 32
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            stop = min(len(contexts), start + batch_size)
            inputs = {
                key: value[start:stop].to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            probability = torch.sigmoid(model(**inputs).logits.squeeze(-1))
            batches.append(probability.cpu().numpy().astype(np.float64))
    return token_matrix, np.concatenate(batches)


def _supervised_parity(
    module: types.ModuleType,
    project_root: Path,
    features: pd.DataFrame,
    transcript_dir: Path,
    offline_contexts: Sequence[str],
    encoder_dir: Path,
    delta_path: Path,
    artifact: Mapping[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, object],
]:
    _, records, response_rows = _runtime_records(module, features, transcript_dir)
    objectives = features["learning_objective"].fillna("").astype(str).tolist()
    runtime_contexts: list[str] = []
    for index, objective in enumerate(objectives):
        context, _ = module.retrieve_objective_context(
            str(records[response_rows[index]]["full_text"]), objective
        )
        runtime_contexts.append(context)

    tokenizer, model, device = module._load_supervised_model(encoder_dir, delta_path)
    max_length = int(artifact["supervised_max_length"])
    offline_tokens, offline_probability = _infer_supervised_pairs(
        tokenizer, model, device, offline_contexts, objectives, max_length
    )
    capture_tokenizer = _CaptureTokenizer(tokenizer)
    original_loader = module._load_supervised_model
    module._load_supervised_model = lambda *_args, **_kwargs: (
        capture_tokenizer,
        model,
        device,
    )
    try:
        runtime_probability = module.predict_supervised_component(
            features, artifact, transcript_dir, encoder_dir, delta_path
        )
    finally:
        module._load_supervised_model = original_loader
    if not capture_tokenizer.input_id_batches:
        raise RuntimeError("Packaged supervised runtime made no tokenizer calls.")
    runtime_tokens = np.concatenate(capture_tokenizer.input_id_batches, axis=0)

    training_report_path = (
        project_root
        / "experiments"
        / "runs"
        / FINAL_SUPERVISED_TRAIN_RUN_ID
        / "report.json"
    )
    if not training_report_path.is_file():
        raise FileNotFoundError(
            f"Final supervised training contract is missing: {training_report_path}"
        )
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    training_hypotheses = [training_mastery_hypothesis(value) for value in objectives]
    runtime_kwargs_exact = bool(
        capture_tokenizer.keyword_arguments
        and all(
            kwargs.get("padding") == "max_length"
            and kwargs.get("truncation") == "only_first"
            and kwargs.get("max_length") == max_length
            and kwargs.get("return_tensors") == "pt"
            for kwargs in capture_tokenizer.keyword_arguments
        )
    )
    hypothesis_exact = capture_tokenizer.hypotheses == training_hypotheses
    runtime_context_order_exact = capture_tokenizer.contexts == list(runtime_contexts)
    max_length_exact = bool(
        max_length == TRAINING_SUPERVISED_MAX_LENGTH
        and training_report.get("max_length") == TRAINING_SUPERVISED_MAX_LENGTH
    )
    training_model = str(training_report.get("model", ""))
    artifact_model = str(artifact.get("supervised_model", ""))
    base_model_exact = bool(
        training_model == artifact_model
        and training_model.rsplit("/", 1)[-1] == encoder_dir.name
    )
    actual_delta_sha = sha256_file(delta_path)
    delta_contract_exact = bool(
        training_report.get("delta_sha256") == actual_delta_sha
        and artifact.get("supervised_delta_sha256") == actual_delta_sha
    )
    contract = {
        "training_run_id": FINAL_SUPERVISED_TRAIN_RUN_ID,
        "training_hypothesis_source": (
            "trace_ace.supervised_encoder_screen.mastery_hypothesis"
        ),
        "training_tokenization_source": (
            "trace_ace.supervised_encoder_screen._tokenize_pairs"
        ),
        "runtime_tokenization_source": (
            "tokenizer calls captured inside exact packaged main.py"
        ),
        "hypotheses_exact": hypothesis_exact,
        "runtime_context_order_exact": runtime_context_order_exact,
        "runtime_padding": "max_length",
        "runtime_truncation": "only_first",
        "runtime_tokenization_arguments_exact": runtime_kwargs_exact,
        "training_max_length": int(training_report["max_length"]),
        "artifact_max_length": max_length,
        "max_length_exact": max_length_exact,
        "training_base_model": training_model,
        "artifact_base_model": artifact_model,
        "encoder_directory": encoder_dir.name,
        "base_model_exact": base_model_exact,
        "delta_contract_exact": delta_contract_exact,
        "probability_comparison_scope": (
            "Both probability paths use the exact packaged base model and delta; "
            "independence is at the training-vs-package hypothesis and tokenization "
            "boundary, not an independently reconstructed model."
        ),
    }
    contract["passed"] = bool(
        hypothesis_exact
        and runtime_context_order_exact
        and runtime_kwargs_exact
        and max_length_exact
        and base_model_exact
        and delta_contract_exact
    )
    return (
        offline_tokens,
        runtime_tokens,
        offline_probability,
        runtime_probability,
        runtime_contexts,
        contract,
    )


def _expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    weights = (
        np.ones(len(target), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    assignments = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    total_weight = float(weights.sum())
    error = 0.0
    for bin_index in range(int(bins)):
        mask = assignments == bin_index
        bin_weight = float(weights[mask].sum())
        if bin_weight <= 0.0:
            continue
        observed = float(np.average(target[mask], weights=weights[mask]))
        predicted = float(np.average(probability[mask], weights=weights[mask]))
        error += (bin_weight / total_weight) * abs(observed - predicted)
    return float(error)


def _calibration_line(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if len(np.unique(target)) < 2:
        return float("nan"), float("nan")
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, tol=1e-10)
    model.fit(logit, target, sample_weight=sample_weight)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _metric_row(
    frame: pd.DataFrame,
    *,
    model: str,
    protocol: str,
    fold: str,
    fidelity: str,
    source: str,
) -> dict[str, object]:
    target = frame["target"].to_numpy(dtype=np.int8)
    probability = np.clip(frame["probability"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    intercept, slope = _calibration_line(target, probability)
    auc = float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else float("nan")
    return {
        "protocol": protocol,
        "fold": fold,
        "model": model,
        "fidelity": fidelity,
        "source": source,
        "rows": int(len(frame)),
        "positive_rate": float(target.mean()),
        "prediction_mean": float(probability.mean()),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": auc,
        "brier_score": float(brier_score_loss(target, probability)),
        "ece_10": _expected_calibration_error(target, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_absolute_logit": float(
            np.mean(np.abs(np.log(probability / (1.0 - probability))))
        ),
    }


def _validate_oof_source_frame(frame: pd.DataFrame, label: str) -> None:
    required = {"protocol", "response_id", "target", "fold", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OOF source {label!r} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"OOF source {label!r} is empty.")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"OOF source {label!r} contains missing identity or prediction data.")
    response_ids = frame["response_id"].astype("string")
    if response_ids.str.strip().eq("").any():
        raise ValueError(f"OOF source {label!r} contains blank response IDs.")
    if frame.duplicated(["protocol", "response_id"]).any():
        raise ValueError(
            f"OOF source {label!r} contains duplicate response IDs within a protocol."
        )
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=np.float64)
    fold = pd.to_numeric(frame["fold"], errors="coerce").to_numpy(dtype=np.float64)
    probability = pd.to_numeric(
        frame["probability"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not set(np.unique(target)).issubset({0.0, 1.0}):
        raise ValueError(f"OOF source {label!r} contains invalid targets.")
    if not np.isfinite(fold).all() or not np.equal(fold, np.floor(fold)).all():
        raise ValueError(f"OOF source {label!r} contains invalid folds.")
    if not np.isfinite(probability).all():
        raise ValueError(f"OOF source {label!r} contains non-finite probabilities.")


def audit_oof_integrity(oof: pd.DataFrame) -> dict[str, object]:
    """Reject ambiguous OOF identities before cross-source calibration comparisons."""
    source_columns = ["protocol", "model", "fidelity", "source"]
    required = {*source_columns, "response_id", "target", "fold", "probability"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF audit is missing columns: {sorted(missing)}")
    if oof.empty:
        raise ValueError("OOF audit input is empty.")
    if oof[list(required)].isna().any().any():
        raise ValueError("OOF audit contains missing source, identity, target, fold, or prediction data.")

    source_rows: list[dict[str, object]] = []
    for keys, frame in oof.groupby(source_columns, sort=False, dropna=False):
        protocol, model, fidelity, source = keys
        label = f"{protocol}/{model}/{fidelity}/{source}"
        _validate_oof_source_frame(frame, label)
        source_rows.append(
            {
                "protocol": str(protocol),
                "model": str(model),
                "fidelity": str(fidelity),
                "source": str(source),
                "rows": int(len(frame)),
                "unique_response_ids": int(frame["response_id"].nunique()),
            }
        )

    normalized = oof[["protocol", "response_id", "target", "fold"]].copy()
    normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    normalized["fold"] = pd.to_numeric(normalized["fold"], errors="coerce")
    consistency = normalized.groupby(
        ["protocol", "response_id"], sort=False, dropna=False
    ).agg(target_values=("target", "nunique"), fold_values=("fold", "nunique"))
    conflicts = consistency.loc[
        consistency["target_values"].ne(1) | consistency["fold_values"].ne(1)
    ]
    if not conflicts.empty:
        examples = [tuple(map(str, index)) for index in conflicts.index[:10]]
        raise ValueError(
            "OOF target/fold values conflict across prediction sources for "
            f"protocol/response IDs: {examples}"
        )
    return {
        "passed": True,
        "rows": int(len(oof)),
        "prediction_sources": int(len(source_rows)),
        "protocol_response_ids": int(len(consistency)),
        "duplicate_source_response_ids": 0,
        "cross_source_target_fold_conflicts": 0,
        "sources": source_rows,
    }


def _append_prediction_set(
    output: list[pd.DataFrame],
    source_frame: pd.DataFrame,
    prediction_column: str,
    *,
    model: str,
    fidelity: str,
    source: str,
    protocol: str | None = None,
) -> None:
    columns = ["response_id", "target", "fold", prediction_column]
    if "protocol" in source_frame:
        columns.append("protocol")
    frame = source_frame[columns].copy()
    frame = frame.rename(columns={prediction_column: "probability"})
    if "protocol" not in frame:
        frame["protocol"] = protocol or PILOT_PROTOCOL
    frame["model"] = model
    frame["fidelity"] = fidelity
    frame["source"] = source
    _validate_oof_source_frame(frame, f"{frame['protocol'].iloc[0]}/{model}/{source}")
    output.append(frame)


def _strict_oof_alignment_merge(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_prediction_column: str,
    label: str,
    require_equal_coverage: bool,
) -> pd.DataFrame:
    """Align a candidate to locked combined OOF without inner-join row loss."""
    identity = ["response_id", "target", "fold"]
    required_reference = {*identity, "protocol", "pred_combined"}
    required_candidate = {*identity, candidate_prediction_column}
    missing_reference = required_reference.difference(reference.columns)
    missing_candidate = required_candidate.difference(candidate.columns)
    if missing_reference or missing_candidate:
        raise ValueError(
            f"{label} alignment columns are incomplete: "
            f"reference={sorted(missing_reference)}, candidate={sorted(missing_candidate)}"
        )
    if reference[identity].isna().any().any() or candidate[identity].isna().any().any():
        raise ValueError(f"{label} alignment contains missing response/target/fold identities.")
    if reference["response_id"].duplicated().any():
        raise ValueError(f"{label} reference contains duplicate response IDs.")
    if candidate["response_id"].duplicated().any():
        raise ValueError(f"{label} candidate contains duplicate response IDs.")

    reference_keys = set(reference[identity].itertuples(index=False, name=None))
    candidate_keys = set(candidate[identity].itertuples(index=False, name=None))
    missing_keys = candidate_keys.difference(reference_keys)
    if missing_keys:
        raise ValueError(
            f"{label} candidate has response/target/fold identities absent from the "
            f"locked combined OOF: {list(missing_keys)[:10]}"
        )
    if require_equal_coverage and reference_keys != candidate_keys:
        missing_candidate = reference_keys.difference(candidate_keys)
        raise ValueError(
            f"{label} does not cover the full locked combined OOF: "
            f"{list(missing_candidate)[:10]}"
        )

    merged = candidate[[*identity, candidate_prediction_column]].merge(
        reference[[*identity, "protocol", "pred_combined"]],
        on=identity,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(candidate) or not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{label} lost rows during OOF alignment.")
    return merged.drop(columns="_merge")


def reconstruct_oof_predictions(project_root: Path) -> pd.DataFrame:
    """Reconstruct the best available v0.4 OOF proxies without refitting outcome models."""
    run_root = project_root / "experiments" / "runs"
    robust_path = run_root / "20260716T183434Z_robust_validation" / "oof_predictions.parquet"
    bge_pilot_path = run_root / "20260716T232134Z_encoder_upgrade_validation" / "oof_predictions.parquet"
    bge_confirm_path = run_root / "20260716T232346Z_encoder_upgrade_validation" / "oof_predictions.parquet"
    nb_pilot_path = run_root / "20260717T023053Z_nbsvm_validation" / "oof_predictions.parquet"
    nb_confirm_path = run_root / "20260717T023209Z_nbsvm_validation" / "oof_predictions.parquet"
    combined_path = run_root / "20260717T023245Z_combined_candidate_validation" / "oof_predictions.parquet"
    supervised_4096_path = (
        run_root / "20260717T045715Z_supervised_encoder_confirmation" / "oof_predictions.parquet"
    )
    supervised_full_paths = {
        0: run_root / "20260717T082126Z_supervised_encoder_screen" / "validation_predictions.parquet",
        3: run_root / "20260717T063544Z_supervised_encoder_screen" / "validation_predictions.parquet",
    }
    required = [
        robust_path,
        bge_pilot_path,
        bge_confirm_path,
        nb_pilot_path,
        nb_confirm_path,
        combined_path,
        supervised_4096_path,
        *supervised_full_paths.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OOF reconstruction inputs are missing: {missing}")

    robust = pd.read_parquet(robust_path)
    bge_pilot = pd.read_parquet(bge_pilot_path)
    bge_pilot = bge_pilot.loc[bge_pilot["regularization_c"].eq(0.1)].copy()
    bge = pd.concat([bge_pilot, pd.read_parquet(bge_confirm_path)], ignore_index=True)
    nb_pilot = pd.read_parquet(nb_pilot_path)
    nb_pilot = nb_pilot.loc[
        nb_pilot["model"].eq("nbsvm_feedback_word_ordered_c1")
    ].copy()
    nb = pd.concat([nb_pilot, pd.read_parquet(nb_confirm_path)], ignore_index=True)
    combined = pd.read_parquet(combined_path)
    supervised_4096 = pd.read_parquet(supervised_4096_path)

    predictions: list[pd.DataFrame] = []
    robust_sources = {
        "pred_full": "full_transcript",
        "pred_role": "role_objective_dense",
        "pred_semantic": "bge_small_semantic_v02",
        "pred_v02": "v02",
    }
    for column, name in robust_sources.items():
        _append_prediction_set(
            predictions,
            robust,
            column,
            model=name,
            fidelity="exact_fold_local_oof_component",
            source=str(robust_path.relative_to(project_root)),
        )
    _append_prediction_set(
        predictions,
        bge,
        "pred_bge_base",
        model="bge_base_semantic",
        fidelity="exact_fold_local_oof_component_locked_c0.1",
        source="encoder_upgrade_validation pilot+confirmation",
    )
    _append_prediction_set(
        predictions,
        nb,
        "pred_nbsvm",
        model="nbsvm_feedback",
        fidelity="exact_fold_local_oof_component_locked_c1",
        source="nbsvm_validation pilot+confirmation",
    )
    _append_prediction_set(
        predictions,
        combined,
        "pred_combined",
        model="combined_pre_supervised",
        fidelity="exact_locked_formula_fold_local_oof",
        source=str(combined_path.relative_to(project_root)),
    )
    _append_prediction_set(
        predictions,
        supervised_4096,
        "pred_supervised",
        model="supervised_bge_small_4096",
        fidelity="proxy_4096_row_fold_local_oof_not_final_refit",
        source=str(supervised_4096_path.relative_to(project_root)),
        protocol=PILOT_PROTOCOL,
    )

    pilot_combined = combined.loc[combined["protocol"].eq(PILOT_PROTOCOL), [
        "response_id", "target", "fold", "protocol", "pred_combined"
    ]]
    proxy_4096 = _strict_oof_alignment_merge(
        pilot_combined,
        supervised_4096,
        candidate_prediction_column="pred_supervised",
        label="4096-row supervised proxy",
        require_equal_coverage=True,
    )
    proxy_4096["pred_v04_proxy"] = (
        0.7 * proxy_4096["pred_combined"] + 0.3 * proxy_4096["pred_supervised"]
    )
    _append_prediction_set(
        predictions,
        proxy_4096,
        "pred_v04_proxy",
        model="v04_proxy_4096",
        fidelity="formula_exact_components_proxy_supervised_4096_not_final_refit",
        source="combined_candidate + supervised_encoder_confirmation",
    )

    fullscale_frames: list[pd.DataFrame] = []
    for fold, path in supervised_full_paths.items():
        supervised = pd.read_parquet(path)
        if set(supervised["fold"]) != {fold}:
            raise RuntimeError(f"Unexpected fold content in {path}")
        fullscale_frames.append(supervised)
    fullscale = pd.concat(fullscale_frames, ignore_index=True)
    _append_prediction_set(
        predictions,
        fullscale,
        "pred_supervised",
        model="supervised_bge_small_fullscale_folds_0_3",
        fidelity="highest_available_fold_local_fullscale_oof_not_final_refit",
        source="supervised_encoder_screen folds 0 and 3",
        protocol=PILOT_PROTOCOL,
    )
    proxy_full = _strict_oof_alignment_merge(
        pilot_combined,
        fullscale,
        candidate_prediction_column="pred_supervised",
        label="full-scale supervised proxy",
        require_equal_coverage=False,
    )
    proxy_full["pred_v04_proxy"] = (
        0.7 * proxy_full["pred_combined"] + 0.3 * proxy_full["pred_supervised"]
    )
    _append_prediction_set(
        predictions,
        proxy_full,
        "pred_v04_proxy",
        model="v04_proxy_fullscale_folds_0_3",
        fidelity="highest_fidelity_available_formula_reconstruction_not_final_refit",
        source="combined_candidate + full-scale supervised folds 0 and 3",
    )
    output = pd.concat(predictions, ignore_index=True)
    output["probability"] = np.clip(output["probability"].to_numpy(float), 1e-6, 1 - 1e-6)
    return output


def calibration_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["protocol", "model", "fidelity", "source"]
    for keys, frame in oof.groupby(group_columns, sort=False, dropna=False):
        protocol, model, fidelity, source = keys
        rows.append(
            _metric_row(
                frame,
                model=str(model),
                protocol=str(protocol),
                fold="pooled",
                fidelity=str(fidelity),
                source=str(source),
            )
        )
        for fold, fold_frame in frame.groupby("fold", sort=True):
            rows.append(
                _metric_row(
                    fold_frame,
                    model=str(model),
                    protocol=str(protocol),
                    fold=str(int(fold)),
                    fidelity=str(fidelity),
                    source=str(source),
                )
            )
    return pd.DataFrame(rows)


def prior_shift_sensitivity(
    oof: pd.DataFrame,
    target_priors: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in oof.groupby(
        ["protocol", "model", "fidelity", "source"], sort=False, dropna=False
    ):
        protocol, model, fidelity, source = keys
        target = frame["target"].to_numpy(dtype=np.int8)
        probability = frame["probability"].to_numpy(dtype=np.float64)
        observed = float(target.mean())
        if observed <= 0.0 or observed >= 1.0:
            continue
        for target_prior in target_priors:
            weights = np.where(
                target == 1,
                float(target_prior) / observed,
                (1.0 - float(target_prior)) / (1.0 - observed),
            )
            intercept, slope = _calibration_line(target, probability, sample_weight=weights)
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "fidelity": fidelity,
                    "source": source,
                    "rows": int(len(frame)),
                    "observed_positive_rate": observed,
                    "assumed_positive_rate": float(target_prior),
                    "weighted_log_loss": float(
                        log_loss(target, probability, labels=[0, 1], sample_weight=weights)
                    ),
                    "weighted_brier_score": float(
                        np.average((target - probability) ** 2, weights=weights)
                    ),
                    "weighted_ece_10": _expected_calibration_error(
                        target, probability, sample_weight=weights
                    ),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "weighted_prediction_mean": float(np.average(probability, weights=weights)),
                }
            )
    return pd.DataFrame(rows)


def _quantile_groups(values: pd.Series, prefix: str, q: int = 4) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        grouped = pd.qcut(numeric, q=q, duplicates="drop")
    except ValueError:
        grouped = pd.cut(numeric, bins=q, duplicates="drop")
    return grouped.astype("string").fillna("missing").map(lambda value: f"{prefix}:{value}")


def _diagnostic_base(project_root: Path) -> pd.DataFrame:
    cache = project_root / "data_cache"
    base = pd.read_parquet(cache / "modeling_base.parquet").reset_index(drop=True)
    context = pd.read_parquet(
        cache / "response_objective_context.parquet",
        columns=["response_id", "retrieval_term_coverage"],
    )
    behavior = pd.read_parquet(cache / "session_behavior.parquet")
    assignments = pd.read_parquet(
        project_root
        / "experiments"
        / "runs"
        / "20260716T183434Z_robust_validation"
        / "objective_family_assignments.parquet"
    )
    family = assignments.loc[
        assignments["protocol"].eq(PILOT_PROTOCOL),
        ["learning_objective_id", "semantic_family"],
    ].drop_duplicates()
    if family["learning_objective_id"].duplicated().any():
        raise ValueError("Pilot semantic-family mapping has duplicate objective IDs.")
    merged = (
        base.merge(context, on="response_id", how="left", validate="one_to_one")
        .merge(behavior, on="session_id", how="left", validate="many_to_one")
        .merge(family, on="learning_objective_id", how="left", validate="many_to_one")
    )
    multiplicity = merged.groupby("session_id")["response_id"].transform("size")
    role_imbalance = (
        merged["student_fraction"].astype(float) - merged["tutor_fraction"].astype(float)
    ).abs()
    if merged["semantic_family"].isna().any():
        raise ValueError("At least one response lacks a semantic-family assignment.")
    merged["objective_family"] = merged["semantic_family"].map(
        lambda value: f"semantic_k50_s0_family:{int(value)}"
    )
    merged["transcript_length"] = _quantile_groups(
        merged["content_chars"], "content_chars_q"
    )
    merged["asr_uncertainty"] = _quantile_groups(
        merged["unclear_fraction"], "unclear_fraction_q"
    )
    merged["background_rate"] = _quantile_groups(
        merged["background_fraction"], "background_fraction_q"
    )
    merged["retrieval_coverage"] = pd.cut(
        merged["retrieval_term_coverage"].astype(float),
        bins=[-np.inf, 0.0, 0.25, 0.50, 0.75, 0.999999, np.inf],
        labels=["zero", "0_to_0.25", "0.25_to_0.50", "0.50_to_0.75", "0.75_to_lt1", "one"],
    ).astype("string").fillna("missing")
    merged["role_balance"] = _quantile_groups(role_imbalance, "absolute_role_imbalance_q")
    merged["session_multiplicity"] = pd.cut(
        multiplicity,
        bins=[0, 1, 2, 4, np.inf],
        labels=["1", "2", "3_to_4", "5_plus"],
    ).astype("string")
    return merged


def group_error_diagnostics(project_root: Path, oof: pd.DataFrame) -> pd.DataFrame:
    base = _diagnostic_base(project_root)
    pilot = oof.loc[oof["protocol"].eq(PILOT_PROTOCOL)].copy()
    dimensions = [
        "objective_family",
        "transcript_length",
        "asr_uncertainty",
        "background_rate",
        "retrieval_coverage",
        "role_balance",
        "session_multiplicity",
    ]
    rows: list[dict[str, object]] = []
    for keys, predictions in pilot.groupby(
        ["model", "fidelity", "source"], sort=False, dropna=False
    ):
        model, fidelity, source = keys
        aligned = predictions.merge(
            base[["response_id", *dimensions]],
            on="response_id",
            how="left",
            validate="one_to_one",
        )
        target = aligned["target"].to_numpy(dtype=np.int8)
        probability = aligned["probability"].to_numpy(dtype=np.float64)
        overall_loss = float(log_loss(target, probability, labels=[0, 1]))
        reference = (
            pilot.loc[pilot["model"].eq("v02"), ["response_id", "probability"]]
            .drop_duplicates("response_id")
            .rename(columns={"probability": "v02_probability"})
        )
        aligned = aligned.merge(reference, on="response_id", how="left", validate="one_to_one")
        for dimension in dimensions:
            for group, group_frame in aligned.groupby(dimension, sort=True, dropna=False):
                group_target = group_frame["target"].to_numpy(dtype=np.int8)
                group_probability = group_frame["probability"].to_numpy(dtype=np.float64)
                group_loss = float(log_loss(group_target, group_probability, labels=[0, 1]))
                rows.append(
                    {
                        "protocol": PILOT_PROTOCOL,
                        "model": model,
                        "fidelity": fidelity,
                        "source": source,
                        "group_dimension": dimension,
                        "group": str(group),
                        "rows": int(len(group_frame)),
                        "sessions": int(
                            base.loc[
                                base["response_id"].isin(group_frame["response_id"]), "session_id"
                            ].nunique()
                        ),
                        "positive_rate": float(group_target.mean()),
                        "prediction_mean": float(group_probability.mean()),
                        "log_loss": group_loss,
                        "excess_log_loss_vs_model_overall": group_loss - overall_loss,
                        "brier_score": float(
                            np.mean((group_target.astype(float) - group_probability) ** 2)
                        ),
                        "mean_absolute_error": float(
                            np.mean(np.abs(group_target.astype(float) - group_probability))
                        ),
                        "mean_absolute_disagreement_vs_v02": float(
                            np.nanmean(
                                np.abs(
                                    group_probability
                                    - group_frame["v02_probability"].to_numpy(dtype=float)
                                )
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _cached_preprocessing(
    project_root: Path,
    sample_positions: np.ndarray,
    sample_frame: pd.DataFrame,
) -> dict[str, object]:
    cache = project_root / "data_cache"
    sessions = pd.Index(sample_frame["session_id"].drop_duplicates())
    session_text = pd.read_parquet(cache / "session_texts.parquet").set_index("session_id")
    role_text = pd.read_parquet(cache / "session_role_texts.parquet").set_index("session_id")
    behavior = pd.read_parquet(cache / "session_behavior.parquet").set_index("session_id")
    contexts_all = pd.read_parquet(cache / "response_objective_context.parquet")
    multiview_all = pd.read_parquet(cache / "response_multiview_texts.parquet")
    ordered_all = pd.read_parquet(cache / "response_ordered_features.parquet")
    return {
        "sessions": sessions,
        "full_text": session_text.loc[sessions, "full_text"].astype(str).tolist(),
        "student_text": role_text.loc[sessions, "student_text"].fillna("").astype(str).tolist(),
        "tutor_text": role_text.loc[sessions, "tutor_text"].fillna("").astype(str).tolist(),
        "opening_text": role_text.loc[sessions, "opening_text"].fillna("").astype(str).tolist(),
        "closing_student_text": role_text.loc[sessions, "closing_student_text"].fillna("").astype(str).tolist(),
        "closing_tutor_text": role_text.loc[sessions, "closing_tutor_text"].fillna("").astype(str).tolist(),
        "behavior_raw": behavior.loc[sessions].to_numpy(dtype=np.float64),
        "objective_context": contexts_all.loc[sample_positions, "objective_context"].astype(str).tolist(),
        "retrieval_raw": contexts_all.loc[
            sample_positions,
            [
                "retrieval_total_lines",
                "retrieval_positive_lines",
                "retrieval_selected_lines",
                "retrieval_objective_terms",
                "retrieval_matched_terms",
                "retrieval_term_coverage",
                "retrieval_first_match_position",
                "retrieval_last_match_position",
                "retrieval_mean_positive_score",
                "retrieval_max_score",
            ],
        ].to_numpy(dtype=np.float64),
        "feedback_evidence": multiview_all.loc[
            sample_positions, "feedback_evidence"
        ].astype(str).tolist(),
        "ordered": ordered_all.loc[sample_positions, ORDERED_FEATURE_NAMES].to_numpy(
            dtype=np.float64
        ),
    }


def _runtime_preprocessing(
    module: types.ModuleType,
    features: pd.DataFrame,
    transcript_dir: Path,
) -> dict[str, object]:
    sessions, records, response_rows = _runtime_records(module, features, transcript_dir)
    objectives = features["learning_objective"].fillna("").astype(str).tolist()
    contexts: list[str] = []
    retrieval: list[np.ndarray] = []
    feedback: list[str] = []
    ordered: list[np.ndarray] = []
    for row_index, objective in enumerate(objectives):
        record = records[response_rows[row_index]]
        context, stats = module.retrieve_objective_context(str(record["full_text"]), objective)
        contexts.append(context)
        retrieval.append(stats)
        feedback.append(module._feedback_windows(record["dialogue_lines"], objective))
        ordered.append(module._ordered_feature_values(record["dialogue_lines"], objective))
    return {
        "sessions": sessions,
        "full_text": [str(record["full_text"]) for record in records],
        "student_text": [str(record["student_text"]) for record in records],
        "tutor_text": [str(record["tutor_text"]) for record in records],
        "opening_text": [str(record["opening_text"]) for record in records],
        "closing_student_text": [str(record["closing_student_text"]) for record in records],
        "closing_tutor_text": [str(record["closing_tutor_text"]) for record in records],
        "behavior_raw": np.vstack([record["behavior"] for record in records]),
        "objective_context": contexts,
        "retrieval_raw": np.vstack(retrieval),
        "feedback_evidence": feedback,
        "ordered": np.vstack(ordered),
    }


def _transcript_hash_audit(
    project_root: Path,
    features: pd.DataFrame,
    smoke_transcript_dir: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for session_id in features["session_id"].drop_duplicates():
        training = project_root / "Train transcripts" / f"{session_id}.csv"
        smoke = smoke_transcript_dir / f"{session_id}.csv"
        if not training.is_file() or not smoke.is_file():
            raise FileNotFoundError(f"Missing transcript parity source for session {session_id}")
        training_sha = sha256_file(training)
        smoke_sha = sha256_file(smoke)
        rows.append(
            {
                "session_id": str(session_id),
                "training_sha256": training_sha,
                "smoke_sha256": smoke_sha,
                "match": training_sha == smoke_sha,
            }
        )
    mismatches = [row["session_id"] for row in rows if not row["match"]]
    if mismatches:
        raise RuntimeError(f"Smoke and training transcript files differ: {mismatches}")
    return {
        "files_verified": len(rows),
        "all_match": True,
        "combined_training_sha256": _hash_texts([row["training_sha256"] for row in rows]),
        "combined_smoke_sha256": _hash_texts([row["smoke_sha256"] for row in rows]),
    }


def _verified_replay_probability(project_root: Path, features: pd.DataFrame) -> np.ndarray:
    replay_path = project_root / "tmp" / "final_ensemble_v04_rank_replay" / "submission.csv"
    replay = pd.read_csv(replay_path, dtype={"response_id": "string"})
    if list(replay.columns) != ["response_id", "probability"]:
        raise ValueError("Preserved v0.4 replay has an invalid output schema.")
    if replay["response_id"].duplicated().any():
        raise ValueError("Preserved v0.4 replay contains duplicate response IDs.")
    aligned = features[["response_id"]].merge(
        replay, on="response_id", how="left", validate="one_to_one"
    )
    if aligned["probability"].isna().any() or len(aligned) != len(features):
        raise ValueError("Preserved v0.4 replay does not align with the 100-row audit sample.")
    probability = aligned["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all():
        raise ValueError("Preserved v0.4 replay contains non-finite probabilities.")
    return probability


def gate_a_decision(
    component_parity: pd.DataFrame,
    preprocessing_parity: pd.DataFrame,
    package_verification: Mapping[str, object],
    supervised_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    required_component_columns = {
        "response_id",
        "component",
        "expected_width",
        "offline_width",
        "runtime_width",
        "offline_probability",
        "runtime_probability",
        "absolute_probability_difference",
    }
    missing_component_columns = required_component_columns.difference(
        component_parity.columns
    )
    if missing_component_columns:
        raise ValueError(
            "Gate A component evidence is missing columns: "
            f"{sorted(missing_component_columns)}"
        )
    required_preprocessing_columns = {
        "check",
        "comparison_type",
        "offline_shape",
        "runtime_shape",
        "max_abs_difference",
        "exact",
    }
    missing_preprocessing_columns = required_preprocessing_columns.difference(
        preprocessing_parity.columns
    )
    if missing_preprocessing_columns:
        raise ValueError(
            "Gate A preprocessing evidence is missing columns: "
            f"{sorted(missing_preprocessing_columns)}"
        )

    expected_components = set(GATE_A_COMPONENTS.values())
    component_labels = component_parity["component"].astype("string")
    actual_components = set(component_labels.dropna().astype(str))
    component_set_exact = actual_components == expected_components
    component_row_counts = {
        label: int(component_labels.eq(label).sum())
        for label in GATE_A_COMPONENTS.values()
    }
    component_rows_exact = bool(
        component_set_exact
        and all(count == GATE_A_SAMPLE_ROWS for count in component_row_counts.values())
        and len(component_parity) == len(expected_components) * GATE_A_SAMPLE_ROWS
    )
    response_ids = component_parity["response_id"].astype("string")
    response_ids_valid = bool(
        response_ids.notna().all() and response_ids.str.strip().ne("").all()
    )
    duplicate_component_response_ids = int(
        component_parity.assign(response_id=response_ids).duplicated(
            ["component", "response_id"]
        ).sum()
    )
    component_id_sets = {
        label: set(
            response_ids.loc[component_labels.eq(label)].dropna().astype(str).tolist()
        )
        for label in GATE_A_COMPONENTS.values()
    }
    reference_ids = component_id_sets[GATE_A_COMPONENTS["full"]]
    component_coverage_exact = bool(
        component_set_exact
        and component_rows_exact
        and response_ids_valid
        and duplicate_component_response_ids == 0
        and len(reference_ids) == GATE_A_SAMPLE_ROWS
        and all(ids == reference_ids for ids in component_id_sets.values())
    )
    coverage_union = set().union(*component_id_sets.values())
    coverage_intersection = (
        set.intersection(*component_id_sets.values()) if component_id_sets else set()
    )

    offline_probability = pd.to_numeric(
        component_parity["offline_probability"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    runtime_probability = pd.to_numeric(
        component_parity["runtime_probability"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    probability_difference = pd.to_numeric(
        component_parity["absolute_probability_difference"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    offline_probabilities_finite = bool(np.isfinite(offline_probability).all())
    runtime_probabilities_finite = bool(np.isfinite(runtime_probability).all())
    probability_differences_finite = bool(np.isfinite(probability_difference).all())
    component_probabilities_finite = bool(
        offline_probabilities_finite
        and runtime_probabilities_finite
        and probability_differences_finite
    )
    maximum_probability_difference = (
        float(np.max(probability_difference))
        if len(probability_difference) and probability_differences_finite
        else None
    )

    expected_width = pd.to_numeric(
        component_parity["expected_width"], errors="coerce"
    )
    offline_width = pd.to_numeric(component_parity["offline_width"], errors="coerce")
    runtime_width = pd.to_numeric(component_parity["runtime_width"], errors="coerce")
    locked_width = component_labels.map(EXPECTED_WIDTHS)
    widths_exact = bool(
        component_set_exact
        and expected_width.notna().all()
        and offline_width.notna().all()
        and runtime_width.notna().all()
        and expected_width.eq(locked_width).all()
        and offline_width.eq(locked_width).all()
        and runtime_width.eq(locked_width).all()
    )

    preprocessing_exact = bool(
        len(preprocessing_parity) > 0 and preprocessing_parity["exact"].eq(True).all()
    )
    text_rows = preprocessing_parity["comparison_type"].eq("text")
    token_rows = preprocessing_parity["check"].eq("supervised_input_ids")
    discrete_exact = bool(
        (text_rows | token_rows).any()
        and preprocessing_parity.loc[text_rows | token_rows, "exact"].eq(True).all()
    )
    numeric = preprocessing_parity.loc[
        preprocessing_parity["comparison_type"].eq("matrix")
    ].copy()
    numeric_shapes_exact = bool(
        len(numeric) > 0
        and numeric["offline_shape"].astype(str).eq(numeric["runtime_shape"].astype(str)).all()
    )
    numeric_differences = pd.to_numeric(
        numeric["max_abs_difference"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    numeric_differences_finite = bool(
        len(numeric_differences) > 0 and np.isfinite(numeric_differences).all()
    )
    numeric_maximum = (
        float(np.max(numeric_differences)) if numeric_differences_finite else None
    )
    numeric_within_tolerance = bool(
        numeric_shapes_exact
        and numeric_differences_finite
        and numeric_maximum is not None
        and numeric_maximum <= PREPROCESSING_TOLERANCE
    )
    preprocessing_within_tolerance = bool(discrete_exact and numeric_within_tolerance)
    local_assets_exact = bool(
        package_verification.get("packaged_local_artifact_match", False)
        and package_verification.get("packaged_local_supervised_delta_match", False)
        and package_verification.get("encoder_asset_hashes_match", False)
        and package_verification.get("replay_integrity_verified", False)
    )
    supervised_contract_exact = bool(
        supervised_contract is not None and supervised_contract.get("passed", False)
    )
    final_mask = component_labels.eq(GATE_A_COMPONENTS["final"])
    final_differences = probability_difference[final_mask.to_numpy(dtype=bool)]
    final_output_maximum = (
        float(np.max(final_differences))
        if len(final_differences) == GATE_A_SAMPLE_ROWS
        and np.isfinite(final_differences).all()
        else None
    )
    passed = bool(
        component_set_exact
        and component_rows_exact
        and component_coverage_exact
        and component_probabilities_finite
        and widths_exact
        and preprocessing_within_tolerance
        and local_assets_exact
        and supervised_contract_exact
        and maximum_probability_difference is not None
        and maximum_probability_difference <= GATE_A_TOLERANCE
        and final_output_maximum is not None
        and final_output_maximum <= GATE_A_TOLERANCE
    )
    return {
        "tolerance": GATE_A_TOLERANCE,
        "expected_component_labels": dict(GATE_A_COMPONENTS),
        "actual_component_labels": sorted(actual_components),
        "component_set_exact": component_set_exact,
        "component_row_counts": component_row_counts,
        "component_rows_exact": component_rows_exact,
        "duplicate_component_response_ids": duplicate_component_response_ids,
        "component_coverage_exact": component_coverage_exact,
        "component_coverage_gap_count": int(
            len(coverage_union.difference(coverage_intersection))
        ),
        "offline_component_probabilities_finite": offline_probabilities_finite,
        "runtime_component_probabilities_finite": runtime_probabilities_finite,
        "component_probability_differences_finite": probability_differences_finite,
        "component_probabilities_finite": component_probabilities_finite,
        "maximum_component_probability_difference": maximum_probability_difference,
        "feature_widths_exact": widths_exact,
        "preprocessing_bitwise_exact": preprocessing_exact,
        "discrete_preprocessing_exact": discrete_exact,
        "numeric_preprocessing_shapes_exact": numeric_shapes_exact,
        "numeric_preprocessing_differences_finite": numeric_differences_finite,
        "numeric_preprocessing_maximum_difference": numeric_maximum,
        "preprocessing_tolerance": PREPROCESSING_TOLERANCE,
        "preprocessing_within_tolerance": preprocessing_within_tolerance,
        "packaged_local_assets_exact": local_assets_exact,
        "supervised_training_package_contract_exact": supervised_contract_exact,
        "replay_integrity_verified": bool(
            package_verification.get("replay_integrity_verified", False)
        ),
        "final_output_maximum_probability_difference": final_output_maximum,
        "passed": passed,
        "classification": (
            "no parity defect observed on preserved 100-row smoke sample; "
            "validation/domain shift is the leading explanation (not proof for "
            "hidden/container edges)"
            if passed
            else "a parity or evidence-integrity defect was observed in the preserved "
            "100-row smoke audit"
        ),
    }


def run_v04_forensic_audit(
    project_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    root = paths.root
    run_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else paths.experiments_dir / "runs" / _utc_run_id()
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    (root / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v04_forensic_", dir=root / "tmp") as temporary:
        temporary_root = Path(temporary)
        artifacts, package_verification = _verify_and_extract_package(root, temporary_root)
        module = import_packaged_main(artifacts.package_main)
        features, sample_positions = _sample_frame(root, artifacts.smoke_features)
        artifact = joblib.load(artifacts.model)

        transcript_verification = _transcript_hash_audit(
            root, features, artifacts.smoke_transcripts
        )
        cached = _cached_preprocessing(root, sample_positions, features)
        runtime = _runtime_preprocessing(module, features, artifacts.smoke_transcripts)
        preprocessing_rows: list[dict[str, object]] = []
        for field in (
            "full_text",
            "student_text",
            "tutor_text",
            "opening_text",
            "closing_student_text",
            "closing_tutor_text",
            "objective_context",
            "feedback_evidence",
        ):
            preprocessing_rows.append(_text_check(field, cached[field], runtime[field]))
        preprocessing_rows.extend(
            [
                _matrix_check("behavior_raw", cached["behavior_raw"], runtime["behavior_raw"]),
                _matrix_check("retrieval_raw", cached["retrieval_raw"], runtime["retrieval_raw"]),
                _matrix_check("ordered_features", cached["ordered"], runtime["ordered"]),
            ]
        )

        offline_matrices = _offline_component_features(
            root, sample_positions, features, artifact
        )
        runtime_matrices, runtime_probabilities = _runtime_component_features(
            module,
            features,
            artifacts.smoke_transcripts,
            root / "assets" / "pretrained" / "bge-base-en-v1.5",
            artifact,
        )

        offline_contexts = cached["objective_context"]
        (
            offline_tokens,
            runtime_tokens,
            supervised_offline_probability,
            supervised_runtime_probability,
            supervised_runtime_contexts,
            supervised_contract,
        ) = _supervised_parity(
            module,
            root,
            features,
            artifacts.smoke_transcripts,
            offline_contexts,
            root / "assets" / "pretrained" / "bge-small-en-v1.5",
            artifacts.delta,
            artifact,
        )
        preprocessing_rows.append(
            _text_check(
                "supervised_objective_context",
                offline_contexts,
                supervised_runtime_contexts,
            )
        )
        preprocessing_rows.append(
            _matrix_check("supervised_input_ids", offline_tokens, runtime_tokens)
        )
        for component in (
            "full_transcript",
            "role_objective_dense",
            "bge_base_semantic",
            "nbsvm_feedback",
        ):
            preprocessing_rows.append(
                _matrix_check(
                    f"{component}_model_matrix",
                    offline_matrices[component],
                    runtime_matrices[component],
                )
            )

        component_rows: list[dict[str, object]] = []
        offline_component_probabilities: dict[str, np.ndarray] = {}
        model_keys = {
            "full_transcript": "full_model",
            "role_objective_dense": "role_model",
            "bge_base_semantic": "semantic_model",
            "nbsvm_feedback": "nbsvm_model",
        }
        for component, model_key in model_keys.items():
            offline_probability = artifact[model_key].predict_proba(
                offline_matrices[component]
            )[:, 1]
            offline_component_probabilities[component] = np.asarray(
                offline_probability, dtype=np.float64
            )
            runtime_probability = runtime_probabilities[component]
            for row_index, response_id in enumerate(features["response_id"]):
                component_rows.append(
                    {
                        "response_id": str(response_id),
                        "component": component,
                        "expected_width": EXPECTED_WIDTHS[component],
                        "offline_width": int(offline_matrices[component].shape[1]),
                        "runtime_width": int(runtime_matrices[component].shape[1]),
                        "offline_probability": float(offline_probability[row_index]),
                        "runtime_probability": float(runtime_probability[row_index]),
                        "absolute_probability_difference": float(
                            abs(offline_probability[row_index] - runtime_probability[row_index])
                        ),
                    }
                )
        for row_index, response_id in enumerate(features["response_id"]):
            component_rows.append(
                {
                    "response_id": str(response_id),
                    "component": "supervised_bge_small",
                    "expected_width": EXPECTED_WIDTHS["supervised_bge_small"],
                    "offline_width": int(offline_tokens.shape[1]),
                    "runtime_width": int(runtime_tokens.shape[1]),
                    "offline_probability": float(supervised_offline_probability[row_index]),
                    "runtime_probability": float(supervised_runtime_probability[row_index]),
                    "absolute_probability_difference": float(
                        abs(
                            supervised_offline_probability[row_index]
                            - supervised_runtime_probability[row_index]
                        )
                    ),
                }
            )
        offline_component_probabilities["supervised_bge_small"] = np.asarray(
            supervised_offline_probability, dtype=np.float64
        )

        ensemble_weights = artifact["ensemble_weights"]
        offline_base = (
            float(ensemble_weights["full_transcript"])
            * offline_component_probabilities["full_transcript"]
            + float(ensemble_weights["role_objective_dense"])
            * offline_component_probabilities["role_objective_dense"]
            + float(ensemble_weights["semantic_interaction_dense"])
            * offline_component_probabilities["bge_base_semantic"]
        )
        nb_weight = float(artifact["nbsvm_component_weight"])
        offline_combined = (
            (1.0 - nb_weight) * offline_base
            + nb_weight * offline_component_probabilities["nbsvm_feedback"]
        )
        supervised_weight = float(artifact["supervised_component_weight"])
        offline_final = np.clip(
            (1.0 - supervised_weight) * offline_combined
            + supervised_weight * offline_component_probabilities["supervised_bge_small"],
            float(artifact.get("probability_clip", 1e-6)),
            1.0 - float(artifact.get("probability_clip", 1e-6)),
        )
        replay_probability = _verified_replay_probability(root, features)
        for row_index, response_id in enumerate(features["response_id"]):
            component_rows.append(
                {
                    "response_id": str(response_id),
                    "component": "final_v04_output",
                    "expected_width": EXPECTED_WIDTHS["final_v04_output"],
                    "offline_width": 1,
                    "runtime_width": 1,
                    "offline_probability": float(offline_final[row_index]),
                    "runtime_probability": float(replay_probability[row_index]),
                    "absolute_probability_difference": float(
                        abs(offline_final[row_index] - replay_probability[row_index])
                    ),
                }
            )
        component_parity = pd.DataFrame(component_rows)
        preprocessing_parity = pd.DataFrame(preprocessing_rows)

        oof = reconstruct_oof_predictions(root)
        oof_integrity = audit_oof_integrity(oof)
        calibration = calibration_metrics(oof)
        groups = group_error_diagnostics(root, oof)
        prior = prior_shift_sensitivity(oof)

        component_parity.to_csv(run_dir / "component_parity.csv", index=False)
        preprocessing_parity.to_csv(run_dir / "preprocessing_parity.csv", index=False)
        calibration.to_csv(run_dir / "calibration_metrics.csv", index=False)
        groups.to_csv(run_dir / "group_error_diagnostics.csv", index=False)
        prior.to_csv(run_dir / "prior_shift_sensitivity.csv", index=False)

        component_summary = (
            component_parity.groupby("component", sort=False)
            .agg(
                rows=("response_id", "size"),
                expected_width=("expected_width", "first"),
                offline_width=("offline_width", "first"),
                runtime_width=("runtime_width", "first"),
                max_probability_difference=("absolute_probability_difference", "max"),
                mean_probability_difference=("absolute_probability_difference", "mean"),
            )
            .reset_index()
        )
        component_summary["widths_exact"] = (
            component_summary["offline_width"].eq(component_summary["expected_width"])
            & component_summary["runtime_width"].eq(component_summary["expected_width"])
        )
        gate_a = gate_a_decision(
            component_parity,
            preprocessing_parity,
            package_verification,
            supervised_contract,
        )
        report = {
            "run_id": run_dir.name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "A_zero_training_forensic_audit",
            "sample_rows": int(len(features)),
            "sample_source": str(artifacts.smoke_features.relative_to(root)),
            "package_runtime_import": "exact main.py extracted from final_ensemble_v04_rank.zip",
            "package_verification": package_verification,
            "transcript_verification": transcript_verification,
            "supervised_training_package_contract": supervised_contract,
            "component_summary": component_summary.to_dict(orient="records"),
            "preprocessing_checks": preprocessing_parity.to_dict(orient="records"),
            "oof_integrity": oof_integrity,
            "oof_fidelity_note": (
                "No exact OOF prediction exists for the final full-data v0.4 refit. "
                "The highest-fidelity reconstruction uses fold-local full-scale supervised "
                "predictions only on folds 0 and 3; the five-fold reconstruction uses 4096-row "
                "supervised proxies. Every output row carries a fidelity label."
            ),
            "diagnostic_rows": {
                "calibration_metrics": int(len(calibration)),
                "group_error_diagnostics": int(len(groups)),
                "prior_shift_sensitivity": int(len(prior)),
            },
            "gate_a": gate_a,
            "outputs": [
                "component_parity.csv",
                "preprocessing_parity.csv",
                "calibration_metrics.csv",
                "group_error_diagnostics.csv",
                "prior_shift_sensitivity.csv",
                "report.json",
            ],
        }
        (run_dir / "report.json").write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            )
            + "\n",
            encoding="utf-8",
        )
    return report
