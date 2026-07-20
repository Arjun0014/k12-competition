from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import optimize, sparse
from scipy.special import expit, logsumexp
from sklearn.preprocessing import StandardScaler

from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.environment_component_validation import (
    CONFIRMATION_ENVIRONMENT,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEVELOPMENT_ENVIRONMENTS,
    EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
    PROBABILITY_CLIP,
    _align_environment,
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
    _binary_log_loss_rows,
    _bootstrap_row,
    _canonical_sha256,
    _json_safe,
    _shared_cluster_bootstrap,
    gate_c_decision,
    modeling_fingerprint,
    ordered_frame_sha256,
    paired_bootstrap_summary,
    prediction_input_sha256,
    prediction_metric_tables,
)
from trace_ace.feedback_hash_validation import (
    HASH_SCHEMA_TAG,
    WORD_FEATURES,
)
from trace_ace.foundation import sha256_file
from trace_ace.io import discover_project_paths
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
from trace_ace.retrieval_hard_validation import RETRIEVAL_DENSE_COLUMNS
from trace_ace.role_hard_validation import _response_rows, prepare_behavior_features
from trace_ace.validation_environments import (
    load_development_selection_assignments,
    load_joint_confirmation_assignments,
    purged_fold_masks,
)


D100_NAME = "D100_source_robust_joint_probe"
D100_SCHEMA_VERSION = "2026-07-20-d100-v1"
D100_FEATURE_WIDTH = 132_694
D100_MIN_GROUP_SESSIONS = 128
D100_TAU = 0.10
D100_CALIBRATION_PENALTY = 2.0
D100_L2_OBJECTIVE_WEIGHT = 0.5e-4
D100_BLEND_WEIGHT = 0.50
D100_MAX_RSS_BYTES = 8 * 1024**3
D100_FIRST_FOLD_TIMEOUT_SECONDS = 15 * 60
D100_PHASE_C_RUN_ID = "20260720T075633Z_environment_component_validation"
D100_BOOTSTRAP_SEED = 20260720
EXPECTED_D100_RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.5.1",
    "pandas": "2.3.3",
    "scipy": "1.18.0",
    "sklearn": "1.8.0",
}

D100_SPEC: dict[str, object] = {
    "schema_version": D100_SCHEMA_VERSION,
    "candidate": D100_NAME,
    "frozen_before_phase_c_results": True,
    "features": {
        "bge_base_interaction": {
            "width": 1536,
            "formula": "concat(6.0*(context*objective),0.7*abs(context-objective))/sqrt(2)",
            "row_normalization": False,
        },
        "ordered_feedback_word_hash": {
            "width": 131072,
            "ngram_range": [1, 2],
            "l2_row_normalized": True,
            "scale": "1/sqrt(2)",
        },
        "behavior_retrieval_cosine": {
            "width": 35,
            "standardization": "outer_training_only",
            "scale": 0.08,
        },
        "ordered_e220": {
            "width": 51,
            "standardization": "outer_training_only",
            "scale": 0.12,
        },
        "total_width": D100_FEATURE_WIDTH,
        "excluded": [
            "raw_objective_id",
            "target_prior",
            "provider_or_style_id",
            "session_multiplicity",
            "nb_log_count_ratio",
            "sibling_artifact",
            "test_batch_statistic",
        ],
    },
    "risk_groups": {
        "axes": ["semantic_family", "style_cell"],
        "minimum_outer_training_sessions": D100_MIN_GROUP_SESSIONS,
        "small_group_action": "merge_into_axis_rare",
        "inference_features": False,
    },
    "objective": {
        "group_risk": "mean_bce + 2*mean(prediction-target)^2",
        "tau": D100_TAU,
        "formula": (
            "0.50*global_mean_bce + 0.25*smooth_max_family_risk + "
            "0.25*smooth_max_style_risk + 0.5e-4*||w||^2"
        ),
        "sample_weights": None,
    },
    "optimizer": {
        "implementation": "scipy.optimize.minimize",
        "method": "L-BFGS-B",
        "coefficient_initialization": "zeros",
        "intercept_initialization": "outer_training_logit_prior",
        "maxiter": 60,
        "maxcor": 10,
        "maxls": 20,
        "ftol": 1e-9,
        "gtol": 1e-5,
        "penalize_intercept": False,
    },
    "candidate_formula": "0.50*v02 + 0.50*robust_head",
    "probability_clip": [PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP],
    "calibration": "none",
    "selection_gate": {
        "minimum_mean_log_loss_gain": 0.0025,
        "minimum_improved_environments": 2,
        "maximum_environment_regression": 0.0005,
        "macro_auc_ece_brier_non_regression": True,
        "minimum_session_bootstrap_support": 0.90,
        "minimum_family_bootstrap_support": 0.90,
    },
    "abort": {
        "first_retained_fold_seconds": D100_FIRST_FOLD_TIMEOUT_SECONDS,
        "rss_bytes": D100_MAX_RSS_BYTES,
        "non_finite_optimization": True,
        "changed_contract": True,
    },
}
D100_SPEC_SHA256 = _canonical_sha256(D100_SPEC)
EXPECTED_D100_SPEC_SHA256 = (
    "c62b90597fadb3ed0da1d6737dfa167444163d75580eb750e3e4f3f6c47bcf96"
)


@dataclass(frozen=True)
class D100FeatureBlocks:
    fixed_sparse: sparse.csr_matrix
    controls: np.ndarray
    ordered: np.ndarray

    @property
    def total_width(self) -> int:
        return int(self.fixed_sparse.shape[1] + self.controls.shape[1] + self.ordered.shape[1])


@dataclass(frozen=True)
class AxisGroups:
    name: str
    codes: np.ndarray
    labels: tuple[str, ...]
    counts: np.ndarray


def validate_d100_spec() -> None:
    if D100_SPEC_SHA256 != EXPECTED_D100_SPEC_SHA256:
        raise ValueError("D100 specification differs from the dated preregistration.")
    if int(D100_SPEC["features"]["total_width"]) != D100_FEATURE_WIDTH:  # type: ignore[index]
        raise ValueError("D100 total feature width changed.")
    if WORD_FEATURES != 2**17 or len(ORDERED_FEATURE_NAMES) != 51:
        raise ValueError("D100 cache feature contracts changed.")


def validate_d100_runtime() -> None:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    if observed != EXPECTED_D100_RUNTIME or sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "D100 must run in the frozen competition-aligned .venv "
            f"(expected={EXPECTED_D100_RUNTIME}, observed={observed})."
        )


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _current_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if platform.system() == "Darwin" else rss * 1024
    except (ImportError, OSError):
        return 0


def _enforce_rss_limit() -> int:
    rss = _current_rss_bytes()
    if rss > D100_MAX_RSS_BYTES:
        raise MemoryError(
            f"D100 process RSS {rss} exceeds the preregistered {D100_MAX_RSS_BYTES} byte limit."
        )
    return rss


def prepare_d100_feature_blocks(
    project_root: str | Path,
    frame: pd.DataFrame,
) -> D100FeatureBlocks:
    paths = discover_project_paths(project_root)
    bge_all, bge_similarity = prepare_bge_base_features(paths.cache_dir, frame)
    if bge_all.shape != (len(frame), 3072):
        raise ValueError(f"Unexpected BGE-base feature shape: {bge_all.shape}")
    bge_interaction = np.asarray(
        bge_all[:, 1536:] / np.float32(math.sqrt(2.0)), dtype=np.float32
    )
    role_texts = pd.read_parquet(
        paths.cache_dir / "session_role_texts.parquet", columns=["session_id"]
    )
    if not role_texts["session_id"].is_unique:
        raise ValueError("D100 session-role order contains duplicate session IDs.")
    session_order = role_texts["session_id"]
    response_rows = _response_rows(frame, session_order)
    behavior, behavior_names = prepare_behavior_features(
        frame, paths.cache_dir, session_order, response_rows
    )
    contexts = pd.read_parquet(paths.cache_dir / "response_objective_context.parquet")
    if list(contexts["response_id"].astype(str)) != list(frame["response_id"].astype(str)):
        raise ValueError("D100 objective-context rows do not align with modeling rows.")
    retrieval = contexts[RETRIEVAL_DENSE_COLUMNS].to_numpy(dtype=np.float64)
    retrieval[:, [0, 1, 2, 3, 4, 8, 9]] = np.log1p(
        np.maximum(retrieval[:, [0, 1, 2, 3, 4, 8, 9]], 0.0)
    )
    controls = np.column_stack([behavior, retrieval, bge_similarity])
    control_names = behavior_names + RETRIEVAL_DENSE_COLUMNS + [
        "bge_objective_context_cosine"
    ]
    if controls.shape != (len(frame), 35) or len(control_names) != 35:
        raise ValueError("D100 requires exactly 35 behavior/retrieval controls.")

    feedback_path = (
        paths.cache_dir / f"hash_feedback_word_{WORD_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    )
    feedback_word = sparse.load_npz(feedback_path).tocsr()
    if feedback_word.shape != (len(frame), 2**17):
        raise ValueError("D100 feedback word hash has an unexpected shape.")
    feedback_word = feedback_word.astype(np.float32) * np.float32(1.0 / math.sqrt(2.0))

    ordered_frame = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    if list(ordered_frame["response_id"].astype(str)) != list(
        frame["response_id"].astype(str)
    ):
        raise ValueError("D100 ordered-feature rows do not align with modeling rows.")
    ordered = ordered_frame[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    if ordered.shape != (len(frame), 51):
        raise ValueError("D100 requires exactly 51 ordered E220 features.")
    if not np.isfinite(bge_interaction).all() or not np.isfinite(controls).all() or not np.isfinite(ordered).all():
        raise ValueError("D100 feature blocks contain non-finite values.")

    fixed_sparse = sparse.hstack(
        [sparse.csr_matrix(bge_interaction), feedback_word],
        format="csr",
        dtype=np.float32,
    )
    blocks = D100FeatureBlocks(
        fixed_sparse=fixed_sparse,
        controls=controls,
        ordered=ordered,
    )
    if blocks.total_width != D100_FEATURE_WIDTH:
        raise ValueError(
            f"D100 feature width is {blocks.total_width}; expected {D100_FEATURE_WIDTH}."
        )
    _enforce_rss_limit()
    return blocks


def merge_rare_axis_groups(
    axis_values: Sequence[object],
    session_ids: Sequence[object],
    *,
    axis_name: str,
    minimum_sessions: int = D100_MIN_GROUP_SESSIONS,
) -> AxisGroups:
    frame = pd.DataFrame(
        {
            "axis": pd.Series(axis_values, dtype="string").fillna(""),
            "session": pd.Series(session_ids, dtype="string").fillna(""),
        }
    )
    if frame.empty or frame["axis"].str.len().eq(0).any() or frame["session"].str.len().eq(0).any():
        raise ValueError(f"{axis_name} grouping requires nonempty axis and session IDs.")
    counts = frame.groupby("axis", sort=True)["session"].nunique()
    retained = set(counts.index[counts.ge(minimum_sessions)].astype(str))
    labels = frame["axis"].map(
        lambda value: f"{axis_name}:{value}" if str(value) in retained else f"{axis_name}:rare"
    )
    codes, uniques = pd.factorize(labels, sort=True)
    if (codes < 0).any() or len(uniques) < 1:
        raise ValueError(f"{axis_name} grouping could not produce risk groups.")
    group_counts = np.bincount(codes, minlength=len(uniques)).astype(np.int64)
    return AxisGroups(
        name=axis_name,
        codes=codes.astype(np.int32),
        labels=tuple(str(value) for value in uniques),
        counts=group_counts,
    )


def _smooth_axis_risk_and_logit_gradient(
    losses: np.ndarray,
    residual: np.ndarray,
    probability_variance: np.ndarray,
    groups: AxisGroups,
) -> tuple[float, np.ndarray, np.ndarray]:
    counts = groups.counts.astype(np.float64)
    loss_mean = np.bincount(
        groups.codes, weights=losses, minlength=len(counts)
    ) / counts
    bias = np.bincount(
        groups.codes, weights=residual, minlength=len(counts)
    ) / counts
    risks = loss_mean + D100_CALIBRATION_PENALTY * bias**2
    soft_weights = np.exp((risks - float(risks.max())) / D100_TAU)
    soft_weights /= soft_weights.sum()
    smooth_risk = float(D100_TAU * logsumexp(risks / D100_TAU))
    per_row = (
        residual / counts[groups.codes]
        + (2.0 * D100_CALIBRATION_PENALTY)
        * bias[groups.codes]
        * probability_variance
        / counts[groups.codes]
    )
    logit_gradient = soft_weights[groups.codes] * per_row
    return smooth_risk, logit_gradient, risks


def robust_objective_and_gradient(
    parameters: np.ndarray,
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    family_groups: AxisGroups,
    style_groups: AxisGroups,
) -> tuple[float, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    coefficients = np.asarray(parameters[:-1], dtype=np.float64)
    intercept = float(parameters[-1])
    if matrix.shape != (len(target), len(coefficients)):
        raise ValueError("D100 objective feature/parameter dimensions do not align.")
    logits = np.asarray(matrix @ coefficients, dtype=np.float64).ravel() + intercept
    probability = expit(logits)
    losses = np.logaddexp(0.0, logits) - target * logits
    residual = probability - target
    probability_variance = probability * (1.0 - probability)
    family_risk, family_gradient, _ = _smooth_axis_risk_and_logit_gradient(
        losses, residual, probability_variance, family_groups
    )
    style_risk, style_gradient, _ = _smooth_axis_risk_and_logit_gradient(
        losses, residual, probability_variance, style_groups
    )
    objective = (
        0.50 * float(losses.mean())
        + 0.25 * family_risk
        + 0.25 * style_risk
        + D100_L2_OBJECTIVE_WEIGHT * float(coefficients @ coefficients)
    )
    logit_gradient = (
        0.50 * residual / len(target)
        + 0.25 * family_gradient
        + 0.25 * style_gradient
    )
    coefficient_gradient = np.asarray(matrix.T @ logit_gradient).ravel()
    coefficient_gradient += 2.0 * D100_L2_OBJECTIVE_WEIGHT * coefficients
    gradient = np.concatenate(
        [coefficient_gradient, np.asarray([logit_gradient.sum()], dtype=np.float64)]
    )
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError("D100 objective became non-finite.")
    return objective, gradient


def fit_source_robust_head(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    family_groups: AxisGroups,
    style_groups: AxisGroups,
    *,
    enforce_first_fold_timeout: bool,
) -> tuple[np.ndarray, float, dict[str, object]]:
    target = np.asarray(target, dtype=np.float64)
    if set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("D100 training target must contain both binary classes.")
    if matrix.shape[0] != len(target) or matrix.shape[1] != D100_FEATURE_WIDTH:
        raise ValueError("D100 training matrix violates its fixed shape contract.")
    prior = float(np.clip(target.mean(), PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP))
    initial = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    initial[-1] = math.log(prior / (1.0 - prior))
    started = time.perf_counter()
    evaluations = 0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        if evaluations == 1 or evaluations % 10 == 0:
            _enforce_rss_limit()
            if enforce_first_fold_timeout and (
                time.perf_counter() - started > D100_FIRST_FOLD_TIMEOUT_SECONDS
            ):
                raise TimeoutError("The first retained D100 fold exceeded 15 minutes.")
        return robust_objective_and_gradient(
            parameters,
            matrix,
            target,
            family_groups,
            style_groups,
        )

    result = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 60,
            "maxcor": 10,
            "maxls": 20,
            "ftol": 1e-9,
            "gtol": 1e-5,
        },
    )
    elapsed = time.perf_counter() - started
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise FloatingPointError("D100 optimizer returned non-finite output.")
    coefficients = np.asarray(result.x[:-1], dtype=np.float64)
    intercept = float(result.x[-1])
    diagnostics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(result.fun),
        "elapsed_seconds": float(elapsed),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "intercept": intercept,
        "rss_bytes": int(_enforce_rss_limit()),
    }
    return coefficients, intercept, diagnostics


def predict_source_robust_head(
    matrix: sparse.csr_matrix,
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    probability = expit(np.asarray(matrix @ coefficients).ravel() + float(intercept))
    if probability.shape != (matrix.shape[0],) or not np.isfinite(probability).all():
        raise RuntimeError("D100 robust-head prediction is invalid.")
    return np.asarray(probability, dtype=np.float64)


def _fold_matrices(
    blocks: D100FeatureBlocks,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, object]]:
    control_scaler = StandardScaler()
    ordered_scaler = StandardScaler()
    control_train = control_scaler.fit_transform(blocks.controls[train_mask]).astype(np.float32)
    control_validation = control_scaler.transform(blocks.controls[validation_mask]).astype(np.float32)
    ordered_train = ordered_scaler.fit_transform(blocks.ordered[train_mask]).astype(np.float32)
    ordered_validation = ordered_scaler.transform(blocks.ordered[validation_mask]).astype(np.float32)
    train = sparse.hstack(
        [
            blocks.fixed_sparse[train_mask],
            sparse.csr_matrix(control_train * np.float32(0.08)),
            sparse.csr_matrix(ordered_train * np.float32(0.12)),
        ],
        format="csr",
        dtype=np.float32,
    )
    validation = sparse.hstack(
        [
            blocks.fixed_sparse[validation_mask],
            sparse.csr_matrix(control_validation * np.float32(0.08)),
            sparse.csr_matrix(ordered_validation * np.float32(0.12)),
        ],
        format="csr",
        dtype=np.float32,
    )
    if train.shape[1] != D100_FEATURE_WIDTH or validation.shape[1] != D100_FEATURE_WIDTH:
        raise ValueError("D100 fold matrix width changed.")
    scaler_hashes = {
        "control_mean_sha256": _array_sha256(control_scaler.mean_),
        "control_scale_sha256": _array_sha256(control_scaler.scale_),
        "ordered_mean_sha256": _array_sha256(ordered_scaler.mean_),
        "ordered_scale_sha256": _array_sha256(ordered_scaler.scale_),
    }
    _enforce_rss_limit()
    return train, validation, scaler_hashes


def _baseline_component_paths(phase_c_run_dir: Path, environment: str) -> tuple[Path, Path]:
    return (
        phase_c_run_dir / f"component_oof_{environment}.parquet",
        phase_c_run_dir / "checkpoints" / f"{environment}.components.json",
    )


def load_v02_baseline(
    phase_c_run_dir: Path,
    frame: pd.DataFrame,
    environment: str,
) -> pd.DataFrame:
    parquet_path, metadata_path = _baseline_component_paths(phase_c_run_dir, environment)
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Phase C baseline artifacts for {environment}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("environment") != environment:
        raise ValueError(f"Incomplete Phase C component checkpoint for {environment}.")
    if metadata.get("parquet_sha256") != sha256_file(parquet_path):
        raise ValueError(f"Stale Phase C component checkpoint for {environment}.")
    components = pd.read_parquet(parquet_path)
    required = {"response_id", "target", "pred_full", "pred_role", "pred_bge_small"}
    missing = sorted(required.difference(components.columns))
    if missing:
        raise ValueError(f"Phase C baseline is missing columns: {missing}")
    if list(components["response_id"].astype(str)) != list(frame["response_id"].astype(str)):
        raise ValueError(f"Phase C baseline row order changed for {environment}.")
    target = frame["target"].to_numpy(dtype=np.int8)
    if not np.array_equal(components["target"].to_numpy(dtype=np.int8), target):
        raise ValueError(f"Phase C baseline targets changed for {environment}.")
    prediction = (
        0.25 * components["pred_full"].to_numpy(dtype=np.float64)
        + 0.25 * components["pred_role"].to_numpy(dtype=np.float64)
        + 0.50 * components["pred_bge_small"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"Phase C baseline contains invalid values for {environment}.")
    return pd.DataFrame(
        {
            "response_id": frame["response_id"].astype(str),
            "pred_v02": np.clip(prediction, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP),
        }
    )


def _fold_checkpoint_paths(run_dir: Path, environment: str, fold: int) -> tuple[Path, Path]:
    stem = run_dir / "checkpoints" / f"{environment}.d100.fold_{fold}"
    return Path(f"{stem}.parquet"), Path(f"{stem}.json")


def _checkpoint_response_sha(frame: pd.DataFrame) -> str:
    return ordered_frame_sha256(frame, ["response_id", "pred_robust_head", "pred_d100"])


def _load_fold_checkpoint(
    run_dir: Path,
    environment: str,
    fold: int,
    validation_ids: Sequence[str],
    *,
    contract_sha256: str,
) -> pd.DataFrame | None:
    parquet_path, metadata_path = _fold_checkpoint_paths(run_dir, environment, fold)
    if not parquet_path.exists() and not metadata_path.exists():
        return None
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Partial D100 checkpoint exists for {environment} fold {fold}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": D100_SCHEMA_VERSION,
        "candidate": D100_NAME,
        "environment": environment,
        "fold": fold,
        "contract_sha256": contract_sha256,
        "spec_sha256": D100_SPEC_SHA256,
        "parquet_sha256": sha256_file(parquet_path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Stale D100 checkpoint {environment}/{fold}: {key}")
    result = pd.read_parquet(parquet_path)
    if list(result["response_id"].astype(str)) != list(map(str, validation_ids)):
        raise ValueError(f"D100 checkpoint row order changed for {environment}/{fold}.")
    if metadata.get("prediction_sha256") != _checkpoint_response_sha(result):
        raise ValueError(f"D100 checkpoint prediction hash changed for {environment}/{fold}.")
    return result


def _write_fold_checkpoint(
    run_dir: Path,
    environment: str,
    fold: int,
    result: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    parquet_path, metadata_path = _fold_checkpoint_paths(run_dir, environment, fold)
    if parquet_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite D100 checkpoint {environment}/{fold}.")
    _atomic_parquet(parquet_path, result)
    payload = {
        **dict(metadata),
        "schema_version": D100_SCHEMA_VERSION,
        "candidate": D100_NAME,
        "environment": environment,
        "fold": int(fold),
        "spec_sha256": D100_SPEC_SHA256,
        "rows": int(len(result)),
        "prediction_sha256": _checkpoint_response_sha(result),
        "parquet_sha256": sha256_file(parquet_path),
        "status": "complete",
    }
    _atomic_json(metadata_path, _json_safe(payload))


def fit_d100_environment(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    blocks: D100FeatureBlocks,
    baseline: pd.DataFrame,
    *,
    environment: str,
    run_dir: Path,
    contract_sha256: str,
    enforce_first_fold_timeout: bool,
) -> pd.DataFrame:
    aligned = _align_environment(frame, assignments, environment)
    if list(baseline["response_id"].astype(str)) != list(frame["response_id"].astype(str)):
        raise ValueError("D100 baseline does not align with modeling rows.")
    target = frame["target"].to_numpy(dtype=np.float64)
    robust = np.full(len(frame), np.nan, dtype=np.float64)
    candidate = np.full(len(frame), np.nan, dtype=np.float64)
    first_new_fold = True
    for fold in range(5):
        train_mask, validation_mask = purged_fold_masks(frame, aligned, fold)
        validation_ids = frame.loc[validation_mask, "response_id"].astype(str).tolist()
        checkpoint = _load_fold_checkpoint(
            run_dir,
            environment,
            fold,
            validation_ids,
            contract_sha256=contract_sha256,
        )
        if checkpoint is not None:
            robust[validation_mask] = checkpoint["pred_robust_head"].to_numpy(dtype=np.float64)
            candidate[validation_mask] = checkpoint["pred_d100"].to_numpy(dtype=np.float64)
            continue

        print(f"D100 {environment} fold {fold}/4", flush=True)
        family_groups = merge_rare_axis_groups(
            aligned.loc[train_mask, "semantic_family"],
            frame.loc[train_mask, "session_id"],
            axis_name="semantic_family",
        )
        style_groups = merge_rare_axis_groups(
            aligned.loc[train_mask, "style_cell"],
            frame.loc[train_mask, "session_id"],
            axis_name="style_cell",
        )
        train_matrix, validation_matrix, scaler_hashes = _fold_matrices(
            blocks, train_mask, validation_mask
        )
        coefficients, intercept, diagnostics = fit_source_robust_head(
            train_matrix,
            target[train_mask],
            family_groups,
            style_groups,
            enforce_first_fold_timeout=enforce_first_fold_timeout and first_new_fold,
        )
        first_new_fold = False
        robust_fold = predict_source_robust_head(validation_matrix, coefficients, intercept)
        baseline_fold = baseline.loc[validation_mask, "pred_v02"].to_numpy(dtype=np.float64)
        candidate_fold = np.clip(
            (1.0 - D100_BLEND_WEIGHT) * baseline_fold + D100_BLEND_WEIGHT * robust_fold,
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        output = pd.DataFrame(
            {
                "response_id": validation_ids,
                "pred_robust_head": robust_fold,
                "pred_d100": candidate_fold,
            }
        )
        _write_fold_checkpoint(
            run_dir,
            environment,
            fold,
            output,
            {
                "contract_sha256": contract_sha256,
                "training_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "training_sessions": int(frame.loc[train_mask, "session_id"].nunique()),
                "validation_sessions": int(frame.loc[validation_mask, "session_id"].nunique()),
                "family_risk_groups": list(family_groups.labels),
                "family_group_rows": family_groups.counts.tolist(),
                "style_risk_groups": list(style_groups.labels),
                "style_group_rows": style_groups.counts.tolist(),
                "scaler_hashes": scaler_hashes,
                "optimizer": diagnostics,
                "coefficient_sha256": _array_sha256(coefficients),
            },
        )
        robust[validation_mask] = robust_fold
        candidate[validation_mask] = candidate_fold
        del train_matrix, validation_matrix, coefficients

    if not np.isfinite(robust).all() or not np.isfinite(candidate).all():
        raise RuntimeError(f"D100 produced incomplete predictions for {environment}.")
    result = pd.DataFrame(
        {
            "environment": environment,
            "response_id": frame["response_id"].astype(str),
            "session_id": frame["session_id"].astype(str),
            "semantic_family": aligned["semantic_family"].to_numpy(dtype=np.int32),
            "style_cell": aligned["style_cell"].to_numpy(dtype=np.int32),
            "fold": aligned["fold"].to_numpy(dtype=np.int8),
            "evaluation_eligible": aligned["evaluation_eligible"].astype(bool).to_numpy(),
            "target": frame["target"].to_numpy(dtype=np.int8),
            "pred_v02": baseline["pred_v02"].to_numpy(dtype=np.float64),
            "pred_robust_head": robust,
            "prediction": candidate,
            "candidate": D100_NAME,
            "calibration": "raw",
        }
    )
    return result


def subset_paired_bootstrap_summary(
    predictions: pd.DataFrame,
    *,
    environments: Sequence[str],
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = D100_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    if n_replicates < DEFAULT_BOOTSTRAP_REPLICATES:
        raise ValueError("D100 selection bootstrap requires at least 5,000 replicates.")
    frame = predictions.loc[predictions["evaluation_eligible"].astype(bool)].copy()
    if set(frame["environment"].astype(str)) != set(environments):
        raise ValueError("D100 bootstrap environments are incomplete.")
    common_ids = set.intersection(
        *[
            set(frame.loc[frame["environment"].eq(environment), "response_id"].astype(str))
            for environment in environments
        ]
    )
    if not common_ids:
        raise ValueError("D100 bootstrap environments have no common eligible rows.")
    frame = frame.loc[frame["response_id"].astype(str).isin(common_ids)].copy()
    frame = frame.sort_values(["environment", "response_id"], kind="mergesort").reset_index(drop=True)
    input_sha = ordered_frame_sha256(
        frame,
        [
            "environment",
            "response_id",
            "session_id",
            "semantic_family",
            "candidate",
            "calibration",
            "target",
            "prediction",
            "pred_v02",
        ],
    )
    gains: list[np.ndarray] = []
    reference: pd.DataFrame | None = None
    for environment in environments:
        group = frame.loc[frame["environment"].eq(environment)].sort_values(
            "response_id", kind="mergesort"
        ).reset_index(drop=True)
        if reference is None:
            reference = group
        else:
            for column in ("response_id", "session_id", "semantic_family", "target"):
                if not np.array_equal(
                    group[column].astype(str).to_numpy(),
                    reference[column].astype(str).to_numpy(),
                ):
                    raise ValueError(f"D100 bootstrap alignment changed for {column}.")
        target = group["target"].to_numpy(dtype=np.int8)
        gains.append(
            _binary_log_loss_rows(target, group["pred_v02"].to_numpy(dtype=np.float64))
            - _binary_log_loss_rows(target, group["prediction"].to_numpy(dtype=np.float64))
        )
    assert reference is not None
    gain_matrix = np.vstack(gains)
    rows: list[dict[str, object]] = []
    for resampler, groups, resample_seed, equal_weight in (
        ("session", reference["session_id"].to_numpy(), seed, False),
        ("objective_family", reference["semantic_family"].to_numpy(), seed + 37, True),
    ):
        values, unique_replicates, stream_sha = _shared_cluster_bootstrap(
            gain_matrix,
            groups,
            n_replicates=n_replicates,
            seed=resample_seed,
            equal_cluster_weight=equal_weight,
        )
        for index, environment in enumerate(environments):
            rows.append(
                _bootstrap_row(
                    environment,
                    resampler,
                    values[index],
                    candidate=D100_NAME,
                    calibration="raw",
                    prediction_input_sha256_value=input_sha,
                    unique_replicates=unique_replicates,
                    shared_resample_sha256=stream_sha,
                    common_rows=len(common_ids),
                )
            )
        rows.append(
            _bootstrap_row(
                "ALL_MACRO",
                resampler,
                values.mean(axis=0),
                candidate=D100_NAME,
                calibration="raw",
                prediction_input_sha256_value=input_sha,
                unique_replicates=unique_replicates,
                shared_resample_sha256=stream_sha,
                common_rows=len(common_ids),
            )
        )
    return pd.DataFrame(rows).sort_values(["resampler", "environment"]).reset_index(drop=True)


def d100_selection_gate(
    environment_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> dict[str, object]:
    rows = environment_metrics.loc[
        environment_metrics["environment"].isin(DEVELOPMENT_ENVIRONMENTS)
        & environment_metrics["candidate"].eq(D100_NAME)
        & environment_metrics["calibration"].eq("raw")
    ].copy()
    if len(rows) != 3 or set(rows["environment"]) != set(DEVELOPMENT_ENVIRONMENTS):
        raise ValueError("D100 selection gate requires exactly three environments.")
    loss_delta = rows["delta_log_loss_vs_v02"].to_numpy(dtype=np.float64)

    def support(resampler: str) -> float:
        match = bootstrap_summary.loc[
            bootstrap_summary["environment"].eq("ALL_MACRO")
            & bootstrap_summary["resampler"].eq(resampler),
            "probability_gain_positive",
        ]
        if len(match) != 1:
            raise ValueError(f"D100 selection bootstrap is missing {resampler} support.")
        return float(match.iloc[0])

    clauses = {
        "mean_log_loss_gain_at_least_0_0025": -float(loss_delta.mean()) >= 0.0025,
        "improves_at_least_two_environments": int(np.sum(loss_delta < 0.0)) >= 2,
        "worst_environment_regression_at_most_0_0005": float(loss_delta.max()) <= 0.0005,
        "macro_auc_non_regression": float(rows["delta_roc_auc_vs_v02"].mean()) >= 0.0,
        "macro_ece_non_regression": float(rows["delta_ece_10_vs_v02"].mean()) <= 0.0,
        "macro_brier_non_regression": float(rows["delta_brier_score_vs_v02"].mean()) <= 0.0,
        "macro_session_bootstrap_90pct": support("session") >= 0.90,
        "macro_family_bootstrap_90pct": support("objective_family") >= 0.90,
    }
    return {
        "candidate": D100_NAME,
        "mean_log_loss_gain": -float(loss_delta.mean()),
        "improved_environments": int(np.sum(loss_delta < 0.0)),
        "worst_environment_delta_log_loss_vs_v02": float(loss_delta.max()),
        "mean_delta_roc_auc_vs_v02": float(rows["delta_roc_auc_vs_v02"].mean()),
        "mean_delta_ece_10_vs_v02": float(rows["delta_ece_10_vs_v02"].mean()),
        "mean_delta_brier_score_vs_v02": float(rows["delta_brier_score_vs_v02"].mean()),
        "clauses": clauses,
        "passes_selection_gate": bool(all(clauses.values())),
        "V_joint_accessed_by_d100": False,
        "V_final_accessed": False,
    }


def _source_hashes(root: Path, phase_c_run_dir: Path) -> dict[str, str]:
    cache = root / "data_cache"
    source = root / "src" / "trace_ace"
    files = {
        "modeling_base.parquet": cache / "modeling_base.parquet",
        "bge_base_context_256.npy": cache / "bge_base_context_256.npy",
        "bge_base_objective_256.npy": cache / "bge_base_objective_256.npy",
        "session_role_texts.parquet": cache / "session_role_texts.parquet",
        "session_behavior.parquet": cache / "session_behavior.parquet",
        "response_objective_context.parquet": cache / "response_objective_context.parquet",
        "response_ordered_features.parquet": cache / "response_ordered_features.parquet",
        f"hash_feedback_word_{WORD_FEATURES}_{HASH_SCHEMA_TAG}.npz": cache / f"hash_feedback_word_{WORD_FEATURES}_{HASH_SCHEMA_TAG}.npz",
        "validation_environments.manifest.json": cache / "validation_environments.manifest.json",
        "validation_environments_selection.parquet": cache / "validation_environments_selection.parquet",
        "phase_c_contract.json": phase_c_run_dir / "phase_c_contract.json",
        "source_robust_validation.py": source / "source_robust_validation.py",
        "environment_component_validation.py": source / "environment_component_validation.py",
        "validation_environments.py": source / "validation_environments.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"D100 contract inputs are missing: {missing}")
    return {name: sha256_file(path) for name, path in sorted(files.items())}


def build_d100_contract(
    project_root: str | Path,
    frame: pd.DataFrame,
    manifest: Mapping[str, object],
    phase_c_run_dir: Path,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    development = manifest.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("D100 validation manifest lacks a development contract.")
    if development.get("aggregate_assignment_sha256") != EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256:
        raise ValueError("D100 aggregate development assignment hash changed.")
    phase_c_report_path = phase_c_run_dir / "report.json"
    phase_c_report = json.loads(phase_c_report_path.read_text(encoding="utf-8"))
    if phase_c_report.get("status") != "failed_gate_c" or phase_c_report.get("V_final_accessed") is not False:
        raise ValueError("D100 requires the completed, V_final-safe Phase C failure report.")
    contract = {
        "schema_version": D100_SCHEMA_VERSION,
        "candidate": D100_NAME,
        "spec_sha256": D100_SPEC_SHA256,
        "modeling_fingerprint_sha256": modeling_fingerprint(frame),
        "row_order_sha256": ordered_frame_sha256(frame, ["response_id", "session_id", "learning_objective_id", "target"]),
        "development_assignment_sha256": development.get("aggregate_assignment_sha256"),
        "selection_assignment_sha256": development.get("selection", {}).get("assignment_sha256"),  # type: ignore[union-attr]
        "phase_c_run_id": phase_c_run_dir.name,
        "phase_c_report_sha256": sha256_file(phase_c_report_path),
        "phase_c_gate_passed": False,
        "V_joint_previously_opened_by_phase_c": True,
        "V_joint_role_for_d100": "development_evidence_only",
        "V_final_accessed": False,
        "source_hashes": _source_hashes(paths.root, phase_c_run_dir),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    return _json_safe(contract)  # type: ignore[return-value]


def _write_or_verify_contract(run_dir: Path, contract: Mapping[str, object]) -> str:
    path = run_dir / "d100_contract.json"
    expected = _canonical_sha256(contract)
    payload = {**dict(contract), "contract_sha256": expected}
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError("Existing D100 run contract differs from current inputs/code.")
    else:
        _atomic_json(path, payload)
    return expected


def _write_environment_output(run_dir: Path, environment: str, frame: pd.DataFrame) -> None:
    path = run_dir / f"d100_oof_{environment}.parquet"
    if path.exists():
        observed = pd.read_parquet(path)
        if not observed.equals(frame):
            raise ValueError(f"Existing D100 output changed for {environment}.")
        return
    _atomic_parquet(path, frame)


def _write_or_verify_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if not pd.read_parquet(path).equals(frame):
            raise ValueError(f"Existing parquet output changed: {path}")
        return
    _atomic_parquet(path, frame)


def _write_or_verify_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        observed = pd.read_csv(path)
        expected = pd.read_csv(
            io.StringIO(frame.to_csv(index=False, lineterminator="\n"))
        )
        if list(observed.columns) != list(expected.columns) or not observed.equals(expected):
            raise ValueError(f"Existing CSV output changed: {path}")
        return
    _atomic_csv(path, frame)


def _write_or_verify_json(path: Path, value: Mapping[str, object]) -> None:
    payload = _json_safe(value)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Existing JSON output changed: {path}")
        return
    _atomic_json(path, payload)


def _create_d100_lock(
    run_dir: Path,
    selection_gate: Mapping[str, object],
    selection_predictions: pd.DataFrame,
    *,
    contract_sha256: str,
) -> dict[str, object]:
    if not bool(selection_gate.get("passes_selection_gate")):
        raise ValueError("Cannot lock D100 before it passes its selection gate.")
    payload = {
        "schema_version": D100_SCHEMA_VERSION,
        "candidate": D100_NAME,
        "calibration": "raw",
        "candidate_formula": D100_SPEC["candidate_formula"],
        "spec_sha256": D100_SPEC_SHA256,
        "contract_sha256": contract_sha256,
        "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
        "selection_environments": list(DEVELOPMENT_ENVIRONMENTS),
        "confirmation_environment": CONFIRMATION_ENVIRONMENT,
        "selection_prediction_sha256": ordered_frame_sha256(
            selection_predictions,
            ["environment", "response_id", "fold", "target", "pred_v02", "prediction"],
        ),
        "selection_gate": _json_safe(selection_gate),
        "V_joint_previously_opened_by_phase_c": True,
        "V_joint_role_for_d100": "development_evidence_only",
        "V_final_accessed": False,
    }
    lock = {**payload, "lock_sha256": _canonical_sha256(payload)}
    path = run_dir / "locked_d100_candidate.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Existing D100 candidate lock differs from frozen selection.")
    else:
        _atomic_json(path, lock)
    return lock


def _new_run_dir(root: Path, run_id: str | None) -> Path:
    effective = run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_source_robust_validation"
    )
    run_dir = root / "experiments" / "runs" / effective
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_source_robust_validation(
    project_root: str | Path,
    *,
    run_id: str | None = None,
    phase_c_run_id: str = D100_PHASE_C_RUN_ID,
) -> dict[str, object]:
    validate_d100_spec()
    validate_d100_runtime()
    paths = discover_project_paths(project_root)
    run_dir = _new_run_dir(paths.root, run_id)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    selection_assignments, manifest = load_development_selection_assignments(paths.root)
    phase_c_run_dir = paths.experiments_dir / "runs" / phase_c_run_id
    contract = build_d100_contract(paths.root, frame, manifest, phase_c_run_dir)
    contract_sha = _write_or_verify_contract(run_dir, contract)
    blocks = prepare_d100_feature_blocks(paths.root, frame)

    selection_outputs: list[pd.DataFrame] = []
    for environment_index, environment in enumerate(DEVELOPMENT_ENVIRONMENTS):
        baseline = load_v02_baseline(phase_c_run_dir, frame, environment)
        output = fit_d100_environment(
            frame,
            selection_assignments,
            blocks,
            baseline,
            environment=environment,
            run_dir=run_dir,
            contract_sha256=contract_sha,
            enforce_first_fold_timeout=environment_index == 0,
        )
        _write_environment_output(run_dir, environment, output)
        selection_outputs.append(output)
    selection_predictions = pd.concat(selection_outputs, ignore_index=True)
    selection_fold_metrics, selection_environment_metrics = prediction_metric_tables(
        selection_predictions
    )
    selection_bootstrap = subset_paired_bootstrap_summary(
        selection_predictions,
        environments=DEVELOPMENT_ENVIRONMENTS,
    )
    selection_gate = d100_selection_gate(selection_environment_metrics, selection_bootstrap)
    _write_or_verify_parquet(run_dir / "selection_predictions.parquet", selection_predictions)
    _write_or_verify_csv(run_dir / "selection_fold_metrics.csv", selection_fold_metrics)
    _write_or_verify_csv(
        run_dir / "selection_environment_metrics.csv", selection_environment_metrics
    )
    _write_or_verify_csv(run_dir / "selection_bootstrap_summary.csv", selection_bootstrap)
    _write_or_verify_json(run_dir / "selection_gate.json", selection_gate)
    if not bool(selection_gate["passes_selection_gate"]):
        report = {
            "run_id": run_dir.name,
            "status": "failed_selection_gate",
            "candidate": D100_NAME,
            "spec_sha256": D100_SPEC_SHA256,
            "contract_sha256": contract_sha,
            "selection_gate": selection_gate,
            "V_joint_accessed_by_d100": False,
            "V_final_accessed": False,
        }
        _write_or_verify_json(run_dir / "report.json", report)
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True), flush=True)
        return report

    lock = _create_d100_lock(
        run_dir,
        selection_gate,
        selection_predictions,
        contract_sha256=contract_sha,
    )
    joint_assignments, _ = load_joint_confirmation_assignments(
        paths.root,
        locked_candidate_sentinel=run_dir / "locked_d100_candidate.json",
    )
    joint_baseline = load_v02_baseline(phase_c_run_dir, frame, CONFIRMATION_ENVIRONMENT)
    joint_output = fit_d100_environment(
        frame,
        joint_assignments,
        blocks,
        joint_baseline,
        environment=CONFIRMATION_ENVIRONMENT,
        run_dir=run_dir,
        contract_sha256=contract_sha,
        enforce_first_fold_timeout=False,
    )
    _write_environment_output(run_dir, CONFIRMATION_ENVIRONMENT, joint_output)
    all_predictions = pd.concat([selection_predictions, joint_output], ignore_index=True)
    prediction_sha = prediction_input_sha256(all_predictions)
    fold_metrics, environment_metrics = prediction_metric_tables(all_predictions)
    environment_metrics["prediction_input_sha256"] = prediction_sha
    environment_metrics["lock_sha256"] = lock["lock_sha256"]
    environment_metrics["phase_contract_sha256"] = contract_sha
    bootstrap = paired_bootstrap_summary(all_predictions)
    gate = gate_c_decision(
        environment_metrics,
        bootstrap,
        candidate=D100_NAME,
        calibration="raw",
        expected_prediction_input_sha256=prediction_sha,
        expected_lock_sha256=str(lock["lock_sha256"]),
        expected_phase_contract_sha256=contract_sha,
    )
    _write_or_verify_parquet(run_dir / "all_development_predictions.parquet", all_predictions)
    _write_or_verify_csv(run_dir / "fold_metrics.csv", fold_metrics)
    _write_or_verify_csv(run_dir / "environment_metrics.csv", environment_metrics)
    _write_or_verify_csv(run_dir / "bootstrap_summary.csv", bootstrap)
    _write_or_verify_json(run_dir / "gate_report.json", gate)
    report = {
        "run_id": run_dir.name,
        "status": "passed_gate_c_requires_v_final" if gate["passes_gate_c"] else "failed_gate_c",
        "candidate": D100_NAME,
        "spec_sha256": D100_SPEC_SHA256,
        "contract_sha256": contract_sha,
        "lock_sha256": lock["lock_sha256"],
        "selection_gate": selection_gate,
        "gate_report": gate,
        "V_joint_previously_opened_by_phase_c": True,
        "V_joint_role_for_d100": "development_evidence_only",
        "V_final_accessed": False,
    }
    _write_or_verify_json(run_dir / "report.json", report)
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True), flush=True)
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered D100 robust-head validation.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--phase-c-run-id", default=D100_PHASE_C_RUN_ID)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_source_robust_validation(
        args.project_root,
        run_id=args.run_id,
        phase_c_run_id=args.phase_c_run_id,
    )


if __name__ == "__main__":
    main()
