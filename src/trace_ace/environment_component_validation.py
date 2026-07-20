from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from scipy import optimize, sparse
from sklearn.model_selection import StratifiedGroupKFold

from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.feedback_hash_validation import prepare_feedback_hashes
from trace_ace.foundation import sha256_file
from trace_ace.hard_validation import _fold_masks, sparse_sgd_oof
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.nbsvm_validation import nbsvm_oof
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
from trace_ace.robust_validation import _prepare_v02_blocks
from trace_ace.role_hard_validation import sparse_dense_sgd_oof
from trace_ace.semantic_hard_validation import semantic_logistic_oof
from trace_ace.validation_environments import (
    assignment_sha256,
    load_development_selection_assignments,
    load_joint_confirmation_assignments,
    purged_fold_masks,
)


EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256 = (
    "67a9002d670500fc7b492582c9b1c718948bc2c86156ce21687951af3f085f53"
)
DEVELOPMENT_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
ALLOWED_ENVIRONMENTS = (*DEVELOPMENT_ENVIRONMENTS, CONFIRMATION_ENVIRONMENT)
COMPONENT_ORDER = ("full", "role", "bge_small", "bge_base", "feedback_sgd", "nbsvm")
BASELINE_CANDIDATE = "v02"
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6
PRIOR_MODEL_WEIGHT = 0.90
CALIBRATION_NONREGRESSION_TOLERANCE = 0.0
CALIBRATION_SANITY_TOLERANCE = 1e-12
EXPECTED_CANDIDATE_REGISTRY_SHA256 = (
    "4765cf1c67d5e6910a4ce55e03748d22e417b959a100f3f7d3e694fe72437130"
)
PLAN_AMENDMENT = {
    "date": "2026-07-20",
    "status": "frozen_before_phase_c_results",
    "formulas": [
        "v02",
        "bge_replace",
        "v02_feedback20",
        "v02_nb30",
        "bge_feedback20",
        "bge_nb30",
    ],
    "excluded": {
        "supervised_bge_small": (
            "No valid outer-fold OOF predictions exist under the new environments."
        ),
        "full_v04": (
            "Its supervised branch would be in-sample under the new environments."
        ),
    },
}
MODEL_HYPERPARAMETERS = {
    "full": {"alpha": 3e-5},
    "role": {"alpha": 3e-5, "dense_weight": 0.2},
    "bge_small": {"regularization_c": 0.1},
    "bge_base": {"regularization_c": 0.1},
    "feedback_sgd": {"alpha": 1e-4, "dense_weight": 0.12},
    "nbsvm": {"regularization_c": 1.0, "dense_weight": 0.12},
    "nested_calibration": {"inner_splits": 3, "seed": 20260720},
    "prior_shrink": {"model_weight": PRIOR_MODEL_WEIGHT},
}


# All weights were fixed before this validation redesign.  Expressing every
# candidate directly over components makes accidental post-hoc reblending hard.
CANDIDATE_REGISTRY: dict[str, dict[str, float]] = {
    "v02": {
        "full": 0.25,
        "role": 0.25,
        "bge_small": 0.50,
    },
    "bge_replace": {
        "full": 0.25,
        "role": 0.25,
        "bge_base": 0.50,
    },
    "v02_feedback20": {
        "full": 0.20,
        "role": 0.20,
        "bge_small": 0.40,
        "feedback_sgd": 0.20,
    },
    "v02_nb30": {
        "full": 0.175,
        "role": 0.175,
        "bge_small": 0.35,
        "nbsvm": 0.30,
    },
    "bge_feedback20": {
        "full": 0.20,
        "role": 0.20,
        "bge_base": 0.40,
        "feedback_sgd": 0.20,
    },
    "bge_nb30": {
        "full": 0.175,
        "role": 0.175,
        "bge_base": 0.35,
        "nbsvm": 0.30,
    },
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


CANDIDATE_REGISTRY_SHA256 = _canonical_sha256(CANDIDATE_REGISTRY)


@dataclass(frozen=True)
class ComponentBlocks:
    full: sparse.csr_matrix
    role_sparse: sparse.csr_matrix
    role_dense: np.ndarray
    bge_small: np.ndarray
    semantic_dense: np.ndarray
    bge_base: np.ndarray
    bge_base_dense: np.ndarray
    feedback_word: sparse.csr_matrix
    ordered: np.ndarray

    def slice(self, mask: np.ndarray) -> "ComponentBlocks":
        mask = np.asarray(mask, dtype=bool)
        return ComponentBlocks(
            full=self.full[mask],
            role_sparse=self.role_sparse[mask],
            role_dense=self.role_dense[mask],
            bge_small=self.bge_small[mask],
            semantic_dense=self.semantic_dense[mask],
            bge_base=self.bge_base[mask],
            bge_base_dense=self.bge_base_dense[mask],
            feedback_word=self.feedback_word[mask],
            ordered=self.ordered[mask],
        )


def validate_candidate_registry(
    registry: Mapping[str, Mapping[str, float]] = CANDIDATE_REGISTRY,
) -> None:
    allowed = {"full", "role", "bge_small", "bge_base", "feedback_sgd", "nbsvm"}
    if set(registry) != set(CANDIDATE_REGISTRY):
        raise ValueError("The fixed Phase C candidate registry was changed.")
    for name, weights in registry.items():
        unknown = sorted(set(weights).difference(allowed))
        if unknown:
            raise ValueError(f"{name} contains unknown components: {unknown}")
        values = np.asarray(list(weights.values()), dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} has invalid component weights.")
        if not np.isclose(values.sum(), 1.0, atol=1e-12):
            raise ValueError(f"{name} weights do not sum to one.")
    digest = _canonical_sha256(registry)
    if digest != EXPECTED_CANDIDATE_REGISTRY_SHA256:
        raise ValueError(
            "The Phase C candidate registry digest differs from the dated frozen amendment."
        )


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    def write(target: Path) -> None:
        target.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    _atomic_replace(path, write)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(
        path,
        lambda target: frame.to_csv(target, index=False, lineterminator="\n"),
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(path, lambda target: frame.to_parquet(target, index=False))


def validate_development_assignment_contract(
    assignments: pd.DataFrame,
    *,
    expected_sha256: str,
    expected_environments: Sequence[str] = DEVELOPMENT_ENVIRONMENTS,
) -> None:
    required = {
        "schema_version",
        "suite",
        "split_seed",
        "environment",
        "response_id",
        "session_id",
        "learning_objective_id",
        "fold",
        "semantic_family",
        "style_cell",
        "joint_cell",
        "evaluation_eligible",
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Development assignments are missing columns: {missing}")
    suites = set(assignments["suite"].astype(str))
    if suites != {"development"}:
        raise ValueError(
            "Phase C accepts suite='development' only; sealed/final assignments are forbidden."
        )
    environments = set(assignments["environment"].astype(str))
    if environments != set(expected_environments):
        raise ValueError(f"Unexpected development environments: {sorted(environments)}")
    if assignment_sha256(assignments) != expected_sha256:
        raise ValueError("Development assignment SHA-256 does not match the frozen contract.")
    for environment in expected_environments:
        subset = assignments.loc[assignments["environment"].eq(environment)]
        if subset["response_id"].duplicated().any():
            raise ValueError(f"{environment} contains duplicate response IDs.")
        if set(subset["fold"].astype(int)) != set(range(5)):
            raise ValueError(f"{environment} must contain exactly five populated folds.")


def load_development_assignments(
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compatibility wrapper that can expose only the pre-lock selection suite."""

    assignments, manifest = load_development_selection_assignments(project_root)
    development = manifest.get("development", {})
    if not isinstance(development, Mapping):
        raise ValueError("Manifest development contract is missing.")
    selection = development.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("Manifest selection contract is missing.")
    expected_sha = str(selection.get("assignment_sha256", ""))
    validate_development_assignment_contract(
        assignments,
        expected_sha256=expected_sha,
        expected_environments=DEVELOPMENT_ENVIRONMENTS,
    )
    if development.get("aggregate_assignment_sha256") != (
        EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256
    ):
        raise ValueError("Manifest aggregate assignment SHA-256 changed.")
    if bool(development.get("sealed", True)):
        raise ValueError("The development suite is unexpectedly marked sealed.")
    # The Phase B loader sanitizes all V_joint location/hash details, and never
    # resolves, stats, or reads either V_joint or V_final here.
    return assignments, manifest


def _align_environment(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    environment: str,
) -> pd.DataFrame:
    if environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(f"Phase C cannot evaluate environment {environment!r}.")
    subset = assignments.loc[assignments["environment"].eq(environment)].copy()
    if set(subset["suite"].astype(str)) != {"development"}:
        raise ValueError("Only development assignments can be aligned in Phase C.")
    if subset["response_id"].duplicated().any():
        raise ValueError(f"Duplicate response IDs in {environment}.")
    lookup = subset.set_index("response_id")
    response_ids = frame["response_id"].astype(str)
    missing = sorted(set(response_ids).difference(lookup.index.astype(str)))
    if missing:
        raise ValueError(f"{environment} is missing {len(missing)} modeling responses.")
    aligned = lookup.loc[response_ids].reset_index()
    if list(aligned["response_id"].astype(str)) != list(response_ids):
        raise RuntimeError(f"Failed to align {environment} by response_id.")
    return aligned


def prepare_component_blocks(project_root: str | Path, frame: pd.DataFrame) -> ComponentBlocks:
    paths = discover_project_paths(project_root)
    full, role_sparse, role_dense, bge_small, semantic_dense = _prepare_v02_blocks(
        frame, paths.cache_dir
    )
    bge_base, bge_base_similarity = prepare_bge_base_features(paths.cache_dir, frame)
    if semantic_dense.ndim != 2 or semantic_dense.shape[1] < 1:
        raise ValueError("Semantic dense controls have an invalid shape.")
    bge_base_dense = semantic_dense.copy()
    bge_base_dense[:, -1:] = bge_base_similarity
    feedback_word, _, _ = prepare_feedback_hashes(paths.cache_dir, frame["response_id"])
    ordered_frame = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    if list(ordered_frame["response_id"]) != list(frame["response_id"]):
        raise ValueError("Ordered feedback rows do not align with modeling rows.")
    ordered = ordered_frame[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    return ComponentBlocks(
        full=full,
        role_sparse=role_sparse,
        role_dense=role_dense,
        bge_small=bge_small,
        semantic_dense=semantic_dense,
        bge_base=bge_base,
        bge_base_dense=bge_base_dense,
        feedback_word=feedback_word,
        ordered=ordered,
    )


def _fit_components_for_folds(
    frame: pd.DataFrame,
    folds: np.ndarray,
    blocks: ComponentBlocks,
    required_components: set[str],
    *,
    n_splits: int,
) -> dict[str, np.ndarray]:
    unknown = sorted(
        required_components.difference(
            {"full", "role", "bge_small", "bge_base", "feedback_sgd", "nbsvm"}
        )
    )
    if unknown:
        raise ValueError(f"Unknown component requests: {unknown}")
    predictions: dict[str, np.ndarray] = {}
    if "full" in required_components:
        predictions["full"] = sparse_sgd_oof(
            frame, folds, blocks.full, alpha=3e-5, n_splits=n_splits
        )
    if "role" in required_components:
        predictions["role"] = sparse_dense_sgd_oof(
            frame,
            folds,
            blocks.role_sparse,
            blocks.role_dense,
            alpha=3e-5,
            n_splits=n_splits,
            dense_weight=0.2,
        )
    if "bge_small" in required_components:
        predictions["bge_small"] = semantic_logistic_oof(
            frame,
            folds,
            blocks.bge_small,
            blocks.semantic_dense,
            regularization_c=0.1,
            n_splits=n_splits,
        )
    if "bge_base" in required_components:
        predictions["bge_base"] = semantic_logistic_oof(
            frame,
            folds,
            blocks.bge_base,
            blocks.bge_base_dense,
            regularization_c=0.1,
            n_splits=n_splits,
        )
    if "feedback_sgd" in required_components:
        predictions["feedback_sgd"] = sparse_dense_sgd_oof(
            frame,
            folds,
            blocks.feedback_word,
            blocks.ordered,
            alpha=1e-4,
            n_splits=n_splits,
            dense_weight=0.12,
        )
    if "nbsvm" in required_components:
        predictions["nbsvm"] = nbsvm_oof(
            frame,
            folds,
            blocks.feedback_word,
            blocks.ordered,
            regularization_c=1.0,
            dense_weight=0.12,
        )
    for name, prediction in predictions.items():
        values = np.asarray(prediction, dtype=np.float64)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise RuntimeError(f"{name} produced an invalid OOF prediction vector.")
    return predictions


def fit_environment_components(
    frame: pd.DataFrame,
    environment_assignments: pd.DataFrame,
    blocks: ComponentBlocks,
    required_components: set[str],
) -> pd.DataFrame:
    environment_values = set(environment_assignments["environment"].astype(str))
    if len(environment_values) != 1:
        raise ValueError("One environment must be supplied per component fit.")
    environment = next(iter(environment_values))
    if environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(f"Forbidden environment: {environment}")
    aligned = _align_environment(frame, environment_assignments, environment)
    folds = aligned["fold"].to_numpy(dtype=np.int8)
    if set(folds.astype(int)) != set(range(5)):
        raise ValueError(f"{environment} does not contain five folds.")
    fold_prior = np.full(len(frame), np.nan, dtype=np.float64)
    target = frame["target"].to_numpy(dtype=np.int8)
    for fold in range(5):
        expected_train, expected_validation = purged_fold_masks(
            frame, environment_assignments, fold
        )
        actual_train, actual_validation = _fold_masks(frame, folds, fold)
        if not np.array_equal(expected_train, actual_train) or not np.array_equal(
            expected_validation, actual_validation
        ):
            raise RuntimeError(f"Fold-mask mismatch in {environment} fold {fold}.")
        fold_prior[actual_validation] = float(target[actual_train].mean())
    predictions = _fit_components_for_folds(
        frame, folds, blocks, required_components, n_splits=5
    )
    result = pd.DataFrame(
        {
            "environment": environment,
            "response_id": frame["response_id"].astype(str),
            "session_id": frame["session_id"].astype(str),
            "learning_objective_id": frame["learning_objective_id"].astype(str),
            "semantic_family": aligned["semantic_family"].to_numpy(dtype=np.int32),
            "fold": folds,
            "evaluation_eligible": aligned["evaluation_eligible"].to_numpy(dtype=bool),
            "target": target,
            "fold_prior": fold_prior,
        }
    )
    for name, prediction in predictions.items():
        result[f"pred_{name}"] = prediction
    return result


def required_components(candidate: str) -> set[str]:
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"Unknown fixed candidate: {candidate}")
    return set(CANDIDATE_REGISTRY[candidate])


def candidate_probability(
    components: Mapping[str, np.ndarray],
    candidate: str,
) -> np.ndarray:
    validate_candidate_registry()
    weights = CANDIDATE_REGISTRY.get(candidate)
    if weights is None:
        raise ValueError(f"Unknown fixed candidate: {candidate}")
    missing = sorted(set(weights).difference(components))
    if missing:
        raise ValueError(f"{candidate} is missing component predictions: {missing}")
    arrays = [np.asarray(components[name], dtype=np.float64) for name in weights]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"{candidate} component shapes differ.")
    prediction = sum(weights[name] * np.asarray(components[name]) for name in weights)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{candidate} produced non-finite probabilities.")
    return np.clip(np.asarray(prediction, dtype=np.float64), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)


def build_raw_candidate_predictions(
    component_frame: pd.DataFrame,
    candidates: Sequence[str],
) -> pd.DataFrame:
    components = {
        name.removeprefix("pred_"): component_frame[name].to_numpy(dtype=np.float64)
        for name in component_frame.columns
        if name.startswith("pred_")
    }
    baseline = candidate_probability(components, BASELINE_CANDIDATE)
    metadata = component_frame[
        [
            "environment",
            "response_id",
            "session_id",
            "learning_objective_id",
            "semantic_family",
            "fold",
            "evaluation_eligible",
            "target",
            "fold_prior",
        ]
    ]
    rows: list[pd.DataFrame] = []
    for candidate in candidates:
        output = metadata.copy()
        output["candidate"] = candidate
        output["calibration"] = "raw"
        output["prediction"] = candidate_probability(components, candidate)
        output["pred_v02"] = baseline
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def prior_shrink_prediction(
    probability: np.ndarray,
    prior: np.ndarray,
    model_weight: float = 0.90,
) -> np.ndarray:
    if not 0.0 < model_weight <= 1.0:
        raise ValueError("Prior-shrink model weight must be in (0, 1].")
    probability = np.asarray(probability, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if probability.shape != prior.shape or not np.isfinite(prior).all():
        raise ValueError("Prior-shrink inputs must be aligned and finite.")
    result = model_weight * probability + (1.0 - model_weight) * prior
    return np.clip(result, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.log(clipped) - np.log1p(-clipped)


def fit_constrained_platt(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=np.float64)
    logit = _logit(probability)
    if target.shape != logit.shape or set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("Platt calibration requires aligned binary targets.")

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = parameters
        score = intercept + slope * logit
        return float(np.mean(np.logaddexp(0.0, score) - target * score))

    result = optimize.minimize(
        objective,
        x0=np.asarray([0.0, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-6.0, 6.0), (0.05, 3.0)),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Constrained Platt fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def apply_platt(probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    if not np.isfinite([intercept, slope]).all() or slope <= 0.0:
        raise ValueError("Platt parameters must be finite with a positive slope.")
    score = np.clip(intercept + slope * _logit(probability), -40.0, 40.0)
    return np.clip(1.0 / (1.0 + np.exp(-score)), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)


INNER_GROUP_COLUMN = {
    "V_seen": "session_id",
    "V_objective": "semantic_family",
    "V_style": "style_cell",
    "V_joint": "joint_cell",
}


def inner_environment_folds(
    frame: pd.DataFrame,
    aligned_assignments: pd.DataFrame,
    environment: str,
    seed: int,
    n_splits: int = 3,
) -> np.ndarray:
    if environment not in INNER_GROUP_COLUMN:
        raise ValueError(f"Unknown inner-calibration environment: {environment}")
    if len(frame) != len(aligned_assignments):
        raise ValueError("Inner calibration frame and assignments are misaligned.")
    group_column = INNER_GROUP_COLUMN[environment]
    if group_column == "session_id":
        groups = frame["session_id"].astype(str).to_numpy()
    else:
        if group_column not in aligned_assignments:
            raise ValueError(f"Inner assignments are missing {group_column}.")
        groups = aligned_assignments[group_column].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    target = frame["target"].to_numpy(dtype=np.int8)
    folds = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(splitter.split(frame, target, groups)):
        folds[validation_indices] = fold
    if (folds < 0).any() or set(folds.astype(int)) != set(range(n_splits)):
        raise RuntimeError("Nested calibration fold construction failed.")
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        train_sessions = set(frame.loc[train_mask, "session_id"].astype(str))
        validation_sessions = set(frame.loc[validation_mask, "session_id"].astype(str))
        if train_sessions.intersection(validation_sessions):
            raise RuntimeError(f"Inner session leakage in {environment} fold {fold}.")
        train_groups = set(groups[train_mask])
        validation_groups = set(groups[validation_mask])
        if train_groups.intersection(validation_groups):
            raise RuntimeError(
                f"Inner {group_column} leakage in {environment} fold {fold}."
            )
    return folds


def nested_platt_predictions(
    frame: pd.DataFrame,
    environment_assignments: pd.DataFrame,
    blocks: ComponentBlocks,
    raw_component_frame: pd.DataFrame,
    candidate: str,
    *,
    inner_splits: int = 3,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    environment_values = set(environment_assignments["environment"].astype(str))
    if len(environment_values) != 1:
        raise ValueError("Nested calibration requires exactly one environment.")
    environment = next(iter(environment_values))
    if environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(f"Forbidden nested-calibration environment: {environment}")
    calibrated = np.full(len(frame), np.nan, dtype=np.float64)
    parameters: list[dict[str, object]] = []
    for outer_fold in range(5):
        validation_mask, fold_prediction, parameter = nested_platt_outer_fold(
            frame,
            environment_assignments,
            blocks,
            raw_component_frame,
            candidate,
            outer_fold=outer_fold,
            inner_splits=inner_splits,
        )
        calibrated[validation_mask] = fold_prediction
        parameters.append(parameter)
    if not np.isfinite(calibrated).all():
        raise RuntimeError(f"Incomplete nested Platt predictions for {environment}.")
    return calibrated, parameters


def nested_platt_outer_fold(
    frame: pd.DataFrame,
    environment_assignments: pd.DataFrame,
    blocks: ComponentBlocks,
    raw_component_frame: pd.DataFrame,
    candidate: str,
    *,
    outer_fold: int,
    inner_splits: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    environment_values = set(environment_assignments["environment"].astype(str))
    if len(environment_values) != 1:
        raise ValueError("Nested calibration requires exactly one environment.")
    environment = next(iter(environment_values))
    if environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(f"Forbidden nested-calibration environment: {environment}")
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be in [0, 4].")
    raw_components = {
        name.removeprefix("pred_"): raw_component_frame[name].to_numpy(dtype=np.float64)
        for name in raw_component_frame.columns
        if name.startswith("pred_")
    }
    raw_outer = candidate_probability(raw_components, candidate)
    outer_train, outer_validation = purged_fold_masks(
        frame, environment_assignments, outer_fold
    )
    aligned = _align_environment(frame, environment_assignments, environment)
    train_frame = frame.loc[outer_train].reset_index(drop=True)
    train_blocks = blocks.slice(outer_train)
    train_assignments = aligned.loc[outer_train].reset_index(drop=True)
    inner_folds = inner_environment_folds(
        train_frame,
        train_assignments,
        environment,
        seed=20260720 + 101 * list(ALLOWED_ENVIRONMENTS).index(environment) + outer_fold,
        n_splits=inner_splits,
    )
    inner_components = _fit_components_for_folds(
        train_frame,
        inner_folds,
        train_blocks,
        required_components(candidate),
        n_splits=inner_splits,
    )
    inner_prediction = candidate_probability(inner_components, candidate)
    intercept, slope = fit_constrained_platt(
        train_frame["target"].to_numpy(dtype=np.int8), inner_prediction
    )
    prediction = apply_platt(raw_outer[outer_validation], intercept, slope)
    parameter: dict[str, object] = {
        "outer_fold": int(outer_fold),
        "inner_rows": int(len(train_frame)),
        "validation_rows": int(outer_validation.sum()),
        "inner_group_column": INNER_GROUP_COLUMN[environment],
        "intercept": intercept,
        "slope": slope,
    }
    return outer_validation, prediction, parameter


def add_selected_calibrations(
    raw_selected: pd.DataFrame,
    nested_prediction: np.ndarray,
) -> pd.DataFrame:
    raw = raw_selected.copy()
    if set(raw["calibration"].astype(str)) != {"raw"}:
        raise ValueError("Selected calibration input must contain raw predictions only.")
    prior = raw.copy()
    prior["calibration"] = "prior_90"
    prior["prediction"] = prior_shrink_prediction(
        prior["prediction"].to_numpy(dtype=np.float64),
        prior["fold_prior"].to_numpy(dtype=np.float64),
        model_weight=PRIOR_MODEL_WEIGHT,
    )
    nested = raw.copy()
    nested["calibration"] = "nested_platt"
    if len(nested_prediction) != len(nested):
        raise ValueError("Nested calibration rows do not match raw selected rows.")
    nested["prediction"] = np.asarray(nested_prediction, dtype=np.float64)
    return pd.concat([raw, prior, nested], ignore_index=True)


def calibration_slope_intercept(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    return fit_constrained_platt(target, probability)


def prediction_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "environment",
        "candidate",
        "calibration",
        "fold",
        "evaluation_eligible",
        "target",
        "prediction",
        "pred_v02",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    rows: list[dict[str, object]] = []
    group_columns = ["environment", "candidate", "calibration", "fold"]
    eligible = predictions.loc[predictions["evaluation_eligible"].astype(bool)].copy()
    for keys, group in eligible.groupby(group_columns, sort=True):
        environment, candidate, calibration, fold = keys
        target = group["target"].to_numpy(dtype=np.int8)
        probability = group["prediction"].to_numpy(dtype=np.float64)
        baseline = group["pred_v02"].to_numpy(dtype=np.float64)
        metrics = binary_metrics(target, probability)
        base_metrics = binary_metrics(target, baseline)
        intercept, slope = calibration_slope_intercept(target, probability)
        baseline_intercept, baseline_slope = calibration_slope_intercept(
            target, baseline
        )
        calibration_sanity_distance = 0.5 * (
            abs(intercept) + abs(slope - 1.0)
        )
        baseline_calibration_sanity_distance = 0.5 * (
            abs(baseline_intercept) + abs(baseline_slope - 1.0)
        )
        logits = np.abs(_logit(probability))
        rows.append(
            {
                "environment": environment,
                "candidate": candidate,
                "calibration": calibration,
                "fold": int(fold),
                "rows": int(len(group)),
                **metrics,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "calibration_sanity_distance": calibration_sanity_distance,
                "delta_calibration_sanity_vs_v02": calibration_sanity_distance
                - baseline_calibration_sanity_distance,
                "mean_absolute_logit": float(logits.mean()),
                "probability_q05": float(np.quantile(probability, 0.05)),
                "probability_q95": float(np.quantile(probability, 0.95)),
                "delta_log_loss_vs_v02": metrics["log_loss"] - base_metrics["log_loss"],
                "delta_roc_auc_vs_v02": metrics["roc_auc"] - base_metrics["roc_auc"],
                "delta_brier_score_vs_v02": metrics["brier_score"]
                - base_metrics["brier_score"],
                "delta_ece_10_vs_v02": metrics["ece_10"] - base_metrics["ece_10"],
            }
        )
    folds = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)
    metric_columns = [
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "prediction_mean",
        "calibration_intercept",
        "calibration_slope",
        "calibration_sanity_distance",
        "delta_calibration_sanity_vs_v02",
        "mean_absolute_logit",
        "probability_q05",
        "probability_q95",
        "delta_log_loss_vs_v02",
        "delta_roc_auc_vs_v02",
        "delta_brier_score_vs_v02",
        "delta_ece_10_vs_v02",
    ]
    environments = (
        folds.groupby(["environment", "candidate", "calibration"], sort=True)
        .agg(rows=("rows", "sum"), **{column: (column, "mean") for column in metric_columns})
        .reset_index()
        .sort_values(["environment", "log_loss", "roc_auc"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def development_selection_table(environment_metrics: pd.DataFrame) -> pd.DataFrame:
    subset = environment_metrics.loc[
        environment_metrics["environment"].isin(DEVELOPMENT_ENVIRONMENTS)
        & environment_metrics["calibration"].eq("raw")
        & ~environment_metrics["candidate"].eq(BASELINE_CANDIDATE)
    ].copy()
    rows: list[dict[str, object]] = []
    for candidate, group in subset.groupby("candidate", sort=True):
        if set(group["environment"]) != set(DEVELOPMENT_ENVIRONMENTS):
            raise ValueError(f"Incomplete development metrics for {candidate}.")
        loss_delta = group["delta_log_loss_vs_v02"].to_numpy(dtype=np.float64)
        result = {
            "candidate": candidate,
            "calibration": "raw",
            "mean_delta_log_loss_vs_v02": float(loss_delta.mean()),
            "worst_environment_delta_log_loss_vs_v02": float(loss_delta.max()),
            "improved_environments": int(np.sum(loss_delta < 0.0)),
            "mean_delta_roc_auc_vs_v02": float(group["delta_roc_auc_vs_v02"].mean()),
            "mean_delta_brier_score_vs_v02": float(
                group["delta_brier_score_vs_v02"].mean()
            ),
            "mean_delta_ece_10_vs_v02": float(group["delta_ece_10_vs_v02"].mean()),
            "worst_environment_delta_brier_score_vs_v02": float(
                group["delta_brier_score_vs_v02"].max()
            ),
            "worst_environment_delta_ece_10_vs_v02": float(
                group["delta_ece_10_vs_v02"].max()
            ),
        }
        result["passes_development_guard"] = bool(
            result["improved_environments"] >= 2
            and result["worst_environment_delta_log_loss_vs_v02"] <= 0.0005
            and result["mean_delta_roc_auc_vs_v02"] >= 0.0
            and result["worst_environment_delta_brier_score_vs_v02"]
            <= CALIBRATION_NONREGRESSION_TOLERANCE
            and result["worst_environment_delta_ece_10_vs_v02"]
            <= CALIBRATION_NONREGRESSION_TOLERANCE
        )
        rows.append(result)
    return pd.DataFrame(rows).sort_values(
        [
            "passes_development_guard",
            "mean_delta_log_loss_vs_v02",
            "worst_environment_delta_log_loss_vs_v02",
            "mean_delta_roc_auc_vs_v02",
        ],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def select_development_candidate(selection: pd.DataFrame) -> str | None:
    passing = selection.loc[selection["passes_development_guard"].astype(bool)]
    # A raw candidate can be over-confident yet become valid after strictly
    # nested calibration.  Prefer a raw candidate that already passes, but let
    # the best deterministic raw architecture reach the calibration stage when
    # none does.  V_joint remains closed unless a calibrated option passes.
    if not passing.empty:
        return str(passing.iloc[0]["candidate"])
    return None if selection.empty else str(selection.iloc[0]["candidate"])


def calibration_selection_table(
    environment_metrics: pd.DataFrame,
    candidates: str | Sequence[str],
) -> pd.DataFrame:
    if isinstance(candidates, str):
        candidates = [candidates]
    subset = environment_metrics.loc[
        environment_metrics["environment"].isin(DEVELOPMENT_ENVIRONMENTS)
        & environment_metrics["candidate"].isin(candidates)
    ].copy()
    rows: list[dict[str, object]] = []
    for (candidate, calibration), group in subset.groupby(
        ["candidate", "calibration"], sort=True
    ):
        if set(group["environment"]) != set(DEVELOPMENT_ENVIRONMENTS):
            raise ValueError(f"Incomplete calibration metrics for {candidate}/{calibration}.")
        loss_delta = group["delta_log_loss_vs_v02"].to_numpy(dtype=np.float64)
        result = {
            "candidate": candidate,
            "calibration": calibration,
            "mean_delta_log_loss_vs_v02": float(loss_delta.mean()),
            "worst_environment_delta_log_loss_vs_v02": float(loss_delta.max()),
            "improved_environments": int(np.sum(loss_delta < 0.0)),
            "mean_delta_roc_auc_vs_v02": float(group["delta_roc_auc_vs_v02"].mean()),
            "mean_delta_brier_score_vs_v02": float(
                group["delta_brier_score_vs_v02"].mean()
            ),
            "mean_delta_ece_10_vs_v02": float(group["delta_ece_10_vs_v02"].mean()),
            "worst_environment_delta_brier_score_vs_v02": float(
                group["delta_brier_score_vs_v02"].max()
            ),
            "worst_environment_delta_ece_10_vs_v02": float(
                group["delta_ece_10_vs_v02"].max()
            ),
        }
        result["lock_eligible"] = bool(
            candidate != BASELINE_CANDIDATE or calibration != "raw"
        )
        result["passes_development_guard"] = bool(
            result["lock_eligible"]
            and result["improved_environments"] >= 2
            and result["worst_environment_delta_log_loss_vs_v02"] <= 0.0005
            and result["mean_delta_roc_auc_vs_v02"] >= 0.0
            and result["worst_environment_delta_brier_score_vs_v02"]
            <= CALIBRATION_NONREGRESSION_TOLERANCE
            and result["worst_environment_delta_ece_10_vs_v02"]
            <= CALIBRATION_NONREGRESSION_TOLERANCE
        )
        rows.append(result)
    return pd.DataFrame(rows).sort_values(
        [
            "passes_development_guard",
            "mean_delta_log_loss_vs_v02",
            "worst_environment_delta_log_loss_vs_v02",
            "mean_delta_roc_auc_vs_v02",
        ],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def ordered_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Fingerprint frame is missing columns: {missing}")
    payload = frame.loc[:, list(columns)].to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prediction_input_sha256(frame: pd.DataFrame) -> str:
    required = [
        "environment",
        "response_id",
        "candidate",
        "calibration",
        "target",
        "prediction",
        "pred_v02",
    ]
    ordered = frame.sort_values(
        ["environment", "response_id", "candidate", "calibration"], kind="mergesort"
    ).reset_index(drop=True)
    return ordered_frame_sha256(ordered, required)


def build_deployment_calibration_rule(
    candidate: str,
    calibration: str,
    v_seen_raw: pd.DataFrame,
    full_training_target: np.ndarray,
) -> dict[str, object]:
    if set(v_seen_raw["environment"].astype(str)) != {"V_seen"}:
        raise ValueError("Deployment calibration must use V_seen OOF predictions only.")
    if set(v_seen_raw["candidate"].astype(str)) != {candidate}:
        raise ValueError("Deployment calibration candidate is misaligned.")
    if set(v_seen_raw["calibration"].astype(str)) != {"raw"}:
        raise ValueError("Deployment calibration source must be raw cross-fitted predictions.")
    if v_seen_raw["response_id"].duplicated().any():
        raise ValueError("Deployment V_seen source contains duplicate responses.")
    source_sha = prediction_input_sha256(v_seen_raw)
    target = np.asarray(full_training_target, dtype=np.int8)
    if len(target) != len(v_seen_raw):
        raise ValueError("Deployment prevalence target is misaligned.")
    if calibration == "nested_platt":
        intercept, slope = fit_constrained_platt(
            v_seen_raw["target"].to_numpy(dtype=np.int8),
            v_seen_raw["prediction"].to_numpy(dtype=np.float64),
        )
        return {
            "rule": "single_platt_fit_on_full_v_seen_cross_fitted_raw_predictions",
            "intercept": intercept,
            "slope": slope,
            "source_prediction_sha256": source_sha,
            "source_rows": int(len(v_seen_raw)),
        }
    if calibration == "prior_90":
        return {
            "rule": "fixed_prior_shrink_with_full_training_prevalence",
            "model_weight": PRIOR_MODEL_WEIGHT,
            "full_training_prevalence": float(target.mean()),
            "source_prediction_sha256": source_sha,
            "source_rows": int(len(v_seen_raw)),
        }
    if calibration == "raw":
        if candidate == BASELINE_CANDIDATE:
            raise ValueError("Raw v0.2 cannot be locked.")
        return {
            "rule": "no_probability_calibration",
            "source_prediction_sha256": source_sha,
            "source_rows": int(len(v_seen_raw)),
        }
    raise ValueError(f"Unsupported deployment calibration: {calibration}")


def create_locked_candidate(
    candidate: str,
    calibration: str,
    selection_metrics: Mapping[str, object],
    nested_parameters: Mapping[str, list[dict[str, object]]],
    *,
    deployment_rule: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    validate_candidate_registry()
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError("A fixed candidate must be locked.")
    if calibration not in {"raw", "prior_90", "nested_platt"}:
        raise ValueError(f"Unsupported locked calibration: {calibration}")
    if candidate == BASELINE_CANDIDATE and calibration == "raw":
        raise ValueError("Raw v0.2 is the anchor and cannot be locked.")
    required_bindings = {
        "phase_contract_sha256",
        "development_oof_sha256",
        "code_sha256",
        "modeling_fingerprint_sha256",
        "feature_cache_hashes",
        "selection_assignment_sha256",
        "protocol_sha256",
    }
    missing_bindings = sorted(required_bindings.difference(bindings))
    if missing_bindings:
        raise ValueError(f"Locked candidate bindings are incomplete: {missing_bindings}")
    if selection_metrics.get("candidate") not in {None, candidate}:
        raise ValueError("Selection metrics reference a different candidate.")
    if selection_metrics.get("calibration") not in {None, calibration}:
        raise ValueError("Selection metrics reference a different calibration.")
    if not bool(selection_metrics.get("passes_development_guard", False)):
        raise ValueError("A candidate cannot be locked before passing the development guard.")
    if calibration == "nested_platt":
        if set(nested_parameters) != set(DEVELOPMENT_ENVIRONMENTS) or any(
            len(nested_parameters[environment]) != 5
            for environment in DEVELOPMENT_ENVIRONMENTS
        ):
            raise ValueError(
                "Nested lock must bind five outer-fold fits in every selection environment."
            )
    payload: dict[str, object] = {
        "schema_version": "2026-07-20-phase-c-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "calibration": calibration,
        "component_weights": CANDIDATE_REGISTRY[candidate],
        "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
        "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
        "plan_amendment": PLAN_AMENDMENT,
        "selection_environments": list(DEVELOPMENT_ENVIRONMENTS),
        "confirmation_environment": CONFIRMATION_ENVIRONMENT,
        "selection_metrics": _json_safe(dict(selection_metrics)),
        "nested_calibration_parameters": _json_safe(dict(nested_parameters)),
        "deployment_calibration": _json_safe(dict(deployment_rule)),
        "bindings": _json_safe(dict(bindings)),
        "V_final_accessed": False,
    }
    payload["lock_sha256"] = _canonical_sha256(payload)
    return payload


def verify_locked_candidate(
    lock: Mapping[str, object],
    *,
    contract: Mapping[str, object] | None = None,
) -> None:
    validate_candidate_registry()
    payload = dict(lock)
    observed = payload.pop("lock_sha256", None)
    if observed != _canonical_sha256(payload):
        raise ValueError("Locked candidate hash verification failed.")
    if payload.get("development_assignment_sha256") != EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256:
        raise ValueError("Locked candidate references the wrong development assignments.")
    if payload.get("candidate_registry_sha256") != CANDIDATE_REGISTRY_SHA256:
        raise ValueError("Locked candidate references a changed candidate registry.")
    if payload.get("candidate_registry_sha256") != EXPECTED_CANDIDATE_REGISTRY_SHA256:
        raise ValueError("Locked candidate does not reference the literal frozen registry hash.")
    if payload.get("V_final_accessed") is not False:
        raise ValueError("Locked candidate indicates forbidden V_final access.")
    candidate = str(payload.get("candidate"))
    calibration = str(payload.get("calibration"))
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError("Locked candidate is not in the frozen registry.")
    if payload.get("component_weights") != CANDIDATE_REGISTRY[candidate]:
        raise ValueError("Locked component weights differ from the frozen formula.")
    if calibration not in {"raw", "prior_90", "nested_platt"}:
        raise ValueError("Locked calibration is unsupported.")
    if candidate == BASELINE_CANDIDATE and calibration == "raw":
        raise ValueError("Raw v0.2 cannot be a locked candidate.")
    if not isinstance(payload.get("bindings"), Mapping) or not isinstance(
        payload.get("deployment_calibration"), dict
    ):
        raise ValueError("Locked candidate lacks immutable deployment bindings.")
    selection_metrics = payload.get("selection_metrics")
    if not isinstance(selection_metrics, Mapping) or (
        selection_metrics.get("passes_development_guard") is not True
    ):
        raise ValueError("Locked candidate did not pass the frozen development guard.")
    deployment = payload["deployment_calibration"]
    assert isinstance(deployment, dict)
    source_hash = str(deployment.get("source_prediction_sha256", ""))
    if len(source_hash) != 64 or int(deployment.get("source_rows", 0)) <= 0:
        raise ValueError("Deployment calibration lacks a valid V_seen source hash.")
    if calibration == "nested_platt":
        if deployment.get("rule") != (
            "single_platt_fit_on_full_v_seen_cross_fitted_raw_predictions"
        ):
            raise ValueError("Locked nested deployment rule changed.")
        nested_parameters = payload.get("nested_calibration_parameters")
        if not isinstance(nested_parameters, Mapping) or set(
            nested_parameters
        ) != set(DEVELOPMENT_ENVIRONMENTS) or any(
            not isinstance(nested_parameters[environment], list)
            or len(nested_parameters[environment]) != 5
            for environment in DEVELOPMENT_ENVIRONMENTS
        ):
            raise ValueError("Locked nested calibration parameters are incomplete.")
        values = [deployment.get("intercept"), deployment.get("slope")]
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("Deployment Platt parameters are non-finite.")
        if float(deployment["slope"]) <= 0.0:
            raise ValueError("Deployment Platt slope must be positive.")
    if calibration == "prior_90":
        if deployment.get("rule") != "fixed_prior_shrink_with_full_training_prevalence":
            raise ValueError("Locked prior-shrink deployment rule changed.")
        if not np.isclose(float(deployment.get("model_weight", np.nan)), PRIOR_MODEL_WEIGHT):
            raise ValueError("Locked prior-shrink weight changed.")
        prevalence = float(deployment.get("full_training_prevalence", np.nan))
        if not np.isfinite(prevalence) or not 0.0 < prevalence < 1.0:
            raise ValueError("Locked full-training prevalence is invalid.")
    if calibration == "raw" and deployment.get("rule") != "no_probability_calibration":
        raise ValueError("Locked raw deployment rule changed.")
    if contract is not None:
        bindings = payload["bindings"]
        assert isinstance(bindings, Mapping)
        expected = {
            "phase_contract_sha256": contract.get("phase_contract_sha256"),
            "code_sha256": contract.get("code_sha256"),
            "modeling_fingerprint_sha256": contract.get(
                "modeling_fingerprint_sha256"
            ),
            "feature_cache_hashes": contract.get("feature_cache_hashes"),
            "selection_assignment_sha256": contract.get(
                "selection_assignment_sha256"
            ),
            "protocol_sha256": contract.get("protocol_sha256"),
        }
        for key, expected_value in expected.items():
            if bindings.get(key) != expected_value:
                raise ValueError(f"Locked candidate has a stale {key} binding.")


def _binary_log_loss_rows(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP
    )
    return -(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))


def _shared_cluster_bootstrap(
    gain: np.ndarray,
    groups: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
    equal_cluster_weight: bool,
) -> tuple[np.ndarray, int, str]:
    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim != 2 or not np.isfinite(gain).all():
        raise ValueError("Shared bootstrap gains must be a finite environment-by-row matrix.")
    groups = np.asarray(groups).astype(str)
    if gain.shape[1] != len(groups):
        raise ValueError("Shared bootstrap groups are misaligned.")
    codes, unique_groups = pd.factorize(groups, sort=True)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("Shared bootstrap requires at least two clusters.")
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    sums = np.vstack(
        [
            np.bincount(codes, weights=row, minlength=n_groups).astype(np.float64)
            for row in gain
        ]
    )
    means = sums / counts[None, :]
    rng = np.random.default_rng(seed)
    output = np.empty((gain.shape[0], n_replicates), dtype=np.float64)
    unique_draws: set[bytes] = set()
    draw_stream = hashlib.sha256()
    chunk = 25
    for start in range(0, n_replicates, chunk):
        width = min(chunk, n_replicates - start)
        indices = rng.integers(0, n_groups, size=(width, n_groups))
        for row in indices:
            digest = hashlib.sha256(row.tobytes()).digest()
            unique_draws.add(digest)
            draw_stream.update(digest)
        if equal_cluster_weight:
            output[:, start : start + width] = means[:, indices].mean(axis=2)
        else:
            output[:, start : start + width] = sums[:, indices].sum(axis=2) / counts[
                indices
            ].sum(axis=1)[None, :]
    return output, len(unique_draws), draw_stream.hexdigest()


def _bootstrap_row(
    environment: str,
    resampler: str,
    values: np.ndarray,
    *,
    candidate: str,
    calibration: str,
    prediction_input_sha256_value: str,
    unique_replicates: int,
    shared_resample_sha256: str,
    common_rows: int,
) -> dict[str, object]:
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap result contains non-finite gains.")
    return {
        "environment": environment,
        "resampler": resampler,
        "candidate": candidate,
        "calibration": calibration,
        "prediction_input_sha256": prediction_input_sha256_value,
        "replicates": int(len(values)),
        "unique_replicates": int(unique_replicates),
        "shared_resample_sha256": shared_resample_sha256,
        "common_eligible_rows": int(common_rows),
        "mean_log_loss_gain": float(values.mean()),
        "probability_gain_positive": float(np.mean(values > 0.0)),
        "gain_q05": float(np.quantile(values, 0.05)),
        "gain_q50": float(np.quantile(values, 0.50)),
        "gain_q95": float(np.quantile(values, 0.95)),
    }


def paired_bootstrap_summary(
    selected_predictions: pd.DataFrame,
    *,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 20260720,
) -> pd.DataFrame:
    if n_replicates < DEFAULT_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"At least {DEFAULT_BOOTSTRAP_REPLICATES} bootstrap replicates are required."
        )
    frame = selected_predictions.loc[
        selected_predictions["evaluation_eligible"].astype(bool)
    ].copy()
    required = {
        "environment",
        "response_id",
        "session_id",
        "semantic_family",
        "candidate",
        "calibration",
        "evaluation_eligible",
        "target",
        "pred_v02",
        "prediction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Bootstrap predictions are missing columns: {missing}")
    if set(frame["environment"].astype(str)) != set(ALLOWED_ENVIRONMENTS):
        raise ValueError("Bootstrap requires exactly the four Phase C environments.")
    candidates = set(frame["candidate"].astype(str))
    calibrations = set(frame["calibration"].astype(str))
    if len(candidates) != 1 or len(calibrations) != 1:
        raise ValueError("Bootstrap input must contain one candidate/calibration decision.")
    candidate = next(iter(candidates))
    calibration = next(iter(calibrations))
    if frame.duplicated(["environment", "response_id"]).any():
        raise ValueError("Bootstrap input contains duplicate environment/response rows.")
    finite_columns = ["target", "prediction", "pred_v02"]
    if not np.isfinite(frame[finite_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Bootstrap input contains non-finite values.")
    response_sets = [
        set(frame.loc[frame["environment"].eq(environment), "response_id"].astype(str))
        for environment in ALLOWED_ENVIRONMENTS
    ]
    common_ids = set.intersection(*response_sets)
    if not common_ids:
        raise ValueError("The four environments have no shared eligible response rows.")
    frame = frame.loc[frame["response_id"].astype(str).isin(common_ids)].copy()
    frame = frame.sort_values(
        ["environment", "response_id"], kind="mergesort"
    ).reset_index(drop=True)
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
    for environment in ALLOWED_ENVIRONMENTS:
        group = (
            frame.loc[frame["environment"].eq(environment)]
            .sort_values("response_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(group) != len(common_ids):
            raise RuntimeError(f"Shared bootstrap alignment failed for {environment}.")
        if reference is None:
            reference = group
        else:
            for column in ("response_id", "session_id", "semantic_family", "target"):
                if not np.array_equal(
                    group[column].astype(str).to_numpy(),
                    reference[column].astype(str).to_numpy(),
                ):
                    raise ValueError(
                        f"Shared bootstrap {column} differs across environments."
                    )
        target = group["target"].to_numpy(dtype=np.int8)
        gains.append(
            _binary_log_loss_rows(
                target, group["pred_v02"].to_numpy(dtype=np.float64)
            )
            - _binary_log_loss_rows(
                target, group["prediction"].to_numpy(dtype=np.float64)
            )
        )
    assert reference is not None
    gain_matrix = np.vstack(gains)

    per_environment: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    specifications = (
        ("session", reference["session_id"].to_numpy(), seed, False),
        (
            "objective_family",
            reference["semantic_family"].to_numpy(),
            seed + 37,
            True,
        ),
    )
    for resampler, groups, resample_seed, equal_cluster_weight in specifications:
        values, unique_replicates, resample_sha = _shared_cluster_bootstrap(
            gain_matrix,
            groups,
            n_replicates=n_replicates,
            seed=resample_seed,
            equal_cluster_weight=equal_cluster_weight,
        )
        for environment_index, environment in enumerate(ALLOWED_ENVIRONMENTS):
            environment_values = values[environment_index]
            per_environment[(environment, resampler)] = environment_values
            rows.append(
                _bootstrap_row(
                    environment,
                    resampler,
                    environment_values,
                    candidate=candidate,
                    calibration=calibration,
                    prediction_input_sha256_value=input_sha,
                    unique_replicates=unique_replicates,
                    shared_resample_sha256=resample_sha,
                    common_rows=len(common_ids),
                )
            )
        macro = np.mean(
            np.vstack(
                [per_environment[(environment, resampler)] for environment in ALLOWED_ENVIRONMENTS]
            ),
            axis=0,
        )
        rows.append(
            _bootstrap_row(
                "ALL_MACRO",
                resampler,
                macro,
                candidate=candidate,
                calibration=calibration,
                prediction_input_sha256_value=input_sha,
                unique_replicates=unique_replicates,
                shared_resample_sha256=resample_sha,
                common_rows=len(common_ids),
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["resampler", "environment"], kind="mergesort"
    ).reset_index(drop=True)


def gate_c_decision(
    environment_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    *,
    candidate: str,
    calibration: str,
    expected_prediction_input_sha256: str,
    expected_lock_sha256: str,
    expected_phase_contract_sha256: str,
) -> dict[str, object]:
    required_metric_columns = {
        "environment",
        "candidate",
        "calibration",
        "delta_log_loss_vs_v02",
        "delta_roc_auc_vs_v02",
        "delta_brier_score_vs_v02",
        "delta_ece_10_vs_v02",
        "calibration_intercept",
        "calibration_slope",
        "calibration_sanity_distance",
        "delta_calibration_sanity_vs_v02",
        "prediction_input_sha256",
        "lock_sha256",
        "phase_contract_sha256",
    }
    missing_metrics = sorted(required_metric_columns.difference(environment_metrics.columns))
    if missing_metrics:
        raise ValueError(f"Gate C metrics are missing columns: {missing_metrics}")
    rows = environment_metrics.loc[
        environment_metrics["candidate"].eq(candidate)
        & environment_metrics["calibration"].eq(calibration)
        & environment_metrics["environment"].isin(ALLOWED_ENVIRONMENTS)
    ].copy()
    if len(rows) != 4 or rows["environment"].nunique() != 4 or (
        set(rows["environment"]) != set(ALLOWED_ENVIRONMENTS)
    ):
        raise ValueError("Gate C requires exactly four unique environment metric rows.")
    finite_metric_columns = [
        "delta_log_loss_vs_v02",
        "delta_roc_auc_vs_v02",
        "delta_brier_score_vs_v02",
        "delta_ece_10_vs_v02",
        "calibration_intercept",
        "calibration_slope",
        "calibration_sanity_distance",
        "delta_calibration_sanity_vs_v02",
    ]
    if not np.isfinite(rows[finite_metric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Gate C metric rows contain non-finite values.")
    expected_bindings = {
        "prediction_input_sha256": expected_prediction_input_sha256,
        "lock_sha256": expected_lock_sha256,
        "phase_contract_sha256": expected_phase_contract_sha256,
    }
    for column, expected in expected_bindings.items():
        if set(rows[column].astype(str)) != {expected}:
            raise ValueError(f"Gate C metrics have a stale {column} binding.")

    required_bootstrap_columns = {
        "environment",
        "resampler",
        "candidate",
        "calibration",
        "prediction_input_sha256",
        "replicates",
        "unique_replicates",
        "shared_resample_sha256",
        "probability_gain_positive",
    }
    missing_bootstrap = sorted(required_bootstrap_columns.difference(bootstrap_summary.columns))
    if missing_bootstrap:
        raise ValueError(f"Gate C bootstrap is missing columns: {missing_bootstrap}")
    expected_bootstrap_keys = {
        (environment, resampler)
        for environment in (*ALLOWED_ENVIRONMENTS, "ALL_MACRO")
        for resampler in ("session", "objective_family")
    }
    observed_bootstrap_keys = list(
        zip(
            bootstrap_summary["environment"].astype(str),
            bootstrap_summary["resampler"].astype(str),
        )
    )
    if len(observed_bootstrap_keys) != len(expected_bootstrap_keys) or (
        set(observed_bootstrap_keys) != expected_bootstrap_keys
    ):
        raise ValueError("Gate C requires ten unique bound bootstrap summaries.")
    if set(bootstrap_summary["candidate"].astype(str)) != {candidate} or set(
        bootstrap_summary["calibration"].astype(str)
    ) != {calibration}:
        raise ValueError("Gate C bootstrap candidate/calibration binding is stale.")
    if set(bootstrap_summary["prediction_input_sha256"].astype(str)) != {
        expected_prediction_input_sha256
    }:
        raise ValueError("Gate C bootstrap prediction-input binding is stale.")
    bootstrap_numeric = bootstrap_summary[
        ["replicates", "unique_replicates", "probability_gain_positive"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(bootstrap_numeric).all():
        raise ValueError("Gate C bootstrap summaries contain non-finite values.")
    if (bootstrap_summary["replicates"].astype(int) < DEFAULT_BOOTSTRAP_REPLICATES).any():
        raise ValueError("Gate C requires at least 5,000 bootstrap replicates.")
    if (
        bootstrap_summary["unique_replicates"].astype(int)
        < DEFAULT_BOOTSTRAP_REPLICATES
    ).any():
        raise ValueError("Gate C requires at least 5,000 unique bootstrap draws.")
    for resampler in ("session", "objective_family"):
        hashes = set(
            bootstrap_summary.loc[
                bootstrap_summary["resampler"].eq(resampler),
                "shared_resample_sha256",
            ].astype(str)
        )
        if len(hashes) != 1 or len(next(iter(hashes), "")) != 64:
            raise ValueError(f"{resampler} bootstrap did not use one shared resample stream.")
    loss_delta = rows["delta_log_loss_vs_v02"].to_numpy(dtype=np.float64)
    mean_loss_gain = -float(loss_delta.mean())
    joint = rows.loc[rows["environment"].eq(CONFIRMATION_ENVIRONMENT)].iloc[0]

    def bootstrap_probability(environment: str, resampler: str) -> float:
        match = bootstrap_summary.loc[
            bootstrap_summary["environment"].eq(environment)
            & bootstrap_summary["resampler"].eq(resampler),
            "probability_gain_positive",
        ]
        if len(match) != 1:
            raise ValueError(f"Missing bootstrap result for {environment}/{resampler}.")
        return float(match.iloc[0])

    clauses = {
        "mean_log_loss_gain_at_least_0_0025": mean_loss_gain >= 0.0025,
        "improves_at_least_three_environments": int(np.sum(loss_delta < 0.0)) >= 3,
        "worst_environment_regression_at_most_0_0005": float(loss_delta.max()) <= 0.0005,
        "every_environment_brier_non_regression": bool(
            (rows["delta_brier_score_vs_v02"] <= CALIBRATION_NONREGRESSION_TOLERANCE).all()
        ),
        "every_environment_ece_non_regression": bool(
            (rows["delta_ece_10_vs_v02"] <= CALIBRATION_NONREGRESSION_TOLERANCE).all()
        ),
        "macro_auc_non_regression": float(rows["delta_roc_auc_vs_v02"].mean()) >= 0.0,
        "joint_auc_non_regression": float(joint["delta_roc_auc_vs_v02"]) >= 0.0,
        "finite_positive_calibration_slope_in_every_environment": bool(
            (rows["calibration_slope"] > 0.0).all()
        ),
        "macro_session_bootstrap_90pct": bootstrap_probability("ALL_MACRO", "session")
        >= 0.90,
        "joint_session_bootstrap_90pct": bootstrap_probability(
            CONFIRMATION_ENVIRONMENT, "session"
        )
        >= 0.90,
        "macro_family_bootstrap_90pct": bootstrap_probability(
            "ALL_MACRO", "objective_family"
        )
        >= 0.90,
    }
    return {
        "candidate": candidate,
        "calibration": calibration,
        "prediction_input_sha256": expected_prediction_input_sha256,
        "lock_sha256": expected_lock_sha256,
        "phase_contract_sha256": expected_phase_contract_sha256,
        "mean_log_loss_gain": mean_loss_gain,
        "improved_environments": int(np.sum(loss_delta < 0.0)),
        "worst_environment_delta_log_loss_vs_v02": float(loss_delta.max()),
        "mean_delta_roc_auc_vs_v02": float(rows["delta_roc_auc_vs_v02"].mean()),
        "mean_delta_brier_score_vs_v02": float(rows["delta_brier_score_vs_v02"].mean()),
        "mean_delta_ece_10_vs_v02": float(rows["delta_ece_10_vs_v02"].mean()),
        "calibration_sanity_rule": (
            "diagnostic_only: equal-fold mean of "
            "0.5*(abs(intercept)+abs(slope-1)); compare with raw v0.2"
        ),
        "calibration_sanity_non_regression_tolerance": (
            CALIBRATION_SANITY_TOLERANCE
        ),
        "calibration_sanity_non_regression_in_every_environment": bool(
            (
                rows["delta_calibration_sanity_vs_v02"]
                <= CALIBRATION_SANITY_TOLERANCE
            ).all()
        ),
        "clauses": clauses,
        "passes_gate_c": bool(all(clauses.values())),
        "V_final_accessed": False,
    }


def _checkpoint_paths(run_dir: Path, environment: str) -> tuple[Path, Path]:
    return (
        run_dir / "checkpoints" / f"{environment}.components.parquet",
        run_dir / "checkpoints" / f"{environment}.components.json",
    )


def load_or_fit_environment_components(
    run_dir: Path,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    blocks: ComponentBlocks,
    environment: str,
    required: set[str],
    contract: Mapping[str, object],
) -> pd.DataFrame:
    parquet_path, metadata_path = _checkpoint_paths(run_dir, environment)
    subset = assignments.loc[assignments["environment"].eq(environment)].copy()
    subset_sha = assignment_sha256(subset)
    base_columns = [
        "environment",
        "response_id",
        "session_id",
        "learning_objective_id",
        "semantic_family",
        "fold",
        "evaluation_eligible",
        "target",
        "fold_prior",
    ]
    expected_columns = [
        *base_columns,
        *(f"pred_{name}" for name in COMPONENT_ORDER if name in required),
    ]

    def validate_cached(result: pd.DataFrame, metadata: Mapping[str, object]) -> None:
        if list(result.columns) != expected_columns:
            raise ValueError(f"Component checkpoint schema mismatch for {environment}.")
        if list(result["response_id"].astype(str)) != list(frame["response_id"].astype(str)):
            raise ValueError(f"Component checkpoint row order is stale for {environment}.")
        if not np.array_equal(
            result["target"].to_numpy(dtype=np.int8),
            frame["target"].to_numpy(dtype=np.int8),
        ):
            raise ValueError(f"Component checkpoint targets are stale for {environment}.")
        if set(result["environment"].astype(str)) != {environment}:
            raise ValueError(f"Component checkpoint environment mismatch for {environment}.")
        numeric = ["fold_prior", *(f"pred_{name}" for name in COMPONENT_ORDER if name in required)]
        if not np.isfinite(result[numeric].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Component checkpoint contains non-finite values for {environment}.")
        row_hash = ordered_frame_sha256(
            result, ["response_id", "target", *numeric]
        )
        if metadata.get("row_output_sha256") != row_hash:
            raise ValueError(f"Component checkpoint row hash mismatch for {environment}.")

    if parquet_path.exists() != metadata_path.exists():
        raise ValueError(f"Incomplete component checkpoint for {environment}.")
    if parquet_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "complete"
            and metadata.get("phase_contract_sha256")
            == contract.get("phase_contract_sha256")
            and metadata.get("modeling_fingerprint_sha256")
            == contract.get("modeling_fingerprint_sha256")
            and metadata.get("assignment_subset_sha256") == subset_sha
            and metadata.get("candidate_registry_sha256")
            == EXPECTED_CANDIDATE_REGISTRY_SHA256
            and set(metadata.get("required_components", [])) == set(required)
            and metadata.get("columns") == expected_columns
            and metadata.get("rows") == len(frame)
            and metadata.get("response_order_sha256")
            == ordered_frame_sha256(frame, ["response_id"])
            and metadata.get("parquet_sha256") == sha256_file(parquet_path)
        ):
            result = pd.read_parquet(parquet_path)
            validate_cached(result, metadata)
            output_path = run_dir / f"component_oof_{environment}.parquet"
            if not output_path.exists():
                _atomic_parquet(output_path, result)
            return result
        raise ValueError(f"Stale or incompatible checkpoint for {environment}.")
    result = fit_environment_components(frame, subset, blocks, required)
    if list(result.columns) != expected_columns:
        raise RuntimeError(f"New component output schema mismatch for {environment}.")
    _atomic_parquet(parquet_path, result)
    row_output_sha = ordered_frame_sha256(
        result,
        [
            "response_id",
            "target",
            "fold_prior",
            *(f"pred_{name}" for name in COMPONENT_ORDER if name in required),
        ],
    )
    _atomic_json(
        metadata_path,
        {
            "status": "complete",
            "environment": environment,
            "phase_contract_sha256": contract.get("phase_contract_sha256"),
            "modeling_fingerprint_sha256": contract.get("modeling_fingerprint_sha256"),
            "assignment_subset_sha256": subset_sha,
            "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
            "required_components": sorted(required),
            "columns": expected_columns,
            "rows": int(len(result)),
            "response_order_sha256": ordered_frame_sha256(result, ["response_id"]),
            "row_output_sha256": row_output_sha,
            "parquet_sha256": sha256_file(parquet_path),
        },
    )
    output_path = run_dir / f"component_oof_{environment}.parquet"
    _atomic_parquet(output_path, result)
    return result


def _calibration_checkpoint_paths(
    run_dir: Path,
    environment: str,
    candidate: str,
    outer_fold: int,
) -> tuple[Path, Path]:
    stem = f"{environment}.{candidate}.nested_platt.fold_{outer_fold}"
    return (
        run_dir / "checkpoints" / f"{stem}.parquet",
        run_dir / "checkpoints" / f"{stem}.json",
    )


def load_or_fit_nested_calibration(
    run_dir: Path,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    blocks: ComponentBlocks,
    component_frame: pd.DataFrame,
    environment: str,
    candidate: str,
    contract: Mapping[str, object],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if candidate not in CANDIDATE_REGISTRY:
        raise ValueError(f"Unknown nested-calibration candidate: {candidate}")
    subset = assignments.loc[assignments["environment"].eq(environment)].copy()
    if set(subset["environment"].astype(str)) != {environment}:
        raise ValueError(f"Missing assignment subset for {environment}.")
    subset_sha = assignment_sha256(subset)
    relevant_prediction_columns = [
        f"pred_{name}" for name in COMPONENT_ORDER if name in required_components(candidate)
    ]
    raw_input_columns = [
        "response_id",
        "target",
        "fold",
        *relevant_prediction_columns,
    ]
    raw_input_sha = ordered_frame_sha256(component_frame, raw_input_columns)
    calibrated = np.full(len(frame), np.nan, dtype=np.float64)
    parameters: list[dict[str, object]] = []
    for outer_fold in range(5):
        parquet_path, metadata_path = _calibration_checkpoint_paths(
            run_dir, environment, candidate, outer_fold
        )
        _, expected_validation = purged_fold_masks(frame, subset, outer_fold)
        expected_ids = list(frame.loc[expected_validation, "response_id"].astype(str))

        def validate_cached(
            cached: pd.DataFrame, metadata: Mapping[str, object]
        ) -> np.ndarray:
            if list(cached.columns) != ["response_id", "prediction"]:
                raise ValueError(
                    f"Nested checkpoint schema mismatch for {environment} fold {outer_fold}."
                )
            if list(cached["response_id"].astype(str)) != expected_ids:
                raise ValueError(
                    f"Nested checkpoint row order is stale for {environment} fold {outer_fold}."
                )
            values = cached["prediction"].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Nested checkpoint is non-finite for {environment} fold {outer_fold}."
                )
            output_sha = ordered_frame_sha256(
                cached, ["response_id", "prediction"]
            )
            if metadata.get("row_output_sha256") != output_sha:
                raise ValueError(
                    f"Nested checkpoint row hash mismatch for {environment} fold {outer_fold}."
                )
            return values

        if parquet_path.exists() != metadata_path.exists():
            raise ValueError(
                f"Incomplete nested checkpoint for {environment} fold {outer_fold}."
            )
        if parquet_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_metadata = {
                "status": "complete",
                "environment": environment,
                "candidate": candidate,
                "outer_fold": outer_fold,
                "phase_contract_sha256": contract.get("phase_contract_sha256"),
                "modeling_fingerprint_sha256": contract.get(
                    "modeling_fingerprint_sha256"
                ),
                "assignment_subset_sha256": subset_sha,
                "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
                "raw_component_input_sha256": raw_input_sha,
                "rows": len(expected_ids),
                "columns": ["response_id", "prediction"],
            }
            for key, expected_value in expected_metadata.items():
                if metadata.get(key) != expected_value:
                    raise ValueError(
                        f"Stale nested checkpoint {key} for {environment} fold {outer_fold}."
                    )
            if metadata.get("parquet_sha256") != sha256_file(parquet_path):
                raise ValueError(
                    f"Nested checkpoint file hash mismatch for {environment} fold {outer_fold}."
                )
            cached = pd.read_parquet(parquet_path)
            values = validate_cached(cached, metadata)
            parameter = metadata.get("parameters")
            if not isinstance(parameter, Mapping):
                raise ValueError(
                    f"Nested checkpoint parameters missing for {environment} fold {outer_fold}."
                )
        else:
            validation_mask, values, parameter = nested_platt_outer_fold(
                frame,
                subset,
                blocks,
                component_frame,
                candidate,
                outer_fold=outer_fold,
                inner_splits=int(
                    MODEL_HYPERPARAMETERS["nested_calibration"]["inner_splits"]
                ),
            )
            if not np.array_equal(validation_mask, expected_validation):
                raise RuntimeError(
                    f"Nested validation mask changed for {environment} fold {outer_fold}."
                )
            cached = pd.DataFrame(
                {
                    "response_id": frame.loc[validation_mask, "response_id"].astype(
                        str
                    ),
                    "prediction": values,
                }
            ).reset_index(drop=True)
            if list(cached["response_id"].astype(str)) != expected_ids or not np.isfinite(
                cached["prediction"].to_numpy(dtype=np.float64)
            ).all():
                raise RuntimeError(
                    f"New nested output is invalid for {environment} fold {outer_fold}."
                )
            _atomic_parquet(parquet_path, cached)
            row_output_sha = ordered_frame_sha256(
                cached, ["response_id", "prediction"]
            )
            _atomic_json(
                metadata_path,
                {
                    "status": "complete",
                    "environment": environment,
                    "candidate": candidate,
                    "outer_fold": outer_fold,
                    "phase_contract_sha256": contract.get("phase_contract_sha256"),
                    "modeling_fingerprint_sha256": contract.get(
                        "modeling_fingerprint_sha256"
                    ),
                    "assignment_subset_sha256": subset_sha,
                    "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
                    "raw_component_input_sha256": raw_input_sha,
                    "parameters": _json_safe(parameter),
                    "rows": int(len(cached)),
                    "columns": ["response_id", "prediction"],
                    "response_order_sha256": ordered_frame_sha256(
                        cached, ["response_id"]
                    ),
                    "row_output_sha256": row_output_sha,
                    "parquet_sha256": sha256_file(parquet_path),
                },
            )
        calibrated[expected_validation] = values
        parameters.append(dict(parameter))
    if not np.isfinite(calibrated).all():
        raise RuntimeError(f"Incomplete nested calibration output for {environment}.")
    combined = pd.DataFrame(
        {
            "response_id": frame["response_id"].astype(str),
            "prediction": calibrated,
        }
    )
    combined_path = run_dir / f"nested_oof_{environment}_{candidate}.parquet"
    _atomic_parquet(combined_path, combined)
    return calibrated, parameters


def _new_run_dir(
    project_root: str | Path,
    run_id: str | None,
    *,
    resume: bool,
) -> tuple[str, Path]:
    paths = discover_project_paths(project_root)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ_environment_component_validation"
        )
    if Path(run_id).name != run_id or not run_id.endswith("_environment_component_validation"):
        raise ValueError("run_id must be a plain environment_component_validation run name.")
    run_dir = paths.experiments_dir / "runs" / run_id
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing Phase C run: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _write_report(run_dir: Path, report: Mapping[str, object]) -> None:
    _atomic_json(run_dir / "report.json", dict(report))


def _feature_cache_hashes(cache_dir: Path) -> dict[str, str]:
    names = (
        "modeling_base.parquet",
        "hash_full_word_131072.npz",
        "hash_student_word_65536.npz",
        "hash_tutor_word_65536.npz",
        "hash_opening_word_65536.npz",
        "hash_closing_student_word_32768.npz",
        "hash_closing_tutor_word_32768.npz",
        "bge_small_context_256.npy",
        "bge_small_objective_256.npy",
        "bge_base_context_256.npy",
        "bge_base_objective_256.npy",
        "hash_feedback_word_131072_v2_budgeted.npz",
        "response_ordered_features.parquet",
        "response_objective_context.parquet",
        "session_behavior.parquet",
    )
    missing = [name for name in names if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Phase C feature caches are missing: {missing}")
    return {name: sha256_file(cache_dir / name) for name in names}


def modeling_fingerprint(frame: pd.DataFrame) -> str:
    required = [
        "response_id",
        "session_id",
        "learning_objective_id",
        "target",
    ]
    if frame["response_id"].duplicated().any():
        raise ValueError("Modeling frame response IDs must be unique.")
    return ordered_frame_sha256(frame, required)


def build_phase_contract(
    project_root: str | Path,
    frame: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    selection_assignment_sha256: str,
    selection_assignment_rows: int,
    n_bootstrap: int,
) -> dict[str, object]:
    if n_bootstrap < DEFAULT_BOOTSTRAP_REPLICATES:
        raise ValueError("Phase C contract requires at least 5,000 bootstrap replicates.")
    paths = discover_project_paths(project_root)
    development = manifest.get("development", {})
    if not isinstance(development, dict):
        raise ValueError("Validation manifest lacks a development contract.")
    selection = development.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("Validation manifest lacks a selection artifact contract.")
    if selection.get("assignment_sha256") != selection_assignment_sha256:
        raise ValueError("Selection assignment SHA differs from its manifest contract.")
    if int(selection.get("row_count", -1)) != int(selection_assignment_rows):
        raise ValueError("Selection assignment row count differs from its manifest contract.")
    if list(selection.get("environments", [])) != list(DEVELOPMENT_ENVIRONMENTS):
        raise ValueError("Selection manifest environments differ from the frozen protocol.")
    source_dir = Path(__file__).resolve().parent
    code_paths = {
        "environment_component_validation.py": Path(__file__).resolve(),
        "validation_environments.py": source_dir / "validation_environments.py",
        "hard_validation.py": source_dir / "hard_validation.py",
        "role_hard_validation.py": source_dir / "role_hard_validation.py",
        "semantic_hard_validation.py": source_dir / "semantic_hard_validation.py",
        "nbsvm_validation.py": source_dir / "nbsvm_validation.py",
        "feedback_hash_validation.py": source_dir / "feedback_hash_validation.py",
        "encoder_upgrade_validation.py": source_dir / "encoder_upgrade_validation.py",
        "robust_validation.py": source_dir / "robust_validation.py",
        "metrics.py": source_dir / "metrics.py",
        "ordered_features.py": source_dir / "ordered_features.py",
        "run_environment_component_validation.py": (
            paths.root / "scripts" / "run_environment_component_validation.py"
        ),
    }
    missing_code = [name for name, path in code_paths.items() if not path.exists()]
    if missing_code:
        raise FileNotFoundError(f"Phase C code files are missing: {missing_code}")
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    payload: dict[str, object] = {
        "schema_version": "2026-07-20-phase-c-contract-v2",
        "plan_amendment": PLAN_AMENDMENT,
        "code_sha256": _canonical_sha256(code_hashes),
        "code_hashes": code_hashes,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "modeling_fingerprint_sha256": modeling_fingerprint(frame),
        "modeling_rows": int(len(frame)),
        "feature_cache_hashes": _feature_cache_hashes(paths.cache_dir),
        "aggregate_assignment_sha256": development.get("aggregate_assignment_sha256"),
        "selection_assignment_sha256": selection_assignment_sha256,
        "selection_assignment_file_sha256": selection.get("file_sha256"),
        "selection_assignment_rows": int(selection_assignment_rows),
        "selection_assignment_path": selection.get("assignment_path"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "source_hashes": manifest.get("source_hashes", {}),
        "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
        "candidate_registry": CANDIDATE_REGISTRY,
        "model_hyperparameters": MODEL_HYPERPARAMETERS,
        "development_environments": list(DEVELOPMENT_ENVIRONMENTS),
        "confirmation_environment": CONFIRMATION_ENVIRONMENT,
        "bootstrap_replicates": int(n_bootstrap),
        "V_final_accessed": False,
    }
    if payload["aggregate_assignment_sha256"] != EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256:
        raise ValueError("Manifest aggregate assignment SHA differs from the frozen value.")
    payload["phase_contract_sha256"] = _canonical_sha256(payload)
    return payload


def write_or_verify_phase_contract(
    run_dir: Path,
    contract: Mapping[str, object],
    *,
    resume: bool,
) -> dict[str, object]:
    path = run_dir / "phase_c_contract.json"
    expected = dict(contract)
    observed_hash = expected.get("phase_contract_sha256")
    unhashed = dict(expected)
    unhashed.pop("phase_contract_sha256", None)
    if observed_hash != _canonical_sha256(unhashed):
        raise ValueError("New Phase C contract self-hash is invalid.")
    if path.exists():
        if not resume:
            raise FileExistsError("A new Phase C run cannot reuse an existing contract.")
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_unhashed = dict(existing)
        existing_hash = existing_unhashed.pop("phase_contract_sha256", None)
        if existing_hash != _canonical_sha256(existing_unhashed):
            raise ValueError("Stored Phase C contract self-hash is invalid.")
        if existing != expected:
            raise ValueError("Resume contract differs from the immutable Phase C contract.")
        return existing
    if resume:
        raise FileNotFoundError("Resume requested but phase_c_contract.json is missing.")
    _atomic_json(path, expected)
    return expected


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    _atomic_json(path, dict(value))


def _joint_opened_payload(
    lock: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "2026-07-20-phase-c-joint-open-v1",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock.get("lock_sha256"),
        "candidate": lock.get("candidate"),
        "calibration": lock.get("calibration"),
        "phase_contract_sha256": contract.get("phase_contract_sha256"),
        "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
        "confirmation_environment": CONFIRMATION_ENVIRONMENT,
        "V_final_accessed": False,
    }
    payload["joint_opened_sha256"] = _canonical_sha256(payload)
    return payload


def write_or_verify_joint_opened(
    run_dir: Path,
    lock: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    path = run_dir / "joint_opened.json"
    expected = _joint_opened_payload(lock, contract)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(existing)
        observed = payload.pop("joint_opened_sha256", None)
        if observed != _canonical_sha256(payload):
            raise ValueError("Joint-opened sentinel self-hash is invalid.")
        immutable_keys = (
            "schema_version",
            "lock_sha256",
            "candidate",
            "calibration",
            "phase_contract_sha256",
            "development_assignment_sha256",
            "confirmation_environment",
            "V_final_accessed",
        )
        for key in immutable_keys:
            if existing.get(key) != expected.get(key):
                raise ValueError(f"Joint-opened sentinel has a stale {key} binding.")
        return existing
    _write_new_json(path, expected)
    return expected


def write_or_verify_joint_assignment_binding(
    run_dir: Path,
    joint_opened: Mapping[str, object],
    joint_assignments: pd.DataFrame,
) -> dict[str, object]:
    path = run_dir / "joint_assignment_binding.json"
    payload: dict[str, object] = {
        "schema_version": "2026-07-20-phase-c-joint-binding-v1",
        "joint_opened_sha256": joint_opened.get("joint_opened_sha256"),
        "joint_assignment_sha256": assignment_sha256(joint_assignments),
        "joint_rows": int(len(joint_assignments)),
        "confirmation_environment": CONFIRMATION_ENVIRONMENT,
        "V_final_accessed": False,
    }
    payload["joint_binding_sha256"] = _canonical_sha256(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_payload = dict(existing)
        observed = existing_payload.pop("joint_binding_sha256", None)
        if observed != _canonical_sha256(existing_payload):
            raise ValueError("Joint assignment binding self-hash is invalid.")
        if existing != payload:
            raise ValueError("Joint assignment binding differs on resume.")
        return existing
    _write_new_json(path, payload)
    return payload


def run_development_stage(
    project_root: str | Path,
    run_dir: Path,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    blocks: ComponentBlocks,
    contract: Mapping[str, object],
) -> dict[str, object]:
    validate_candidate_registry()
    if (run_dir / "joint_opened.json").exists():
        raise RuntimeError(
            "Development scoring is forbidden after the V_joint opened sentinel exists."
        )
    lock_path = run_dir / "locked_candidate.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        verify_locked_candidate(lock, contract=contract)
        development_path = run_dir / "development_candidate_oof.parquet"
        if not development_path.exists():
            raise ValueError("Locked run is missing its bound development OOF artifact.")
        development_predictions = pd.read_parquet(development_path)
        observed_oof_sha = prediction_input_sha256(development_predictions)
        bindings = lock.get("bindings", {})
        if not isinstance(bindings, Mapping) or bindings.get(
            "development_oof_sha256"
        ) != observed_oof_sha:
            raise ValueError("Locked candidate development OOF binding is stale.")
        return {
            "stage": "development",
            "status": "candidate_locked",
            "candidate": lock["candidate"],
            "calibration": lock["calibration"],
            "lock_sha256": lock["lock_sha256"],
            "resumed_existing_lock": True,
            "development_assignment_sha256": (
                EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256
            ),
            "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
            "V_final_accessed": False,
        }
    all_components = {"full", "role", "bge_small", "bge_base", "feedback_sgd", "nbsvm"}
    component_frames: dict[str, pd.DataFrame] = {}
    raw_frames: list[pd.DataFrame] = []
    for environment in DEVELOPMENT_ENVIRONMENTS:
        print(f"Phase C development: {environment}", flush=True)
        components = load_or_fit_environment_components(
            run_dir,
            frame,
            assignments,
            blocks,
            environment,
            all_components,
            contract,
        )
        component_frames[environment] = components
        raw_frames.append(
            build_raw_candidate_predictions(components, list(CANDIDATE_REGISTRY))
        )
    raw_predictions = pd.concat(raw_frames, ignore_index=True)
    raw_folds, raw_environments = prediction_metric_tables(raw_predictions)
    architecture_selection = development_selection_table(raw_environments)
    _atomic_csv(run_dir / "development_architecture_selection.csv", architecture_selection)
    selected = select_development_candidate(architecture_selection)
    if selected is None:
        _atomic_parquet(run_dir / "candidate_oof.parquet", raw_predictions)
        _atomic_parquet(run_dir / "development_candidate_oof.parquet", raw_predictions)
        _atomic_csv(run_dir / "fold_metrics.csv", raw_folds)
        _atomic_csv(run_dir / "environment_metrics.csv", raw_environments)
        _atomic_csv(run_dir / "development_fold_metrics.csv", raw_folds)
        _atomic_csv(run_dir / "development_environment_metrics.csv", raw_environments)
        result = {
            "stage": "development",
            "status": "stopped_no_candidate_passed_development_guard",
            "locked_candidate": None,
            "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
            "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
            "V_final_accessed": False,
        }
        _write_report(run_dir, result)
        return result

    calibration_frames: list[pd.DataFrame] = []
    nested_parameters: dict[str, dict[str, list[dict[str, object]]]] = {
        BASELINE_CANDIDATE: {},
        selected: {},
    }
    for environment in DEVELOPMENT_ENVIRONMENTS:
        for calibration_candidate in (BASELINE_CANDIDATE, selected):
            raw_selected = raw_predictions.loc[
                raw_predictions["environment"].eq(environment)
                & raw_predictions["candidate"].eq(calibration_candidate)
            ].reset_index(drop=True)
            nested, parameters = load_or_fit_nested_calibration(
                run_dir,
                frame,
                assignments,
                blocks,
                component_frames[environment],
                environment,
                calibration_candidate,
                contract,
            )
            nested_parameters[calibration_candidate][environment] = parameters
            calibration_frames.append(add_selected_calibrations(raw_selected, nested))
    selected_calibrations = pd.concat(calibration_frames, ignore_index=True)
    calibrated_candidates = {BASELINE_CANDIDATE, selected}
    non_selected_raw = raw_predictions.loc[
        ~raw_predictions["candidate"].isin(calibrated_candidates)
    ]
    development_predictions = pd.concat(
        [non_selected_raw, selected_calibrations], ignore_index=True
    )
    fold_metrics, environment_metrics = prediction_metric_tables(development_predictions)
    calibration_selection = calibration_selection_table(
        environment_metrics, [BASELINE_CANDIDATE, selected]
    )
    _atomic_csv(run_dir / "development_calibration_selection.csv", calibration_selection)
    passing = calibration_selection.loc[
        calibration_selection["passes_development_guard"].astype(bool)
    ]
    if passing.empty:
        _atomic_parquet(run_dir / "candidate_oof.parquet", development_predictions)
        _atomic_parquet(
            run_dir / "development_candidate_oof.parquet", development_predictions
        )
        _atomic_csv(run_dir / "fold_metrics.csv", fold_metrics)
        _atomic_csv(run_dir / "environment_metrics.csv", environment_metrics)
        _atomic_csv(run_dir / "development_fold_metrics.csv", fold_metrics)
        _atomic_csv(run_dir / "development_environment_metrics.csv", environment_metrics)
        result = {
            "stage": "development",
            "status": "stopped_no_calibration_passed_development_guard",
            "selected_architecture": selected,
            "locked_candidate": None,
            "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
            "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
            "V_final_accessed": False,
        }
        _write_report(run_dir, result)
        return result
    locked_candidate = str(passing.iloc[0]["candidate"])
    calibration = str(passing.iloc[0]["calibration"])
    selection_row = passing.iloc[0].to_dict()
    development_oof_sha = prediction_input_sha256(development_predictions)
    v_seen_raw = development_predictions.loc[
        development_predictions["environment"].eq("V_seen")
        & development_predictions["candidate"].eq(locked_candidate)
        & development_predictions["calibration"].eq("raw")
    ].reset_index(drop=True)
    deployment_rule = build_deployment_calibration_rule(
        locked_candidate,
        calibration,
        v_seen_raw,
        frame["target"].to_numpy(dtype=np.int8),
    )
    bindings = {
        "phase_contract_sha256": contract.get("phase_contract_sha256"),
        "development_oof_sha256": development_oof_sha,
        "code_sha256": contract.get("code_sha256"),
        "modeling_fingerprint_sha256": contract.get(
            "modeling_fingerprint_sha256"
        ),
        "feature_cache_hashes": contract.get("feature_cache_hashes"),
        "selection_assignment_sha256": contract.get(
            "selection_assignment_sha256"
        ),
        "protocol_sha256": contract.get("protocol_sha256"),
    }
    lock = create_locked_candidate(
        locked_candidate,
        calibration,
        selection_row,
        (
            nested_parameters[locked_candidate]
            if calibration == "nested_platt"
            else {}
        ),
        deployment_rule=deployment_rule,
        bindings=bindings,
    )
    _atomic_parquet(run_dir / "candidate_oof.parquet", development_predictions)
    _atomic_parquet(run_dir / "development_candidate_oof.parquet", development_predictions)
    _atomic_csv(run_dir / "fold_metrics.csv", fold_metrics)
    _atomic_csv(run_dir / "environment_metrics.csv", environment_metrics)
    _atomic_csv(run_dir / "development_fold_metrics.csv", fold_metrics)
    _atomic_csv(run_dir / "development_environment_metrics.csv", environment_metrics)
    _write_new_json(lock_path, lock)
    result = {
        "stage": "development",
        "status": "candidate_locked",
        "selected_architecture": selected,
        "candidate": locked_candidate,
        "calibration": calibration,
        "lock_sha256": lock["lock_sha256"],
        "development_oof_sha256": development_oof_sha,
        "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
        "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
        "V_final_accessed": False,
    }
    _write_report(run_dir, result)
    return result


def _apply_locked_calibration(
    calibration: str,
    raw: pd.DataFrame,
    nested_prediction: np.ndarray | None,
) -> pd.DataFrame:
    result = raw.copy()
    result["calibration"] = calibration
    if calibration == "raw":
        return result
    if calibration == "prior_90":
        result["prediction"] = prior_shrink_prediction(
            result["prediction"].to_numpy(dtype=np.float64),
            result["fold_prior"].to_numpy(dtype=np.float64),
            model_weight=PRIOR_MODEL_WEIGHT,
        )
        return result
    if calibration == "nested_platt":
        if nested_prediction is None or len(nested_prediction) != len(result):
            raise ValueError("Nested Platt prediction is missing or misaligned.")
        result["prediction"] = np.asarray(nested_prediction, dtype=np.float64)
        return result
    raise ValueError(f"Unknown locked calibration: {calibration}")


def run_joint_stage(
    project_root: str | Path,
    run_dir: Path,
    frame: pd.DataFrame,
    blocks: ComponentBlocks,
    contract: Mapping[str, object],
    *,
    n_bootstrap: int,
) -> dict[str, object]:
    lock_path = run_dir / "locked_candidate.json"
    development_prediction_path = run_dir / "development_candidate_oof.parquet"
    if not lock_path.exists() or not development_prediction_path.exists():
        raise FileNotFoundError("Run and pass the development stage before V_joint.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_locked_candidate(lock, contract=contract)
    development = pd.read_parquet(development_prediction_path)
    lock_bindings = lock.get("bindings", {})
    if not isinstance(lock_bindings, Mapping) or lock_bindings.get(
        "development_oof_sha256"
    ) != prediction_input_sha256(development):
        raise ValueError("Development OOF artifact no longer matches the locked decision.")
    # Write the irreversible access sentinel before the gated loader can read
    # any V_joint row. A crash during loading therefore cannot reopen selection.
    joint_opened = write_or_verify_joint_opened(run_dir, lock, contract)
    assignments, joint_manifest = load_joint_confirmation_assignments(
        project_root,
        locked_candidate_sentinel=lock_path,
    )
    development_manifest = joint_manifest.get("development", {})
    if not isinstance(development_manifest, Mapping):
        raise ValueError("Joint loader returned no development contract.")
    joint_record = development_manifest.get("joint_confirmation", {})
    if not isinstance(joint_record, Mapping):
        raise ValueError("Joint loader returned no confirmation artifact contract.")
    joint_sha = str(joint_record.get("assignment_sha256", ""))
    validate_development_assignment_contract(
        assignments,
        expected_sha256=joint_sha,
        expected_environments=(CONFIRMATION_ENVIRONMENT,),
    )
    if development_manifest.get("aggregate_assignment_sha256") != (
        EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256
    ):
        raise ValueError("Joint manifest aggregate assignment SHA changed.")
    joint_binding = write_or_verify_joint_assignment_binding(
        run_dir, joint_opened, assignments
    )
    candidate = str(lock["candidate"])
    calibration = str(lock["calibration"])
    required = required_components(BASELINE_CANDIDATE) | required_components(candidate)
    print(
        f"Phase C confirmation: {CONFIRMATION_ENVIRONMENT}, locked={candidate}/{calibration}",
        flush=True,
    )
    components = load_or_fit_environment_components(
        run_dir,
        frame,
        assignments,
        blocks,
        CONFIRMATION_ENVIRONMENT,
        required,
        contract,
    )
    joint_raw = build_raw_candidate_predictions(
        components, list(dict.fromkeys([BASELINE_CANDIDATE, candidate]))
    )
    baseline = joint_raw.loc[
        joint_raw["candidate"].eq(BASELINE_CANDIDATE)
        & joint_raw["calibration"].eq("raw")
    ].copy()
    selected_raw = joint_raw.loc[
        joint_raw["candidate"].eq(candidate)
        & joint_raw["calibration"].eq("raw")
    ].reset_index(drop=True)
    nested: np.ndarray | None = None
    if calibration == "nested_platt":
        nested, _ = load_or_fit_nested_calibration(
            run_dir,
            frame,
            assignments,
            blocks,
            components,
            CONFIRMATION_ENVIRONMENT,
            candidate,
            contract,
        )
    selected_joint = _apply_locked_calibration(calibration, selected_raw, nested)
    joint_predictions = pd.concat([baseline, selected_joint], ignore_index=True)

    development_selected = development.loc[
        development["environment"].isin(DEVELOPMENT_ENVIRONMENTS)
        & (
            (
                development["candidate"].eq(BASELINE_CANDIDATE)
                & development["calibration"].eq("raw")
            )
            | (
                development["candidate"].eq(candidate)
                & development["calibration"].eq(calibration)
            )
        )
    ].copy()
    all_predictions = pd.concat([development_selected, joint_predictions], ignore_index=True)
    fold_metrics, environment_metrics = prediction_metric_tables(all_predictions)
    selected_only = all_predictions.loc[
        all_predictions["candidate"].eq(candidate)
        & all_predictions["calibration"].eq(calibration)
    ].copy()
    bootstrap = paired_bootstrap_summary(
        selected_only, n_replicates=n_bootstrap, seed=20260720
    )
    bootstrap_input_hashes = set(
        bootstrap["prediction_input_sha256"].astype(str)
    )
    if len(bootstrap_input_hashes) != 1:
        raise RuntimeError("Bootstrap emitted inconsistent prediction input hashes.")
    prediction_sha = next(iter(bootstrap_input_hashes))
    selected_metric_mask = (
        environment_metrics["candidate"].eq(candidate)
        & environment_metrics["calibration"].eq(calibration)
        & environment_metrics["environment"].isin(ALLOWED_ENVIRONMENTS)
    )
    environment_metrics["prediction_input_sha256"] = ""
    environment_metrics["lock_sha256"] = ""
    environment_metrics["phase_contract_sha256"] = ""
    environment_metrics.loc[
        selected_metric_mask, "prediction_input_sha256"
    ] = prediction_sha
    environment_metrics.loc[selected_metric_mask, "lock_sha256"] = str(
        lock["lock_sha256"]
    )
    environment_metrics.loc[
        selected_metric_mask, "phase_contract_sha256"
    ] = str(contract["phase_contract_sha256"])
    gate = gate_c_decision(
        environment_metrics,
        bootstrap,
        candidate=candidate,
        calibration=calibration,
        expected_prediction_input_sha256=prediction_sha,
        expected_lock_sha256=str(lock["lock_sha256"]),
        expected_phase_contract_sha256=str(contract["phase_contract_sha256"]),
    )
    _atomic_parquet(run_dir / "candidate_oof.parquet", all_predictions)
    _atomic_csv(run_dir / "fold_metrics.csv", fold_metrics)
    _atomic_csv(run_dir / "environment_metrics.csv", environment_metrics)
    _atomic_csv(run_dir / "bootstrap_summary.csv", bootstrap)
    _atomic_json(run_dir / "gate_report.json", gate)
    result = {
        "stage": "joint",
        "status": "passed_gate_c" if gate["passes_gate_c"] else "failed_gate_c",
        "candidate": candidate,
        "calibration": calibration,
        "gate_report": gate,
        "bootstrap_replicates": int(n_bootstrap),
        "joint_assignment_sha256": joint_sha,
        "joint_opened_sha256": joint_opened["joint_opened_sha256"],
        "joint_binding_sha256": joint_binding["joint_binding_sha256"],
        "development_assignment_sha256": EXPECTED_DEVELOPMENT_ASSIGNMENT_SHA256,
        "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
        "V_final_accessed": False,
    }
    _write_report(run_dir, result)
    return result


def run_environment_component_validation(
    project_root: str | Path,
    *,
    stage: str = "auto",
    run_id: str | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    resume: bool = False,
) -> dict[str, object]:
    if stage not in {"development", "joint", "auto"}:
        raise ValueError("stage must be development, joint, or auto; V_final is forbidden.")
    if stage == "joint" and run_id is None:
        raise ValueError("The joint stage requires the existing development run_id.")
    if stage == "joint" and not resume:
        raise ValueError("The joint stage requires explicit --resume authorization.")
    assignments, manifest = load_development_assignments(project_root)
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    if "target" not in frame:
        raise ValueError("Phase C development fitting requires training targets.")
    development_manifest = manifest.get("development", {})
    if not isinstance(development_manifest, Mapping):
        raise ValueError("Selection loader returned no development manifest.")
    selection_manifest = development_manifest.get("selection", {})
    if not isinstance(selection_manifest, Mapping):
        raise ValueError("Selection loader returned no selection artifact contract.")
    selection_sha = str(selection_manifest.get("assignment_sha256", ""))
    validate_development_assignment_contract(
        assignments,
        expected_sha256=selection_sha,
        expected_environments=DEVELOPMENT_ENVIRONMENTS,
    )
    run_id, run_dir = _new_run_dir(project_root, run_id, resume=resume)
    contract = build_phase_contract(
        project_root,
        frame,
        manifest,
        selection_assignment_sha256=selection_sha,
        selection_assignment_rows=len(assignments),
        n_bootstrap=n_bootstrap,
    )
    contract = write_or_verify_phase_contract(run_dir, contract, resume=resume)
    blocks = prepare_component_blocks(project_root, frame)
    development_result: dict[str, object] | None = None
    should_run_development = stage == "development" or (
        stage == "auto" and not (run_dir / "locked_candidate.json").exists()
    )
    if should_run_development:
        development_result = run_development_stage(
            project_root, run_dir, frame, assignments, blocks, contract
        )
        if development_result.get("locked_candidate") is None and development_result.get(
            "status"
        ) != "candidate_locked":
            return {"run_id": run_id, "run_dir": str(run_dir), **development_result}
    if stage in {"joint", "auto"}:
        joint_result = run_joint_stage(
            project_root,
            run_dir,
            frame,
            blocks,
            contract,
            n_bootstrap=n_bootstrap,
        )
        return {"run_id": run_id, "run_dir": str(run_dir), **joint_result}
    assert development_result is not None
    return {"run_id": run_id, "run_dir": str(run_dir), **development_result}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe Phase C component validation on the frozen development "
            "environments. This command cannot open V_final."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage", choices=("development", "joint", "auto"), default="auto"
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly resume an existing immutable Phase C run directory.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_environment_component_validation(
        args.project_root,
        stage=args.stage,
        run_id=args.run_id,
        n_bootstrap=args.bootstrap_replicates,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
