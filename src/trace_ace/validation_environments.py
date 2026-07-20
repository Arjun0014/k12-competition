from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from trace_ace.foundation import sha256_file
from trace_ace.io import discover_project_paths


SCHEMA_VERSION = "2026-07-20-v1"
PROTOCOL_VERSION = "2026-07-20-v6"
DEVELOPMENT_SEED = 20260720
DEVELOPMENT_SUITE = "development"
ENVIRONMENTS = ("V_seen", "V_objective", "V_style", "V_joint")
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
SELECTION_ASSIGNMENT_PATH = "data_cache/validation_environments_selection.parquet"
JOINT_CONFIRMATION_ASSIGNMENT_PATH = (
    "data_cache/validation_environment_joint_confirmation.parquet"
)
LEGACY_COMBINED_ASSIGNMENT_NAME = "validation_environments_development.parquet"
LEGACY_FINAL_ASSIGNMENT_NAME = "validation_environments_final.parquet"
ENVIRONMENT_DEFINITIONS: dict[str, dict[str, object]] = {
    "V_seen": {
        "display_name": "seen-objective session holdout",
        "holdout_unit": "session",
        "scoring_rule": "fold membership AND evaluation_eligible",
    },
    "V_objective": {
        "display_name": "semantic-family holdout",
        "holdout_unit": "semantic_family",
        "scoring_rule": "fold membership",
    },
    "V_style": {
        "display_name": "session-style-cell holdout",
        "holdout_unit": "style_cell",
        "scoring_rule": "fold membership",
    },
    # Compatibility keeps the public V_joint name.  This is deliberately not
    # described as a simultaneous family-and-style exclusion: either marginal
    # group can still occur in training when the exact combination is held out.
    "V_joint": {
        "display_name": "interaction-combination holdout",
        "holdout_unit": "semantic_family_x_style_cell_combination",
        "scoring_rule": "fold membership",
        "guarantee": "the exact family/style combination is absent from training",
        "non_guarantees": [
            "the semantic family can occur with other style cells in training",
            "the style cell can occur with other semantic families in training",
        ],
    },
}
STYLE_FEATURES = (
    "transcript_length_log",
    "asr_uncertainty",
    "background_share",
    "student_role_share",
    "turn_density_log",
    "typedness_proxy",
)
FORBIDDEN_ASSIGNMENT_COLUMNS = {
    "target",
    "is_correct",
    "correct",
    "probability",
    "prediction",
    "log_loss",
    "roc_auc",
    "auroc",
}
ASSIGNMENT_COLUMNS = (
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
)


def _seeded_int(seed: int, value: object) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assignment_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["suite", "environment", "response_id"], kind="mergesort"
    ).reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_input(frame: pd.DataFrame, objective_embeddings: np.ndarray) -> None:
    required = {
        "response_id",
        "session_id",
        "learning_objective_id",
        "learning_objective",
        "n_utterances",
        "duration_seconds",
        "content_chars",
        "student_fraction",
        "tutor_fraction",
        "background_fraction",
        "unclear_fraction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing validation-environment columns: {missing}")
    normalized_response_ids = frame["response_id"].astype(str)
    if normalized_response_ids.duplicated().any():
        raise ValueError("response_id must be unique.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Validation-environment source columns cannot contain null values.")
    for column in ("response_id", "session_id", "learning_objective_id"):
        if frame[column].astype(str).str.len().eq(0).any():
            raise ValueError(f"{column} cannot contain empty identifiers.")
    if objective_embeddings.ndim != 2 or objective_embeddings.shape[0] != len(frame):
        raise ValueError("Objective embeddings do not align with the modeling rows.")
    if not np.isfinite(objective_embeddings).all():
        raise ValueError("Objective embeddings contain non-finite values.")

    normalized = pd.DataFrame(
        {
            "learning_objective_id": frame["learning_objective_id"].astype(str),
            "learning_objective": frame["learning_objective"].astype(str),
            "row": np.arange(len(frame), dtype=np.int64),
        }
    )
    text_counts = normalized.groupby("learning_objective_id", sort=True)[
        "learning_objective"
    ].nunique(dropna=False)
    inconsistent_text = text_counts.index[text_counts.ne(1)].tolist()
    if inconsistent_text:
        raise ValueError(
            "learning_objective text varies within learning_objective_id: "
            f"{inconsistent_text[:5]}"
        )

    inconsistent_embeddings: list[str] = []
    for objective_id, group in normalized.groupby("learning_objective_id", sort=True):
        positions = group["row"].to_numpy(dtype=np.int64)
        vectors = np.asarray(objective_embeddings[positions])
        if len(vectors) > 1 and not np.array_equal(
            vectors, np.broadcast_to(vectors[0], vectors.shape)
        ):
            inconsistent_embeddings.append(str(objective_id))
    if inconsistent_embeddings:
        raise ValueError(
            "Objective embeddings vary within learning_objective_id: "
            f"{inconsistent_embeddings[:5]}"
        )


def _objective_families(
    frame: pd.DataFrame,
    objective_embeddings: np.ndarray,
    n_families: int,
    seed: int,
) -> pd.Series:
    indexed = frame.reset_index(drop=True).reset_index(names="response_row")
    objectives = (
        indexed.sort_values("learning_objective_id", kind="mergesort")
        .drop_duplicates("learning_objective_id")
        .loc[:, ["learning_objective_id", "response_row"]]
        .reset_index(drop=True)
    )
    if n_families < 2 or n_families > len(objectives):
        raise ValueError("n_families must be between 2 and the objective count.")
    vectors = objective_embeddings[
        objectives["response_row"].to_numpy(dtype=np.int64)
    ].astype(np.float64, copy=True)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    labels = KMeans(
        n_clusters=n_families,
        random_state=seed,
        n_init=20,
        max_iter=500,
    ).fit_predict(vectors)
    if len(np.unique(labels)) != n_families:
        raise ValueError("Objective embeddings do not support the requested family count.")
    lookup = pd.Series(
        labels.astype(np.int32), index=objectives["learning_objective_id"].astype(str)
    )
    mapped = frame["learning_objective_id"].astype(str).map(lookup)
    if mapped.isna().any():
        raise RuntimeError("Incomplete semantic-family assignment.")
    return mapped.astype(np.int32)


def build_session_style_proxies(
    frame: pd.DataFrame,
    session_behavior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build label-free style proxies once per training session.

    Every input is a transcript-derived quantity.  No response label, objective
    outcome, test-set aggregate, or batch-relative statistic is used.
    """

    base_columns = [
        "session_id",
        "n_utterances",
        "duration_seconds",
        "content_chars",
        "student_fraction",
        "tutor_fraction",
        "background_fraction",
        "unclear_fraction",
    ]
    missing = sorted(set(base_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing style source columns: {missing}")
    grouped = frame[base_columns].groupby("session_id", sort=True, as_index=False)
    spread = grouped.nunique(dropna=False)
    inconsistent = spread.drop(columns="session_id").gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError("Session-level transcript features vary within a session.")
    sessions = grouped.first()

    behavior_lookup: pd.DataFrame | None = None
    if session_behavior is not None:
        required_behavior = {"session_id", "role_switches", "tutor_words", "student_words"}
        missing_behavior = sorted(required_behavior.difference(session_behavior.columns))
        if missing_behavior:
            raise ValueError(f"Missing session behavior columns: {missing_behavior}")
        if session_behavior["session_id"].duplicated().any():
            raise ValueError("session_behavior contains duplicate session_id values.")
        behavior_lookup = session_behavior[list(required_behavior)].copy()
        sessions = sessions.merge(
            behavior_lookup, on="session_id", how="left", validate="one_to_one"
        )
        if sessions[["role_switches", "tutor_words", "student_words"]].isna().any().any():
            raise ValueError("At least one training session is missing behavior features.")

    utterances = np.maximum(sessions["n_utterances"].to_numpy(dtype=np.float64), 1.0)
    duration = np.maximum(sessions["duration_seconds"].to_numpy(dtype=np.float64), 1.0)
    role_total = np.maximum(
        sessions["student_fraction"].to_numpy(dtype=np.float64)
        + sessions["tutor_fraction"].to_numpy(dtype=np.float64),
        1e-8,
    )
    if behavior_lookup is None:
        turns = utterances
        words_or_chars = sessions["content_chars"].to_numpy(dtype=np.float64)
    else:
        turns = sessions["role_switches"].to_numpy(dtype=np.float64) + 1.0
        words_or_chars = (
            sessions["tutor_words"].to_numpy(dtype=np.float64)
            + sessions["student_words"].to_numpy(dtype=np.float64)
        )

    result = pd.DataFrame(
        {
            "session_id": sessions["session_id"].astype(str),
            "transcript_length_log": np.log1p(utterances),
            "asr_uncertainty": sessions["unclear_fraction"].to_numpy(dtype=np.float64),
            "background_share": sessions["background_fraction"].to_numpy(dtype=np.float64),
            "student_role_share": (
                sessions["student_fraction"].to_numpy(dtype=np.float64) / role_total
            ),
            "turn_density_log": np.log1p(turns / np.maximum(duration / 60.0, 1.0 / 60.0)),
            "typedness_proxy": (
                np.log1p(words_or_chars / utterances)
                - np.log1p(duration / utterances)
            ),
        }
    )
    values = result[list(STYLE_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Style proxies contain non-finite values.")
    return result.sort_values("session_id", kind="mergesort").reset_index(drop=True)


def _style_cells(
    style_proxies: pd.DataFrame,
    n_style_cells: int,
    seed: int,
) -> pd.Series:
    if n_style_cells < 2 or n_style_cells > len(style_proxies):
        raise ValueError("n_style_cells must be between 2 and the session count.")
    values = style_proxies[list(STYLE_FEATURES)].to_numpy(dtype=np.float64)
    median = np.median(values, axis=0)
    lower = np.quantile(values, 0.25, axis=0)
    upper = np.quantile(values, 0.75, axis=0)
    scale = np.where(upper - lower > 1e-8, upper - lower, 1.0)
    normalized = np.clip((values - median) / scale, -10.0, 10.0)
    labels = KMeans(
        n_clusters=n_style_cells,
        random_state=seed + 101,
        n_init=20,
        max_iter=500,
    ).fit_predict(normalized)
    if len(np.unique(labels)) != n_style_cells:
        raise ValueError("Style proxies do not support the requested cell count.")
    return pd.Series(
        labels.astype(np.int32), index=style_proxies["session_id"].astype(str)
    )


def _balanced_group_folds(
    groups: pd.Series,
    n_splits: int,
    seed: int,
) -> pd.Series:
    weights = groups.astype(str).value_counts(sort=False)
    if len(weights) < n_splits:
        raise ValueError("There must be at least one holdout group per fold.")
    ordered = sorted(
        weights.index,
        key=lambda value: (-int(weights[value]), _seeded_int(seed, value), value),
    )
    loads = np.zeros(n_splits, dtype=np.int64)
    group_counts = np.zeros(n_splits, dtype=np.int64)
    assigned: dict[str, int] = {}
    for group in ordered:
        candidates = sorted(
            range(n_splits),
            key=lambda fold: (
                int(loads[fold]),
                int(group_counts[fold]),
                _seeded_int(seed, f"{group}|{fold}"),
            ),
        )
        fold = candidates[0]
        assigned[str(group)] = fold
        loads[fold] += int(weights[group])
        group_counts[fold] += 1
    return groups.astype(str).map(assigned).astype(np.int8)


def _seen_session_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    pairs = (
        frame[["session_id", "learning_objective_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["session_id", "learning_objective_id"], kind="mergesort")
    )
    session_objectives = pairs.groupby("session_id", sort=True)["learning_objective_id"].agg(tuple)
    objective_totals = pairs.groupby("learning_objective_id")["session_id"].nunique().to_dict()
    row_weights = frame["session_id"].astype(str).value_counts().to_dict()
    order = sorted(
        session_objectives.index,
        key=lambda session: (
            min(objective_totals[obj] for obj in session_objectives[session]),
            -len(session_objectives[session]),
            _seeded_int(seed, session),
            session,
        ),
    )
    loads = np.zeros(n_splits, dtype=np.int64)
    counts = {
        objective: np.zeros(n_splits, dtype=np.int32) for objective in objective_totals
    }
    assigned: dict[str, int] = {}
    target_load = max(1.0, len(frame) / n_splits)
    for session in order:
        objectives = session_objectives[session]
        fold = min(
            range(n_splits),
            key=lambda candidate: (
                sum(
                    (counts[objective][candidate] + 1.0) / objective_totals[objective]
                    for objective in objectives
                )
                + 0.05 * loads[candidate] / target_load,
                int(loads[candidate]),
                _seeded_int(seed, f"{session}|{candidate}"),
            ),
        )
        assigned[session] = fold
        loads[fold] += int(row_weights[session])
        for objective in objectives:
            counts[objective][fold] += 1

    # A target-free repair makes every mathematically eligible objective non-monochromatic.
    for _ in range(max(1, 4 * len(objective_totals))):
        violations = [
            objective
            for objective, total in objective_totals.items()
            if total >= 2 and np.count_nonzero(counts[objective]) == 1
        ]
        if not violations:
            break
        objective = sorted(violations)[0]
        source = int(np.flatnonzero(counts[objective])[0])
        candidates: list[tuple[tuple[float, ...], str, int]] = []
        objective_sessions = pairs.loc[
            pairs["learning_objective_id"].eq(objective), "session_id"
        ]
        for session in objective_sessions:
            session = str(session)
            if assigned[session] != source:
                continue
            for destination in range(n_splits):
                if destination == source:
                    continue
                newly_monochromatic = 0
                for linked in session_objectives[session]:
                    after = counts[linked].copy()
                    after[source] -= 1
                    after[destination] += 1
                    if objective_totals[linked] >= 2 and np.count_nonzero(after) == 1:
                        newly_monochromatic += 1
                score = (
                    float(newly_monochromatic),
                    float(loads[destination] + row_weights[session]),
                    float(-loads[source]),
                    float(_seeded_int(seed, f"repair|{session}|{destination}")),
                )
                candidates.append((score, session, destination))
        if not candidates:
            break
        _, session, destination = min(candidates, key=lambda item: item[0])
        source = assigned[session]
        assigned[session] = destination
        loads[source] -= int(row_weights[session])
        loads[destination] += int(row_weights[session])
        for linked in session_objectives[session]:
            counts[linked][source] -= 1
            counts[linked][destination] += 1

    unresolved = [
        objective
        for objective, total in objective_totals.items()
        if total >= 2 and np.count_nonzero(counts[objective]) == 1
    ]
    if unresolved:
        raise RuntimeError(
            f"Could not construct objective-seen session folds for {len(unresolved)} objectives."
        )
    return frame["session_id"].astype(str).map(assigned).astype(np.int8)


def build_validation_environments(
    frame: pd.DataFrame,
    objective_embeddings: np.ndarray,
    *,
    seed: int,
    suite_name: str,
    n_splits: int = 5,
    n_semantic_families: int | None = None,
    n_style_cells: int | None = None,
    session_behavior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the four frozen, target-independent development environments.

    ``V_joint`` is an interaction-combination holdout.  It excludes an exact
    semantic-family/style-cell pair, not both marginal groups independently.
    Sealed final-holdout assignments are managed outside this developer API.
    """

    if str(suite_name) != DEVELOPMENT_SUITE:
        raise PermissionError(
            "The developer API builds suite='development' only; the sealed final "
            "holdout is managed outside this API."
        )
    if int(seed) != DEVELOPMENT_SEED:
        raise PermissionError(
            "The developer API accepts the frozen development seed only; alternate "
            "or sealed-holdout seeds are outside this API."
        )
    _check_input(frame, objective_embeddings)
    if n_splits < 2 or n_splits > 127:
        raise ValueError("n_splits must be between 2 and 127.")
    objective_count = frame["learning_objective_id"].nunique()
    session_count = frame["session_id"].nunique()
    family_count = n_semantic_families or min(50, objective_count)
    style_count = n_style_cells or min(20, session_count)
    if family_count < n_splits or style_count < n_splits:
        raise ValueError("Semantic-family and style-cell counts must cover every fold.")

    families = _objective_families(frame, objective_embeddings, family_count, seed)
    proxies = build_session_style_proxies(frame, session_behavior)
    style_lookup = _style_cells(proxies, style_count, seed)
    style_cells = frame["session_id"].astype(str).map(style_lookup).astype(np.int32)
    if style_cells.isna().any():
        raise RuntimeError("Incomplete style-cell assignment.")
    joint_cells = pd.Series(
        [f"family_{family:03d}__style_{style:03d}" for family, style in zip(families, style_cells)],
        index=frame.index,
        dtype="string",
    )

    seen_folds = _seen_session_folds(frame, n_splits, seed + 11)
    objective_folds = _balanced_group_folds(
        families.astype(str), n_splits, seed + 23
    )
    style_folds = _balanced_group_folds(
        style_cells.astype(str), n_splits, seed + 37
    )
    joint_folds = _balanced_group_folds(joint_cells, n_splits, seed + 53)
    objective_session_counts = frame.groupby("learning_objective_id")["session_id"].nunique()
    seen_eligible = (
        frame["learning_objective_id"].map(objective_session_counts).to_numpy() >= 2
    )

    rows: list[pd.DataFrame] = []
    for environment, folds in (
        ("V_seen", seen_folds),
        ("V_objective", objective_folds),
        ("V_style", style_folds),
        ("V_joint", joint_folds),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "schema_version": SCHEMA_VERSION,
                    "suite": DEVELOPMENT_SUITE,
                    "split_seed": int(seed),
                    "environment": environment,
                    "response_id": frame["response_id"].astype(str).to_numpy(),
                    "session_id": frame["session_id"].astype(str).to_numpy(),
                    "learning_objective_id": frame["learning_objective_id"].astype(str).to_numpy(),
                    "fold": np.asarray(folds, dtype=np.int8),
                    "semantic_family": families.to_numpy(dtype=np.int32),
                    "style_cell": style_cells.to_numpy(dtype=np.int32),
                    "joint_cell": joint_cells.astype(str).to_numpy(),
                    "evaluation_eligible": (
                        seen_eligible if environment == "V_seen" else np.ones(len(frame), dtype=bool)
                    ),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    forbidden = FORBIDDEN_ASSIGNMENT_COLUMNS.intersection(result.columns)
    if forbidden:
        raise RuntimeError(f"Target-derived columns leaked into assignments: {sorted(forbidden)}")
    return result.sort_values(
        ["suite", "environment", "response_id"], kind="mergesort"
    ).reset_index(drop=True)


def _strict_integer_series(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain finite integers.")
    return pd.Series(array.astype(np.int64), index=values.index, name=name)


def _strict_boolean_array(values: pd.Series, name: str) -> np.ndarray:
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{name} must contain booleans.")
    return values.to_numpy(dtype=bool)


def _aligned_single_environment(
    frame: pd.DataFrame,
    environment_assignments: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    required = {
        "suite",
        "environment",
        "response_id",
        "session_id",
        "learning_objective_id",
        "fold",
        "evaluation_eligible",
    }
    missing = sorted(required.difference(environment_assignments.columns))
    if missing:
        raise ValueError(f"Environment assignments are missing columns: {missing}")
    if environment_assignments[list(required)].isna().any().any():
        raise ValueError("Environment assignments cannot contain null contract values.")

    subset = environment_assignments.copy()
    subset["suite"] = subset["suite"].astype(str)
    subset["environment"] = subset["environment"].astype(str)
    subset["response_id"] = subset["response_id"].astype(str)
    subset["session_id"] = subset["session_id"].astype(str)
    subset["learning_objective_id"] = subset["learning_objective_id"].astype(str)
    suites = set(subset["suite"])
    if suites != {DEVELOPMENT_SUITE}:
        raise PermissionError(
            "Fold-mask APIs accept suite='development' only; sealed final assignments "
            "are outside the developer API."
        )
    environments = set(subset["environment"])
    if len(environments) != 1 or not environments.issubset(ENVIRONMENTS):
        raise ValueError(f"Expected one known environment, found {sorted(environments)}.")
    environment = next(iter(environments))

    required_frame = {"response_id", "session_id", "learning_objective_id"}
    missing_frame = sorted(required_frame.difference(frame.columns))
    if missing_frame:
        raise ValueError(f"Modeling frame is missing columns: {missing_frame}")
    source = frame[["response_id", "session_id", "learning_objective_id"]].copy()
    for column in source.columns:
        source[column] = source[column].astype(str)
    if source["response_id"].duplicated().any():
        raise ValueError("The modeling frame must contain unique response_id values.")
    if subset["response_id"].duplicated().any():
        raise ValueError("Environment assignments must contain one row per response.")
    expected_ids = set(source["response_id"])
    assigned_ids = set(subset["response_id"])
    if len(subset) != len(source) or assigned_ids != expected_ids:
        extra = len(assigned_ids.difference(expected_ids))
        missing_count = len(expected_ids.difference(assigned_ids))
        raise ValueError(
            "Environment assignments do not match the modeling frame exactly "
            f"(missing={missing_count}, extra={extra})."
        )

    aligned = subset.set_index("response_id").loc[source["response_id"]].reset_index()
    for column in ("session_id", "learning_objective_id"):
        expected = source[column].to_numpy(dtype=str)
        actual = aligned[column].to_numpy(dtype=str)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Assignment {column} does not match the source frame.")
    aligned["fold"] = _strict_integer_series(aligned["fold"], "fold")
    _strict_boolean_array(aligned["evaluation_eligible"], "evaluation_eligible")
    return environment, aligned


def evaluation_mask_for_fold(
    frame: pd.DataFrame,
    environment_assignments: pd.DataFrame,
    fold: int,
) -> np.ndarray:
    """Return the eligibility-aware scoring mask for one development fold."""

    _, aligned = _aligned_single_environment(frame, environment_assignments)
    held_out = aligned["fold"].to_numpy(dtype=np.int64) == int(fold)
    eligible = _strict_boolean_array(
        aligned["evaluation_eligible"], "evaluation_eligible"
    )
    return held_out & eligible


def purged_fold_masks(
    frame: pd.DataFrame,
    environment_assignments: pd.DataFrame,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return session-purged training and complete held-out-fold masks.

    The complete fold, including score-ineligible ``V_seen`` rows, establishes
    the sessions purged from training.  Call :func:`evaluation_mask_for_fold`
    for the narrower scoring mask.  The two-array return shape is preserved for
    downstream Phase C compatibility.
    """

    _, aligned = _aligned_single_environment(frame, environment_assignments)
    validation_mask = aligned["fold"].to_numpy(dtype=np.int64) == int(fold)
    validation_sessions = set(frame.loc[validation_mask, "session_id"].astype(str))
    training_mask = (~validation_mask) & (
        ~frame["session_id"].astype(str).isin(validation_sessions).to_numpy()
    )
    return training_mask, validation_mask


def _validate_assignment_contract(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    n_splits: int,
) -> None:
    forbidden = FORBIDDEN_ASSIGNMENT_COLUMNS.intersection(assignments.columns)
    if forbidden:
        raise ValueError(f"Assignments contain forbidden columns: {sorted(forbidden)}")
    missing = sorted(set(ASSIGNMENT_COLUMNS).difference(assignments.columns))
    if missing:
        raise ValueError(f"Assignments are missing contract columns: {missing}")
    if assignments[list(ASSIGNMENT_COLUMNS)].isna().any().any():
        raise ValueError("Assignments cannot contain null contract values.")
    if n_splits < 2 or n_splits > 127:
        raise ValueError("n_splits must be between 2 and 127.")

    contract = assignments[list(ASSIGNMENT_COLUMNS)].copy()
    for column in (
        "schema_version",
        "suite",
        "environment",
        "response_id",
        "session_id",
        "learning_objective_id",
        "joint_cell",
    ):
        contract[column] = contract[column].astype(str)

    suites = set(contract["suite"])
    if suites != {DEVELOPMENT_SUITE}:
        raise PermissionError(
            "Validation accepts suite='development' only; unknown or sealed suites "
            f"are forbidden: {sorted(suites)}."
        )
    environments = set(contract["environment"])
    if environments != set(ENVIRONMENTS):
        unknown = sorted(environments.difference(ENVIRONMENTS))
        missing_environments = sorted(set(ENVIRONMENTS).difference(environments))
        raise ValueError(
            "Assignments must contain exactly the known environments "
            f"(unknown={unknown}, missing={missing_environments})."
        )
    if set(contract["schema_version"]) != {SCHEMA_VERSION}:
        raise ValueError("Assignments use an unknown schema_version.")
    split_seeds = _strict_integer_series(contract["split_seed"], "split_seed")
    if set(split_seeds.astype(int)) != {DEVELOPMENT_SEED}:
        raise ValueError("Assignments must use the frozen development split_seed.")
    contract["fold"] = _strict_integer_series(contract["fold"], "fold")
    contract["semantic_family"] = _strict_integer_series(
        contract["semantic_family"], "semantic_family"
    )
    contract["style_cell"] = _strict_integer_series(contract["style_cell"], "style_cell")
    if (contract[["semantic_family", "style_cell"]] < 0).any().any():
        raise ValueError("semantic_family and style_cell must be non-negative.")
    eligible = _strict_boolean_array(
        contract["evaluation_eligible"], "evaluation_eligible"
    )
    contract["evaluation_eligible"] = eligible

    required_frame = {"response_id", "session_id", "learning_objective_id"}
    missing_frame = sorted(required_frame.difference(frame.columns))
    if missing_frame:
        raise ValueError(f"Modeling frame is missing columns: {missing_frame}")
    source = frame[["response_id", "session_id", "learning_objective_id"]].copy()
    for column in source.columns:
        source[column] = source[column].astype(str)
    if source["response_id"].duplicated().any():
        raise ValueError("The modeling frame must contain unique response_id values.")
    expected_rows = len(source) * len(ENVIRONMENTS)
    if len(contract) != expected_rows:
        raise ValueError(
            f"Assignments contain {len(contract)} rows; expected exactly {expected_rows}."
        )

    expected_ids = set(source["response_id"])
    source_by_id = source.set_index("response_id")
    for environment in ENVIRONMENTS:
        subset = contract.loc[contract["environment"].eq(environment)].copy()
        if subset["response_id"].duplicated().any():
            raise ValueError(f"{environment} contains duplicate response assignments.")
        assigned_ids = set(subset["response_id"])
        if len(subset) != len(source) or assigned_ids != expected_ids:
            extra = len(assigned_ids.difference(expected_ids))
            missing_count = len(expected_ids.difference(assigned_ids))
            raise ValueError(
                f"{environment} response rows do not match the source frame "
                f"(missing={missing_count}, extra={extra})."
            )
        aligned = subset.set_index("response_id").loc[source["response_id"]]
        for column in ("session_id", "learning_objective_id"):
            expected = source_by_id.loc[source["response_id"], column].to_numpy(dtype=str)
            if not np.array_equal(aligned[column].to_numpy(dtype=str), expected):
                raise ValueError(
                    f"{environment} assignment {column} does not match the source frame."
                )
        if set(aligned["fold"].astype(int)) != set(range(n_splits)):
            raise ValueError(f"{environment} does not populate every fold.")

    response_consistency = contract.groupby("response_id", sort=True)[
        [
            "session_id",
            "learning_objective_id",
            "semantic_family",
            "style_cell",
            "joint_cell",
        ]
    ].nunique(dropna=False)
    if response_consistency.gt(1).any().any():
        raise ValueError("Assignment identity fields vary by environment for a response.")
    family_counts = contract.groupby("learning_objective_id", sort=True)[
        "semantic_family"
    ].nunique(dropna=False)
    if family_counts.ne(1).any():
        raise ValueError("semantic_family must be constant per learning_objective_id.")
    style_counts = contract.groupby("session_id", sort=True)["style_cell"].nunique(
        dropna=False
    )
    if style_counts.ne(1).any():
        raise ValueError("style_cell must be constant per session_id.")
    expected_joint = np.asarray(
        [
            f"family_{family:03d}__style_{style:03d}"
            for family, style in zip(
                contract["semantic_family"].to_numpy(dtype=np.int64),
                contract["style_cell"].to_numpy(dtype=np.int64),
            )
        ],
        dtype=str,
    )
    if not np.array_equal(contract["joint_cell"].to_numpy(dtype=str), expected_joint):
        raise ValueError("joint_cell is not the canonical family/style combination.")

    objective_session_counts = source.groupby("learning_objective_id")[
        "session_id"
    ].nunique()
    expected_seen = source["learning_objective_id"].map(objective_session_counts).ge(2)
    for environment in ENVIRONMENTS:
        subset = contract.loc[contract["environment"].eq(environment)].set_index(
            "response_id"
        )
        actual = subset.loc[source["response_id"], "evaluation_eligible"].to_numpy(
            dtype=bool
        )
        expected = (
            expected_seen.to_numpy(dtype=bool)
            if environment == "V_seen"
            else np.ones(len(source), dtype=bool)
        )
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{environment} evaluation_eligible does not match the protocol."
            )


def validate_environment_assignments(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    n_splits: int,
    min_validation_rows: int = 1,
    min_class_rows: int = 1,
    include_label_diagnostics: bool = True,
    diagnostic_environments: Iterable[str] = ENVIRONMENTS,
) -> list[dict[str, object]]:
    _validate_assignment_contract(frame, assignments, n_splits=n_splits)
    if min_validation_rows < 1 or min_class_rows < 0:
        raise ValueError("Validation and class row thresholds are invalid.")
    requested_environments = tuple(str(value) for value in diagnostic_environments)
    if len(set(requested_environments)) != len(requested_environments):
        raise ValueError("diagnostic_environments cannot contain duplicates.")
    unknown_diagnostics = sorted(set(requested_environments).difference(ENVIRONMENTS))
    if unknown_diagnostics:
        raise ValueError(f"Unknown diagnostic environments: {unknown_diagnostics}")
    summaries: list[dict[str, object]] = []
    for environment in requested_environments:
        subset = assignments.loc[assignments["environment"].eq(environment)].copy()
        _, aligned = _aligned_single_environment(frame, subset)
        for fold in range(n_splits):
            train_mask, held_out_mask = purged_fold_masks(frame, subset, fold)
            scoring_mask = evaluation_mask_for_fold(frame, subset, fold)
            scoring_rows = int(scoring_mask.sum())
            if scoring_rows < min_validation_rows:
                raise ValueError(
                    f"{environment} fold {fold} has only {scoring_rows} eligible validation rows."
                )
            train_sessions = set(frame.loc[train_mask, "session_id"].astype(str))
            validation_sessions = set(frame.loc[held_out_mask, "session_id"].astype(str))
            if train_sessions.intersection(validation_sessions):
                raise RuntimeError(f"Session leakage in {environment} fold {fold}.")
            if environment == "V_objective":
                train_objectives = set(aligned.loc[train_mask, "learning_objective_id"])
                val_objectives = set(aligned.loc[held_out_mask, "learning_objective_id"])
                train_families = set(aligned.loc[train_mask, "semantic_family"])
                val_families = set(aligned.loc[held_out_mask, "semantic_family"])
                if train_objectives.intersection(val_objectives):
                    raise RuntimeError(f"Objective leakage in V_objective fold {fold}.")
                if train_families.intersection(val_families):
                    raise RuntimeError(f"Semantic-family leakage in V_objective fold {fold}.")
            elif environment == "V_style":
                if set(aligned.loc[train_mask, "style_cell"]).intersection(
                    set(aligned.loc[held_out_mask, "style_cell"])
                ):
                    raise RuntimeError(f"Style-cell leakage in V_style fold {fold}.")
            elif environment == "V_joint":
                if set(aligned.loc[train_mask, "joint_cell"]).intersection(
                    set(aligned.loc[held_out_mask, "joint_cell"])
                ):
                    raise RuntimeError(f"Joint-cell leakage in V_joint fold {fold}.")
            elif environment == "V_seen":
                eligible_objectives = set(aligned.loc[scoring_mask, "learning_objective_id"])
                train_objectives = set(aligned.loc[train_mask, "learning_objective_id"])
                missing_seen = eligible_objectives.difference(train_objectives)
                if missing_seen:
                    raise RuntimeError(
                        f"V_seen fold {fold} has {len(missing_seen)} eligible unseen objectives."
                    )

            summary: dict[str, object] = {
                "environment": environment,
                "fold": fold,
                "train_rows_after_session_purge": int(train_mask.sum()),
                "held_out_rows": int(held_out_mask.sum()),
                "validation_rows": scoring_rows,
                "ineligible_rows_excluded_from_scoring": int(
                    held_out_mask.sum() - scoring_mask.sum()
                ),
                "validation_sessions": len(validation_sessions),
                "validation_objectives": int(
                    frame.loc[scoring_mask, "learning_objective_id"].nunique()
                ),
            }
            if include_label_diagnostics:
                if "target" not in frame:
                    raise ValueError("target is required for development class diagnostics.")
                labels = frame.loc[scoring_mask, "target"].astype(int)
                if not labels.isin([0, 1]).all():
                    raise ValueError("Development targets must be binary.")
                class_counts = labels.value_counts().reindex([0, 1], fill_value=0)
                if int(class_counts.min()) < min_class_rows:
                    raise ValueError(
                        f"{environment} fold {fold} fails class adequacy: {class_counts.to_dict()}."
                    )
                summary.update(
                    {
                        "negative_rows": int(class_counts[0]),
                        "positive_rows": int(class_counts[1]),
                        "positive_rate": float(labels.mean()),
                    }
                )
            summaries.append(summary)
    return summaries


def build_validation_manifest(
    *,
    development_assignments: pd.DataFrame,
    source_hashes: Mapping[str, str],
    development_diagnostics: list[dict[str, object]],
    n_splits: int,
    n_semantic_families: int,
    n_style_cells: int,
    min_validation_rows: int = 1,
    min_class_rows: int = 1,
    session_behavior_used: bool = False,
    protocol_version: str = PROTOCOL_VERSION,
    project_root: str | Path | None = None,
    supersedes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not str(protocol_version).strip():
        raise ValueError("protocol_version cannot be empty.")
    missing_assignment_columns = sorted(
        set(ASSIGNMENT_COLUMNS).difference(development_assignments.columns)
    )
    if missing_assignment_columns:
        raise ValueError(
            f"Development assignments are missing columns: {missing_assignment_columns}"
        )
    if set(development_assignments["suite"].astype(str)) != {DEVELOPMENT_SUITE}:
        raise PermissionError(
            "Manifest construction accepts development assignments only; sealed final "
            "assignments are outside this API."
        )
    if set(development_assignments["environment"].astype(str)) != set(ENVIRONMENTS):
        raise ValueError("Manifest construction requires exactly the four known environments.")
    manifest_seeds = _strict_integer_series(
        development_assignments["split_seed"], "split_seed"
    )
    if set(manifest_seeds.astype(int)) != {DEVELOPMENT_SEED}:
        raise ValueError("Manifest construction requires the frozen development seed.")
    forbidden = FORBIDDEN_ASSIGNMENT_COLUMNS.intersection(development_assignments.columns)
    if forbidden:
        raise ValueError(
            f"Development assignments contain forbidden columns: {sorted(forbidden)}"
        )
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    protocol = _build_protocol(
        project_root=root,
        source_hashes=source_hashes,
        protocol_version=str(protocol_version),
        n_splits=n_splits,
        n_semantic_families=n_semantic_families,
        n_style_cells=n_style_cells,
        min_validation_rows=min_validation_rows,
        min_class_rows=min_class_rows,
        session_behavior_used=session_behavior_used,
    )
    selection_assignments = development_assignments.loc[
        development_assignments["environment"].isin(SELECTION_ENVIRONMENTS)
    ].copy()
    joint_assignments = development_assignments.loc[
        development_assignments["environment"].eq(CONFIRMATION_ENVIRONMENT)
    ].copy()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": str(protocol_version),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": dict(sorted(source_hashes.items())),
        "provenance": protocol["provenance"],
        "protocol": protocol,
        "protocol_sha256": _canonical_json_sha256(protocol),
        "development": {
            "seed": DEVELOPMENT_SEED,
            "sealed": False,
            "aggregate_assignment_sha256": assignment_sha256(development_assignments),
            "selection": {
                "access_policy": "normal_developer_api",
                "assignment_path": SELECTION_ASSIGNMENT_PATH,
                "assignment_sha256": assignment_sha256(selection_assignments),
                "row_count": int(len(selection_assignments)),
                "environments": list(SELECTION_ENVIRONMENTS),
                "integrity_and_class_diagnostics": development_diagnostics,
            },
            "joint_confirmation": {
                "access_policy": "locked_candidate_sentinel_required",
                "assignment_path": JOINT_CONFIRMATION_ASSIGNMENT_PATH,
                "assignment_sha256": assignment_sha256(joint_assignments),
                "row_count": int(len(joint_assignments)),
                "environments": [CONFIRMATION_ENVIRONMENT],
                "structural_validation_only": True,
                "targets_or_performance_inspected": False,
            },
        },
        # No location, seed, assignment hash, labels, predictions, or metrics are
        # exposed.  Provisioning and one-time evaluation belong to an external
        # sealed-holdout process, not this developer module or CLI.
        "sealed_final_holdout": {
            "suite": "V_final",
            "sealed": True,
            "managed_outside_developer_api": True,
            "normal_developer_api_access": False,
            "assignment_metadata_exposed": False,
            "targets_or_performance_inspected": False,
            "legacy_workspace_artifacts_absent": True,
        },
    }
    if supersedes is not None:
        manifest["supersedes"] = dict(supersedes)
    return manifest


def _dependency_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for distribution in ("scikit-learn", "scipy", "pyarrow", "joblib", "trace-ace"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed-as-distribution"
    return result


def _implementation_hashes(project_root: Path) -> dict[str, str]:
    candidates = (
        Path(__file__).resolve(),
        project_root / "scripts" / "build_validation_environments.py",
    )
    result: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            name = path.relative_to(project_root).as_posix()
        except ValueError:
            name = path.name
        result[name] = sha256_file(path)
    return dict(sorted(result.items()))


def _build_protocol(
    *,
    project_root: Path,
    source_hashes: Mapping[str, str],
    protocol_version: str,
    n_splits: int,
    n_semantic_families: int,
    n_style_cells: int,
    min_validation_rows: int,
    min_class_rows: int,
    session_behavior_used: bool,
) -> dict[str, object]:
    return {
        "protocol_version": str(protocol_version),
        "assignment_schema_version": SCHEMA_VERSION,
        "suite": DEVELOPMENT_SUITE,
        "environments": {
            name: dict(ENVIRONMENT_DEFINITIONS[name]) for name in ENVIRONMENTS
        },
        "passed_parameters": {
            "n_splits": int(n_splits),
            "n_semantic_families": int(n_semantic_families),
            "n_style_cells": int(n_style_cells),
            "min_validation_rows": int(min_validation_rows),
            "min_class_rows": int(min_class_rows),
            "session_behavior_used": bool(session_behavior_used),
        },
        "seeds": {
            "development_base": DEVELOPMENT_SEED,
            "semantic_family_kmeans": DEVELOPMENT_SEED,
            "style_cell_kmeans": DEVELOPMENT_SEED + 101,
            "seen_session_folds": DEVELOPMENT_SEED + 11,
            "objective_family_folds": DEVELOPMENT_SEED + 23,
            "style_cell_folds": DEVELOPMENT_SEED + 37,
            "interaction_combination_folds": DEVELOPMENT_SEED + 53,
        },
        "algorithms": {
            "objective_family_clustering": {
                "implementation": "sklearn.cluster.KMeans",
                "normalization": "row_l2_with_1e-12_floor",
                "n_init": 20,
                "max_iter": 500,
                "objective_embedding_invariance": "exact_per_learning_objective_id",
                "objective_text_invariance": "exact_per_learning_objective_id",
            },
            "style_cell_clustering": {
                "implementation": "sklearn.cluster.KMeans",
                "fit_unit": "unique_training_session",
                "normalization": "median_and_iqr_then_clip_minus10_plus10",
                "lower_quantile": 0.25,
                "upper_quantile": 0.75,
                "iqr_floor": 1e-8,
                "constant_feature_scale": 1.0,
                "clip_range": [-10.0, 10.0],
                "n_init": 20,
                "max_iter": 500,
                "features": list(STYLE_FEATURES),
            },
            "balanced_group_folds": {
                "implementation": "deterministic_greedy_row_load_balance",
                "tie_breaker": "sha256_seeded_64_bit_integer",
            },
            "seen_session_folds": {
                "implementation": "deterministic_objective_coverage_balance_with_target_free_repair",
                "row_load_penalty": 0.05,
                "evaluation_eligible_min_distinct_sessions": 2,
                "repair_iteration_limit": "4_times_objective_count",
            },
            "assignment_serialization": {
                "sort": ["suite", "environment", "response_id"],
                "hash_format": "utf8_csv_lf_sha256",
            },
        },
        "leakage_controls": {
            "target_independent_assignments": True,
            "session_purge": "all held-out-fold sessions including score-ineligible rows",
            "V_seen_scoring": "fold membership AND evaluation_eligible",
            "development_selection_artifact": list(SELECTION_ENVIRONMENTS),
            "locked_confirmation_artifact": [CONFIRMATION_ENVIRONMENT],
            "confirmation_access": "valid locked-candidate sentinel required before load",
            "sealed_final_holdout": "managed outside developer API",
            "legacy_combined_and_final_artifacts": "must_be_absent_from_data_cache",
        },
        "provenance": {
            "source_sha256": dict(sorted(source_hashes.items())),
            "code_sha256": _implementation_hashes(project_root),
            "dependencies": _dependency_versions(),
        },
    }


def _source_hashes(root: Path, candidates: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in candidates:
        if path.exists():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    foundation_report = root / "data_cache" / "foundation_report.json"
    if foundation_report.exists():
        with foundation_report.open("r", encoding="utf-8") as handle:
            source_manifest_hash = json.load(handle).get("source_manifest_sha256")
        if source_manifest_hash:
            result["competition_source_manifest"] = str(source_manifest_hash)
    return result


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(path, lambda target: frame.to_parquet(target, index=False))


def _atomic_json(path: Path, value: object) -> None:
    _atomic_replace(
        path,
        lambda target: target.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
    )


def _read_existing_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot verify existing validation manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Existing validation manifest must be a JSON object.")
    return value


def _verified_legacy_artifact_path(cache_dir: Path, filename: str) -> Path:
    cache_root = cache_dir.resolve()
    candidate = (cache_root / filename).resolve()
    if candidate.parent != cache_root or candidate.name != filename:
        raise RuntimeError(
            f"Refusing legacy-artifact operation outside the exact cache root: {candidate}"
        )
    return candidate


def _remove_legacy_generated_artifacts(cache_dir: Path) -> list[str]:
    removed: list[str] = []
    for filename in (LEGACY_COMBINED_ASSIGNMENT_NAME, LEGACY_FINAL_ASSIGNMENT_NAME):
        path = _verified_legacy_artifact_path(cache_dir, filename)
        if not path.exists():
            continue
        if not path.is_file():
            raise RuntimeError(f"Legacy generated artifact is not a regular file: {path}")
        path.unlink()
        if path.exists():
            raise RuntimeError(f"Failed to remove legacy generated artifact: {path}")
        removed.append(f"data_cache/{filename}")
    return removed


def _guard_frozen_development_state(
    *,
    selection_path: Path,
    joint_confirmation_path: Path,
    legacy_combined_path: Path,
    legacy_final_path: Path,
    manifest_path: Path,
    proposed_manifest: Mapping[str, object],
    protocol_version_override: str | None,
) -> Mapping[str, object] | None:
    selection_exists = selection_path.exists()
    joint_exists = joint_confirmation_path.exists()
    legacy_exists = legacy_combined_path.exists()
    legacy_final_exists = legacy_final_path.exists()
    existing_manifest = _read_existing_manifest(manifest_path)
    if (
        not selection_exists
        and not joint_exists
        and not legacy_exists
        and not legacy_final_exists
        and existing_manifest is None
    ):
        return None

    changes: list[str] = []
    actual_selection_hash: str | None = None
    actual_selection_file_hash = sha256_file(selection_path) if selection_exists else None
    if selection_exists:
        try:
            # The normal developer boundary may inspect the selection artifact.
            actual_selection_hash = assignment_sha256(pd.read_parquet(selection_path))
        except Exception as exc:  # Integrity failures must never be silently overwritten.
            changes.append(f"existing selection assignment is unreadable ({type(exc).__name__})")
    else:
        changes.append("selection assignment is missing")
    if not joint_exists:
        changes.append("joint-confirmation assignment is missing")
    # Never deserialize the confirmation artifact here.  A raw file hash is
    # sufficient for freeze-integrity checks and does not cross its lock gate.
    actual_joint_file_hash = sha256_file(joint_confirmation_path) if joint_exists else None
    if legacy_exists:
        changes.append("legacy combined development artifact must be removed")
    if legacy_final_exists:
        changes.append("legacy sealed-final artifact must be removed")

    proposed_development = proposed_manifest.get("development", {})
    proposed_selection = (
        proposed_development.get("selection", {})
        if isinstance(proposed_development, Mapping)
        else {}
    )
    proposed_joint = (
        proposed_development.get("joint_confirmation", {})
        if isinstance(proposed_development, Mapping)
        else {}
    )
    proposed_selection_hash = (
        proposed_selection.get("assignment_sha256")
        if isinstance(proposed_selection, Mapping)
        else None
    )
    proposed_joint_hash = (
        proposed_joint.get("assignment_sha256")
        if isinstance(proposed_joint, Mapping)
        else None
    )
    if actual_selection_hash is not None and actual_selection_hash != proposed_selection_hash:
        changes.append("selection assignment SHA-256 changed")

    if existing_manifest is None:
        changes.append("validation manifest is missing")
        existing_version: str | None = None
        existing_development: Mapping[str, object] = {}
    else:
        raw_development = existing_manifest.get("development", {})
        existing_development = (
            raw_development if isinstance(raw_development, Mapping) else {}
        )
        existing_selection = existing_development.get("selection", {})
        existing_joint = existing_development.get("joint_confirmation", {})
        recorded_selection_hash = (
            existing_selection.get("assignment_sha256")
            if isinstance(existing_selection, Mapping)
            else None
        )
        recorded_joint_hash = (
            existing_joint.get("assignment_sha256")
            if isinstance(existing_joint, Mapping)
            else None
        )
        recorded_joint_file_hash = (
            existing_joint.get("file_sha256")
            if isinstance(existing_joint, Mapping)
            else None
        )
        recorded_selection_file_hash = (
            existing_selection.get("file_sha256")
            if isinstance(existing_selection, Mapping)
            else None
        )
        if recorded_selection_hash != actual_selection_hash:
            changes.append("manifest selection SHA-256 does not match the existing artifact")
        if recorded_selection_file_hash != actual_selection_file_hash:
            changes.append("manifest selection file SHA-256 mismatch")
        if recorded_selection_hash != proposed_selection_hash:
            changes.append("manifest selection assignment SHA-256 would change")
        if recorded_joint_hash != proposed_joint_hash:
            changes.append("manifest joint-confirmation assignment SHA-256 would change")
        if recorded_joint_file_hash != actual_joint_file_hash:
            changes.append("manifest joint-confirmation file SHA-256 mismatch")
        if existing_manifest.get("source_hashes") != proposed_manifest.get("source_hashes"):
            changes.append("source hashes changed")
        if existing_manifest.get("protocol_sha256") != proposed_manifest.get(
            "protocol_sha256"
        ):
            changes.append("protocol SHA-256 changed")
        raw_version = existing_manifest.get("protocol_version")
        if raw_version is None and isinstance(existing_manifest.get("protocol"), Mapping):
            raw_version = existing_manifest["protocol"].get("protocol_version")
        existing_version = str(raw_version) if raw_version is not None else None

    if not changes:
        if protocol_version_override is not None:
            raise ValueError(
                "A protocol-version override is unnecessary and must name a new version."
            )
        return None

    if protocol_version_override is None:
        raise RuntimeError(
            "Refusing to overwrite frozen development validation state: "
            + "; ".join(dict.fromkeys(changes))
            + ". Pass an explicit new protocol_version_override to acknowledge a versioned rebuild."
        )
    override = str(protocol_version_override).strip()
    if not override:
        raise ValueError("protocol_version_override cannot be empty.")
    if existing_version is not None and override == existing_version:
        raise RuntimeError(
            "protocol_version_override must name a new version, not the existing version."
        )

    prior_source_hashes = (
        existing_manifest.get("source_hashes", {}) if existing_manifest is not None else {}
    )
    prior_protocol_hash = (
        existing_manifest.get("protocol_sha256") if existing_manifest is not None else None
    )
    prior_development = (
        existing_manifest.get("development", {}) if existing_manifest is not None else {}
    )
    prior_recorded_hash = (
        prior_development.get(
            "aggregate_assignment_sha256", prior_development.get("assignment_sha256")
        )
        if isinstance(prior_development, Mapping)
        else None
    )
    return {
        "protocol_version": existing_version,
        "protocol_sha256": prior_protocol_hash,
        "development_aggregate_assignment_sha256": prior_recorded_hash,
        "actual_selection_assignment_sha256": actual_selection_hash,
        "actual_selection_file_sha256": actual_selection_file_hash,
        "actual_joint_confirmation_file_sha256": actual_joint_file_hash,
        "source_hashes_sha256": _canonical_json_sha256(prior_source_hashes),
        "change_reasons": list(dict.fromkeys(changes)),
        "prior_supersedes": (
            existing_manifest.get("supersedes") if existing_manifest is not None else None
        ),
    }


def _manifest_development_artifact(
    manifest: Mapping[str, object],
    artifact_name: str,
) -> Mapping[str, object]:
    development = manifest.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("Validation manifest has no development contract.")
    artifact = development.get(artifact_name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"Validation manifest has no {artifact_name} contract.")
    return artifact


def _validate_manifest_integrity(
    manifest: Mapping[str, object],
    *,
    project_root: Path,
) -> None:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Validation manifest has no protocol object.")
    if manifest.get("protocol_sha256") != _canonical_json_sha256(protocol):
        raise ValueError("Validation manifest protocol SHA-256 mismatch.")
    if manifest.get("protocol_version") != protocol.get("protocol_version"):
        raise ValueError("Validation manifest protocol versions disagree.")
    provenance = protocol.get("provenance")
    if not isinstance(provenance, Mapping) or manifest.get("provenance") != provenance:
        raise ValueError("Validation manifest provenance is inconsistent.")
    if provenance.get("source_sha256") != manifest.get("source_hashes"):
        raise ValueError("Validation manifest source hashes are inconsistent.")
    if provenance.get("code_sha256") != _implementation_hashes(project_root):
        raise ValueError(
            "Validation implementation changed since the manifest was frozen; "
            "perform an explicit versioned rebuild."
        )
    if provenance.get("dependencies") != _dependency_versions():
        raise ValueError(
            "Validation dependencies changed since the manifest was frozen; "
            "perform an explicit versioned rebuild."
        )
    final_policy = manifest.get("sealed_final_holdout")
    if not isinstance(final_policy, Mapping):
        raise ValueError("Validation manifest has no sealed-final policy.")
    if (
        final_policy.get("normal_developer_api_access") is not False
        or final_policy.get("assignment_metadata_exposed") is not False
        or final_policy.get("legacy_workspace_artifacts_absent") is not True
    ):
        raise PermissionError("Validation manifest violates the sealed-final boundary.")
    if {"assignment_path", "assignment_sha256", "seed"}.intersection(final_policy):
        raise PermissionError("Validation manifest exposes sealed-final assignment metadata.")
    for filename in (LEGACY_COMBINED_ASSIGNMENT_NAME, LEGACY_FINAL_ASSIGNMENT_NAME):
        if _verified_legacy_artifact_path(project_root / "data_cache", filename).exists():
            raise PermissionError(
                "A plaintext legacy validation artifact exists inside the developer workspace."
            )


def _load_frozen_assignment_artifact(
    *,
    path: Path,
    manifest_record: Mapping[str, object],
    expected_relative_path: str,
    expected_environments: tuple[str, ...],
) -> pd.DataFrame:
    if manifest_record.get("assignment_path") != expected_relative_path:
        raise ValueError("Validation manifest contains an unexpected assignment path.")
    if list(manifest_record.get("environments", [])) != list(expected_environments):
        raise ValueError("Validation manifest contains unexpected artifact environments.")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen validation artifact not found: {path}")
    if manifest_record.get("file_sha256") != sha256_file(path):
        raise ValueError("Frozen validation artifact file SHA-256 mismatch.")
    assignments = pd.read_parquet(path)
    missing = sorted(set(ASSIGNMENT_COLUMNS).difference(assignments.columns))
    if missing:
        raise ValueError(f"Frozen validation artifact is missing columns: {missing}")
    forbidden = FORBIDDEN_ASSIGNMENT_COLUMNS.intersection(assignments.columns)
    if forbidden:
        raise ValueError(f"Frozen assignments contain forbidden columns: {sorted(forbidden)}")
    if len(assignments) != int(manifest_record.get("row_count", -1)):
        raise ValueError("Frozen validation artifact row count mismatch.")
    if set(assignments["suite"].astype(str)) != {DEVELOPMENT_SUITE}:
        raise PermissionError("Only development assignments may cross this loader boundary.")
    if set(assignments["environment"].astype(str)) != set(expected_environments):
        raise ValueError("Frozen validation artifact has unknown or missing environments.")
    if set(assignments["schema_version"].astype(str)) != {SCHEMA_VERSION}:
        raise ValueError("Frozen validation artifact uses an unknown schema version.")
    split_seeds = _strict_integer_series(assignments["split_seed"], "split_seed")
    if set(split_seeds.astype(int)) != {DEVELOPMENT_SEED}:
        raise ValueError("Frozen validation artifact uses the wrong development seed.")
    for environment in expected_environments:
        subset = assignments.loc[assignments["environment"].eq(environment)]
        if subset["response_id"].astype(str).duplicated().any():
            raise ValueError(f"{environment} contains duplicate response IDs.")
    if assignment_sha256(assignments) != manifest_record.get("assignment_sha256"):
        raise ValueError("Frozen validation artifact assignment SHA-256 mismatch.")

    families = _strict_integer_series(assignments["semantic_family"], "semantic_family")
    styles = _strict_integer_series(assignments["style_cell"], "style_cell")
    canonical_joint = np.asarray(
        [
            f"family_{family:03d}__style_{style:03d}"
            for family, style in zip(families, styles)
        ],
        dtype=str,
    )
    if not np.array_equal(assignments["joint_cell"].astype(str), canonical_joint):
        raise ValueError("Frozen validation artifact has non-canonical joint_cell values.")
    family_counts = assignments.assign(semantic_family=families).groupby(
        assignments["learning_objective_id"].astype(str), sort=True
    )["semantic_family"].nunique(dropna=False)
    if family_counts.ne(1).any():
        raise ValueError("Frozen semantic_family is not constant per objective.")
    style_counts = assignments.assign(style_cell=styles).groupby(
        assignments["session_id"].astype(str), sort=True
    )["style_cell"].nunique(dropna=False)
    if style_counts.ne(1).any():
        raise ValueError("Frozen style_cell is not constant per session.")
    return assignments


def load_development_selection_assignments(
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load only the three pre-lock selection environments."""

    paths = discover_project_paths(project_root)
    manifest_path = paths.cache_dir / "validation_environments.manifest.json"
    manifest = _read_existing_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError("Build the frozen development environments first.")
    _validate_manifest_integrity(manifest, project_root=paths.root)
    selection_record = _manifest_development_artifact(manifest, "selection")
    assignments = _load_frozen_assignment_artifact(
        path=paths.root / SELECTION_ASSIGNMENT_PATH,
        manifest_record=selection_record,
        expected_relative_path=SELECTION_ASSIGNMENT_PATH,
        expected_environments=SELECTION_ENVIRONMENTS,
    )
    # Do not return confirmation location or hashes through the normal loader.
    sanitized_manifest = dict(manifest)
    sanitized_manifest.pop("supersedes", None)
    development = dict(manifest["development"])
    development.pop("joint_confirmation", None)
    development["confirmation_access"] = "locked_candidate_sentinel_required"
    sanitized_manifest["development"] = development
    return assignments, sanitized_manifest


def _verify_locked_candidate_sentinel(
    sentinel_path: Path,
    *,
    aggregate_assignment_sha256: object,
) -> dict[str, object]:
    if not sentinel_path.is_file():
        raise FileNotFoundError("Locked-candidate sentinel not found.")
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Locked-candidate sentinel is unreadable: {exc}") from exc
    if not isinstance(sentinel, dict):
        raise ValueError("Locked-candidate sentinel must be a JSON object.")
    payload = dict(sentinel)
    observed = payload.pop("lock_sha256", None)
    if observed != _canonical_json_sha256(payload):
        raise ValueError("Locked-candidate sentinel hash verification failed.")
    if payload.get("development_assignment_sha256") != aggregate_assignment_sha256:
        raise ValueError("Locked-candidate sentinel references different assignments.")
    if list(payload.get("selection_environments", [])) != list(SELECTION_ENVIRONMENTS):
        raise ValueError("Locked-candidate sentinel has unexpected selection environments.")
    if payload.get("confirmation_environment") != CONFIRMATION_ENVIRONMENT:
        raise ValueError("Locked-candidate sentinel has an unexpected confirmation environment.")
    if payload.get("V_final_accessed") is not False:
        raise ValueError("Locked-candidate sentinel indicates forbidden final-holdout access.")
    if not str(payload.get("candidate", "")).strip() or not str(
        payload.get("calibration", "")
    ).strip():
        raise ValueError("Locked-candidate sentinel does not identify a locked decision.")
    return sentinel


def load_joint_confirmation_assignments(
    project_root: str | Path,
    *,
    locked_candidate_sentinel: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load V_joint only after a self-authenticating candidate-lock sentinel."""

    paths = discover_project_paths(project_root)
    manifest_path = paths.cache_dir / "validation_environments.manifest.json"
    manifest = _read_existing_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError("Build the frozen development environments first.")
    _validate_manifest_integrity(manifest, project_root=paths.root)
    development = manifest.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("Validation manifest has no development contract.")
    _verify_locked_candidate_sentinel(
        Path(locked_candidate_sentinel),
        aggregate_assignment_sha256=development.get("aggregate_assignment_sha256"),
    )
    joint_record = _manifest_development_artifact(manifest, "joint_confirmation")
    assignments = _load_frozen_assignment_artifact(
        path=paths.root / JOINT_CONFIRMATION_ASSIGNMENT_PATH,
        manifest_record=joint_record,
        expected_relative_path=JOINT_CONFIRMATION_ASSIGNMENT_PATH,
        expected_environments=(CONFIRMATION_ENVIRONMENT,),
    )
    return assignments, manifest


def run_validation_environment_build(
    project_root: str | Path,
    *,
    n_splits: int = 5,
    n_semantic_families: int = 50,
    n_style_cells: int = 20,
    min_validation_rows: int = 100,
    min_class_rows: int = 10,
    protocol_version_override: str | None = None,
) -> dict[str, object]:
    """Build and freeze development environments without accessing V_final."""

    effective_protocol_version = (
        str(protocol_version_override).strip()
        if protocol_version_override is not None
        else PROTOCOL_VERSION
    )
    if not effective_protocol_version:
        raise ValueError("protocol_version_override cannot be empty.")
    paths = discover_project_paths(project_root)
    modeling_path = paths.cache_dir / "modeling_base.parquet"
    embeddings_path = paths.cache_dir / "bge_small_objective_256.npy"
    behavior_path = paths.cache_dir / "session_behavior.parquet"
    if not modeling_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("Run the foundation and BGE-small cache stages first.")
    frame = pd.read_parquet(modeling_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    behavior = pd.read_parquet(behavior_path) if behavior_path.exists() else None

    try:
        development = build_validation_environments(
            frame,
            embeddings,
            seed=DEVELOPMENT_SEED,
            suite_name=DEVELOPMENT_SUITE,
            n_splits=n_splits,
            n_semantic_families=n_semantic_families,
            n_style_cells=n_style_cells,
            session_behavior=behavior,
        )
    finally:
        memory_map = getattr(embeddings, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
    # Provisioning verifies all four assignment structures without labels.
    # Only the three selection environments receive development label diagnostics;
    # V_joint remains an untouched confirmation artifact until candidate lock.
    validate_environment_assignments(
        frame.drop(columns=["target"], errors="ignore"),
        development,
        n_splits=n_splits,
        min_validation_rows=min_validation_rows,
        include_label_diagnostics=False,
    )
    development_diagnostics = validate_environment_assignments(
        frame,
        development,
        n_splits=n_splits,
        min_validation_rows=min_validation_rows,
        min_class_rows=min_class_rows,
        include_label_diagnostics=True,
        diagnostic_environments=SELECTION_ENVIRONMENTS,
    )

    selection = development.loc[
        development["environment"].isin(SELECTION_ENVIRONMENTS)
    ].reset_index(drop=True)
    joint_confirmation = development.loc[
        development["environment"].eq(CONFIRMATION_ENVIRONMENT)
    ].reset_index(drop=True)
    selection_path = paths.root / SELECTION_ASSIGNMENT_PATH
    joint_confirmation_path = paths.root / JOINT_CONFIRMATION_ASSIGNMENT_PATH
    legacy_combined_path = _verified_legacy_artifact_path(
        paths.cache_dir, LEGACY_COMBINED_ASSIGNMENT_NAME
    )
    legacy_final_path = _verified_legacy_artifact_path(
        paths.cache_dir, LEGACY_FINAL_ASSIGNMENT_NAME
    )
    manifest_path = paths.cache_dir / "validation_environments.manifest.json"
    source_hashes = _source_hashes(
        paths.root,
        [modeling_path, embeddings_path, behavior_path, paths.cache_dir / "raw_manifest.csv"],
    )
    manifest = build_validation_manifest(
        development_assignments=development,
        source_hashes=source_hashes,
        development_diagnostics=development_diagnostics,
        n_splits=n_splits,
        n_semantic_families=n_semantic_families,
        n_style_cells=n_style_cells,
        min_validation_rows=min_validation_rows,
        min_class_rows=min_class_rows,
        session_behavior_used=behavior is not None,
        protocol_version=effective_protocol_version,
        project_root=paths.root,
    )
    supersedes = _guard_frozen_development_state(
        selection_path=selection_path,
        joint_confirmation_path=joint_confirmation_path,
        legacy_combined_path=legacy_combined_path,
        legacy_final_path=legacy_final_path,
        manifest_path=manifest_path,
        proposed_manifest=manifest,
        protocol_version_override=protocol_version_override,
    )
    if supersedes is None:
        existing_manifest = _read_existing_manifest(manifest_path)
        if existing_manifest is not None:
            _validate_manifest_integrity(existing_manifest, project_root=paths.root)
            existing_development = existing_manifest.get("development")
            if not isinstance(existing_development, Mapping):
                raise ValueError("Existing manifest has no development contract.")
            existing_selection = existing_development.get("selection")
            if not isinstance(existing_selection, Mapping):
                raise ValueError("Existing manifest has no selection contract.")
            # The guard already verified source, protocol, assignment, and raw
            # artifact hashes.  A frozen identical build is a literal no-op so
            # generated_at, provenance history, parquet bytes, and mtimes cannot drift.
            return {
                "selection_path": str(selection_path),
                "selection_assignment_sha256": existing_selection[
                    "assignment_sha256"
                ],
                "development_assignment_sha256": existing_development[
                    "aggregate_assignment_sha256"
                ],
                "manifest_path": str(manifest_path),
                "protocol_version": existing_manifest["protocol_version"],
                "protocol_sha256": existing_manifest["protocol_sha256"],
                "V_final_accessed": False,
            }
    if supersedes is not None:
        manifest["supersedes"] = dict(supersedes)
    _atomic_parquet(selection_path, selection)
    _atomic_parquet(joint_confirmation_path, joint_confirmation)
    development_record = manifest["development"]
    assert isinstance(development_record, dict)
    selection_record = development_record["selection"]
    joint_record = development_record["joint_confirmation"]
    assert isinstance(selection_record, dict) and isinstance(joint_record, dict)
    selection_record["file_sha256"] = sha256_file(selection_path)
    joint_record["file_sha256"] = sha256_file(joint_confirmation_path)
    if protocol_version_override is not None:
        removed = _remove_legacy_generated_artifacts(paths.cache_dir)
        manifest["legacy_artifact_cleanup"] = {
            "explicit_protocol_version": effective_protocol_version,
            "verified_cache_root": paths.cache_dir.resolve().as_posix(),
            "removed": removed,
            "legacy_artifacts_absent": True,
        }
    else:
        existing_manifest = _read_existing_manifest(manifest_path)
        if existing_manifest is not None and "legacy_artifact_cleanup" in existing_manifest:
            manifest["legacy_artifact_cleanup"] = existing_manifest[
                "legacy_artifact_cleanup"
            ]
    if legacy_combined_path.exists() or legacy_final_path.exists():
        raise RuntimeError("Legacy validation artifacts remain after the build boundary.")
    _atomic_json(manifest_path, manifest)
    return {
        "selection_path": str(selection_path),
        "selection_assignment_sha256": selection_record["assignment_sha256"],
        "development_assignment_sha256": development_record[
            "aggregate_assignment_sha256"
        ],
        "manifest_path": str(manifest_path),
        "protocol_version": manifest["protocol_version"],
        "protocol_sha256": manifest["protocol_sha256"],
        "V_final_accessed": False,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen Trace the Ace development validation environments. "
            "The sealed final holdout is outside this command's API."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--semantic-families", type=int, default=50)
    parser.add_argument("--style-cells", type=int, default=20)
    parser.add_argument("--min-validation-rows", type=int, default=100)
    parser.add_argument("--min-class-rows", type=int, default=10)
    parser.add_argument(
        "--protocol-version-override",
        help=(
            "Explicit new protocol version required before replacing frozen state "
            "whose assignment, source, or protocol hashes changed."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_validation_environment_build(
        args.project_root,
        n_splits=args.n_splits,
        n_semantic_families=args.semantic_families,
        n_style_cells=args.style_cells,
        min_validation_rows=args.min_validation_rows,
        min_class_rows=args.min_class_rows,
        protocol_version_override=args.protocol_version_override,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
