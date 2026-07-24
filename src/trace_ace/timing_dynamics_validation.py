from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.foundation import sha256_file
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


PROTOCOL_ID = "E510_timing_dynamics_v1"
SEED = 20260724
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
DEVELOPMENT_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
JOINT_ENVIRONMENT = "V_joint"
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
EXPECTED_UTTERANCE_SHA256 = (
    "80a231d18dbb0641989ebb972f08988f0ddd3cb7c4fb769ad2fe90b5a98ba5a4"
)
EXPECTED_COMPONENT_SHA256 = {
    "V_seen": "1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af",
    "V_objective": "c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7",
    "V_style": "6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038",
    "V_joint": "f09b522f71011aa8129121b560396f28c2f133c23c8cfb88eac2da7311fdfaab",
}
TIMING_FEATURE_NAMES = (
    "timing_distinct_time_share",
    "timing_zero_gap_share",
    "timing_positive_gap_mean_log",
    "timing_positive_gap_median_log",
    "timing_positive_gap_p90_log",
    "timing_positive_gap_p95_log",
    "timing_positive_gap_p99_log",
    "timing_positive_gap_max_log",
    "timing_positive_gap_std_log",
    "timing_gap_over_10_share",
    "timing_gap_over_30_share",
    "timing_tutor_student_mean_log",
    "timing_tutor_student_median_log",
    "timing_tutor_student_p90_log",
    "timing_tutor_student_zero_share",
    "timing_student_tutor_mean_log",
    "timing_student_tutor_median_log",
    "timing_student_tutor_p90_log",
    "timing_student_tutor_zero_share",
    "timing_transition_median_asymmetry",
    "timing_early_gap_median_log",
    "timing_late_gap_median_log",
    "timing_gap_median_log_delta",
    "timing_early_tutor_student_median_log",
    "timing_late_tutor_student_median_log",
    "timing_tutor_student_median_log_delta",
    "timing_turns_per_minute_log",
    "timing_student_turns_per_minute_log",
    "timing_tutor_turns_per_minute_log",
    "timing_background_turns_per_minute_log",
    "timing_gap_bin_entropy",
)
PROBABILITY_CLIP = 1e-6
DEFAULT_BOOTSTRAP_REPLICATES = 5_000


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
    _atomic_replace(
        path,
        lambda target: target.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(
        path, lambda target: frame.to_csv(target, index=False, lineterminator="\n")
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    _atomic_replace(path, lambda target: frame.to_parquet(target, index=False))


def _timestamp_seconds(values: pd.Series) -> np.ndarray:
    text = values.astype(str)
    valid = text.str.fullmatch(r"\d{2}:\d{2}:\d{2}")
    if not bool(valid.all()):
        raise ValueError("E510 requires complete HH:MM:SS timestamps.")
    parts = text.str.split(":", expand=True).astype(np.int32)
    seconds = (
        parts.iloc[:, 0] * 3600 + parts.iloc[:, 1] * 60 + parts.iloc[:, 2]
    ).to_numpy(dtype=np.int32)
    return seconds


def _positive_log_stat(values: np.ndarray, statistic: str) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    if not len(positive):
        return 0.0
    if statistic == "mean":
        result = float(positive.mean())
    elif statistic == "median":
        result = float(np.median(positive))
    elif statistic == "p90":
        result = float(np.quantile(positive, 0.90))
    elif statistic == "p95":
        result = float(np.quantile(positive, 0.95))
    elif statistic == "p99":
        result = float(np.quantile(positive, 0.99))
    elif statistic == "max":
        result = float(positive.max())
    elif statistic == "std":
        result = float(positive.std(ddof=0))
    else:
        raise ValueError(f"Unknown timing statistic: {statistic}")
    return float(np.log1p(result))


def _transition_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 0.0, 0.0, 0.0, 0.0
    return (
        _positive_log_stat(values, "mean"),
        _positive_log_stat(values, "median"),
        _positive_log_stat(values, "p90"),
        float(np.mean(values == 0)),
    )


def _gap_bin_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        return 0.0
    bin_id = np.select(
        [
            values == 0,
            values == 1,
            values == 2,
            (values >= 3) & (values <= 5),
            (values >= 6) & (values <= 10),
            (values >= 11) & (values <= 30),
        ],
        [0, 1, 2, 3, 4, 5],
        default=6,
    )
    counts = np.bincount(bin_id, minlength=7).astype(np.float64)
    probability = counts[counts > 0] / counts.sum()
    return float(-(probability * np.log(probability)).sum() / np.log(7.0))


def session_timing_feature_row(session: pd.DataFrame) -> dict[str, float | str]:
    required = {"session_id", "utterance_id", "timestamp", "role"}
    missing = sorted(required.difference(session.columns))
    if missing:
        raise ValueError(f"Timing input is missing columns: {missing}")
    session_ids = session["session_id"].astype(str).unique()
    if len(session_ids) != 1:
        raise ValueError("One session is required per timing feature row.")
    ordered = session.sort_values("utterance_id", kind="mergesort")
    seconds = _timestamp_seconds(ordered["timestamp"])
    if np.any(np.diff(seconds) < 0):
        raise ValueError(f"Non-monotonic timestamps in session {session_ids[0]}.")
    roles = ordered["role"].fillna("").astype(str).str.lower().to_numpy()
    gaps = np.diff(seconds).astype(np.int32)
    previous = roles[:-1]
    current = roles[1:]
    tutor_student = gaps[(previous == "tutor") & (current == "student")]
    student_tutor = gaps[(previous == "student") & (current == "tutor")]
    tutor_student_summary = _transition_summary(tutor_student)
    student_tutor_summary = _transition_summary(student_tutor)
    split = len(gaps) // 2
    early = gaps[:split]
    late = gaps[split:]
    early_tutor_student = tutor_student
    late_tutor_student = tutor_student
    if len(gaps):
        positions = np.arange(len(gaps))
        tutor_student_mask = (previous == "tutor") & (current == "student")
        early_tutor_student = gaps[tutor_student_mask & (positions < split)]
        late_tutor_student = gaps[tutor_student_mask & (positions >= split)]
    early_gap_median = _positive_log_stat(early, "median")
    late_gap_median = _positive_log_stat(late, "median")
    early_tutor_student_median = _positive_log_stat(
        early_tutor_student, "median"
    )
    late_tutor_student_median = _positive_log_stat(late_tutor_student, "median")
    duration_minutes = max(float(seconds[-1] - seconds[0]) / 60.0, 1.0 / 60.0)
    n_utterances = len(ordered)
    result: dict[str, float | str] = {
        "session_id": str(session_ids[0]),
        "timing_distinct_time_share": float(
            len(np.unique(seconds)) / max(n_utterances, 1)
        ),
        "timing_zero_gap_share": float(np.mean(gaps == 0)) if len(gaps) else 0.0,
        "timing_positive_gap_mean_log": _positive_log_stat(gaps, "mean"),
        "timing_positive_gap_median_log": _positive_log_stat(gaps, "median"),
        "timing_positive_gap_p90_log": _positive_log_stat(gaps, "p90"),
        "timing_positive_gap_p95_log": _positive_log_stat(gaps, "p95"),
        "timing_positive_gap_p99_log": _positive_log_stat(gaps, "p99"),
        "timing_positive_gap_max_log": _positive_log_stat(gaps, "max"),
        "timing_positive_gap_std_log": _positive_log_stat(gaps, "std"),
        "timing_gap_over_10_share": float(np.mean(gaps > 10)) if len(gaps) else 0.0,
        "timing_gap_over_30_share": float(np.mean(gaps > 30)) if len(gaps) else 0.0,
        "timing_tutor_student_mean_log": tutor_student_summary[0],
        "timing_tutor_student_median_log": tutor_student_summary[1],
        "timing_tutor_student_p90_log": tutor_student_summary[2],
        "timing_tutor_student_zero_share": tutor_student_summary[3],
        "timing_student_tutor_mean_log": student_tutor_summary[0],
        "timing_student_tutor_median_log": student_tutor_summary[1],
        "timing_student_tutor_p90_log": student_tutor_summary[2],
        "timing_student_tutor_zero_share": student_tutor_summary[3],
        "timing_transition_median_asymmetry": (
            tutor_student_summary[1] - student_tutor_summary[1]
        ),
        "timing_early_gap_median_log": early_gap_median,
        "timing_late_gap_median_log": late_gap_median,
        "timing_gap_median_log_delta": late_gap_median - early_gap_median,
        "timing_early_tutor_student_median_log": early_tutor_student_median,
        "timing_late_tutor_student_median_log": late_tutor_student_median,
        "timing_tutor_student_median_log_delta": (
            late_tutor_student_median - early_tutor_student_median
        ),
        "timing_turns_per_minute_log": float(
            np.log1p(n_utterances / duration_minutes)
        ),
        "timing_student_turns_per_minute_log": float(
            np.log1p(np.sum(roles == "student") / duration_minutes)
        ),
        "timing_tutor_turns_per_minute_log": float(
            np.log1p(np.sum(roles == "tutor") / duration_minutes)
        ),
        "timing_background_turns_per_minute_log": float(
            np.log1p(np.sum(roles == "background") / duration_minutes)
        ),
        "timing_gap_bin_entropy": _gap_bin_entropy(gaps),
    }
    return result


def build_session_timing_features(utterances: pd.DataFrame) -> pd.DataFrame:
    required = {"session_id", "utterance_id", "timestamp", "role"}
    missing = sorted(required.difference(utterances.columns))
    if missing:
        raise ValueError(f"Utterance cache is missing columns: {missing}")
    rows = [
        session_timing_feature_row(group)
        for _, group in utterances.groupby("session_id", sort=True)
    ]
    result = pd.DataFrame(rows)
    expected = ["session_id", *TIMING_FEATURE_NAMES]
    if list(result.columns) != expected:
        raise RuntimeError("E510 timing feature schema changed.")
    if result["session_id"].duplicated().any():
        raise RuntimeError("E510 created duplicate session rows.")
    values = result[list(TIMING_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("E510 timing features contain non-finite values.")
    return result


def build_or_load_timing_cache(project_root: str | Path) -> tuple[pd.DataFrame, dict]:
    paths = discover_project_paths(project_root)
    source = paths.cache_dir / "utterances.parquet"
    source_sha = sha256_file(source)
    if source_sha != EXPECTED_UTTERANCE_SHA256:
        raise ValueError("E510 utterance source SHA-256 changed.")
    cache_path = paths.cache_dir / "session_timing_e510.parquet"
    metadata_path = paths.cache_dir / "session_timing_e510.metadata.json"
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol_id") != PROTOCOL_ID
            or metadata.get("source_sha256") != source_sha
            or metadata.get("feature_names") != list(TIMING_FEATURE_NAMES)
            or metadata.get("cache_sha256") != sha256_file(cache_path)
        ):
            raise ValueError("Existing E510 timing cache metadata is stale.")
        frame = pd.read_parquet(cache_path)
        return frame, metadata
    utterances = pd.read_parquet(
        source, columns=["session_id", "utterance_id", "timestamp", "role"]
    )
    frame = build_session_timing_features(utterances)
    _atomic_parquet(cache_path, frame)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "source": source.name,
        "source_sha256": source_sha,
        "sessions": int(len(frame)),
        "feature_names": list(TIMING_FEATURE_NAMES),
        "target_used": False,
        "cache_sha256": sha256_file(cache_path),
    }
    _atomic_json(metadata_path, metadata)
    return frame, metadata


def _component_path(project_root: str | Path, environment: str) -> Path:
    if environment not in EXPECTED_COMPONENT_SHA256:
        raise ValueError(f"Unknown E510 environment: {environment}")
    return (
        Path(project_root).resolve()
        / "experiments"
        / "runs"
        / SOURCE_RUN_ID
        / f"component_oof_{environment}.parquet"
    )


def load_component_oof(project_root: str | Path, environment: str) -> pd.DataFrame:
    path = _component_path(project_root, environment)
    if sha256_file(path) != EXPECTED_COMPONENT_SHA256[environment]:
        raise ValueError(f"Frozen component OOF changed for {environment}.")
    frame = pd.read_parquet(path)
    required = {
        "environment",
        "response_id",
        "session_id",
        "learning_objective_id",
        "semantic_family",
        "fold",
        "evaluation_eligible",
        "target",
        "pred_full",
        "pred_role",
        "pred_bge_small",
        "pred_bge_base",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{environment} component OOF is missing: {missing}")
    if set(frame["environment"].astype(str)) != {environment}:
        raise ValueError(f"Component OOF environment mismatch for {environment}.")
    if frame["response_id"].duplicated().any() or set(frame["fold"]) != set(range(5)):
        raise ValueError(f"Invalid component OOF rows for {environment}.")
    return frame.reset_index(drop=True)


def align_timing_features(
    component: pd.DataFrame, timing: pd.DataFrame
) -> np.ndarray:
    if timing["session_id"].duplicated().any():
        raise ValueError("Timing cache contains duplicate sessions.")
    aligned = component[["session_id"]].merge(
        timing, on="session_id", how="left", validate="many_to_one", sort=False
    )
    if aligned[list(TIMING_FEATURE_NAMES)].isna().any().any():
        raise ValueError("At least one component session lacks E510 timing features.")
    values = aligned[list(TIMING_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Aligned E510 timing matrix contains non-finite values.")
    return values


def fit_timing_oof(component: pd.DataFrame, timing_values: np.ndarray) -> np.ndarray:
    if timing_values.shape != (len(component), len(TIMING_FEATURE_NAMES)):
        raise ValueError("E510 timing matrix shape mismatch.")
    folds = component["fold"].to_numpy(dtype=np.int8)
    target = component["target"].to_numpy(dtype=np.int8)
    prediction = np.full(len(component), np.nan, dtype=np.float64)
    for fold in range(5):
        train_mask, validation_mask = _fold_masks(component, folds, fold)
        training_sessions = set(component.loc[train_mask, "session_id"].astype(str))
        validation_sessions = set(
            component.loc[validation_mask, "session_id"].astype(str)
        )
        if training_sessions.intersection(validation_sessions):
            raise RuntimeError(f"E510 session leakage in fold {fold}.")
        scaler = StandardScaler()
        train_values = scaler.fit_transform(timing_values[train_mask])
        validation_values = scaler.transform(timing_values[validation_mask])
        model = LogisticRegression(
            C=REGULARIZATION_C,
            solver="liblinear",
            max_iter=1_000,
            random_state=SEED,
        )
        model.fit(train_values, target[train_mask])
        prediction[validation_mask] = model.predict_proba(validation_values)[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("E510 timing probe produced invalid OOF predictions.")
    return np.clip(prediction, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)


def build_prediction_rows(
    component: pd.DataFrame,
    timing_prediction: np.ndarray,
    weights: Sequence[float],
) -> pd.DataFrame:
    pred_bge = (
        0.25 * component["pred_full"].to_numpy(dtype=np.float64)
        + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
        + 0.50 * component["pred_bge_base"].to_numpy(dtype=np.float64)
    )
    pred_v02 = (
        0.25 * component["pred_full"].to_numpy(dtype=np.float64)
        + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
        + 0.50 * component["pred_bge_small"].to_numpy(dtype=np.float64)
    )
    metadata = component[
        [
            "environment",
            "response_id",
            "session_id",
            "learning_objective_id",
            "semantic_family",
            "fold",
            "evaluation_eligible",
            "target",
        ]
    ].copy()
    rows: list[pd.DataFrame] = []
    for weight in weights:
        candidate = metadata.copy()
        candidate["timing_weight"] = float(weight)
        candidate["pred_timing"] = timing_prediction
        candidate["pred_bge_replace"] = pred_bge
        candidate["pred_v02"] = pred_v02
        candidate["prediction"] = np.clip(
            (1.0 - weight) * pred_bge + weight * timing_prediction,
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        rows.append(candidate)
    return pd.concat(rows, ignore_index=True)


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for (environment, weight, fold), group in predictions.groupby(
        ["environment", "timing_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        candidate = binary_metrics(target, scored["prediction"])
        bge = binary_metrics(target, scored["pred_bge_replace"])
        v02 = binary_metrics(target, scored["pred_v02"])
        row: dict[str, object] = {
            "environment": environment,
            "timing_weight": float(weight),
            "fold": int(fold),
            "rows": int(len(scored)),
        }
        for name, value in candidate.items():
            row[name] = value
            row[f"delta_{name}_vs_bge"] = value - bge[name]
            row[f"delta_{name}_vs_v02"] = value - v02[name]
            row[f"bge_{name}"] = bge[name]
            row[f"v02_{name}"] = v02[name]
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    metric_columns = [
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "prediction_mean",
    ]
    aggregate_columns = [
        *metric_columns,
        *[f"delta_{name}_vs_bge" for name in metric_columns],
        *[f"delta_{name}_vs_v02" for name in metric_columns],
        *[f"bge_{name}" for name in metric_columns],
        *[f"v02_{name}" for name in metric_columns],
    ]
    environments = (
        folds.groupby(["environment", "timing_weight"], as_index=False)[
            aggregate_columns
        ]
        .mean()
        .sort_values(["environment", "timing_weight"], kind="mergesort")
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def _loss_rows(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )
    target = np.asarray(target, dtype=np.float64)
    return -(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))


def _macro_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    group_column: str,
    n_replicates: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    environment_samples: list[np.ndarray] = []
    observed: list[float] = []
    for environment, group in predictions.groupby("environment", sort=True):
        scored = group.loc[group["evaluation_eligible"].astype(bool)].copy()
        gain = _loss_rows(
            scored["target"].to_numpy(), scored[baseline_column].to_numpy()
        ) - _loss_rows(
            scored["target"].to_numpy(), scored[candidate_column].to_numpy()
        )
        grouped = (
            pd.DataFrame(
                {
                    "group": scored[group_column].astype(str).to_numpy(),
                    "gain": gain,
                }
            )
            .groupby("group", sort=True)["gain"]
            .agg(["sum", "count"])
        )
        sums = grouped["sum"].to_numpy(dtype=np.float64)
        counts = grouped["count"].to_numpy(dtype=np.float64)
        replicate = np.empty(n_replicates, dtype=np.float64)
        batch_size = 64
        for start in range(0, n_replicates, batch_size):
            stop = min(start + batch_size, n_replicates)
            sample = rng.integers(
                0, len(grouped), size=(stop - start, len(grouped))
            )
            replicate[start:stop] = (
                sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
            )
        environment_samples.append(replicate)
        observed.append(float(sums.sum() / counts.sum()))
    macro = np.vstack(environment_samples).mean(axis=0)
    return {
        "baseline": baseline_column,
        "group_column": group_column,
        "environments": int(len(environment_samples)),
        "replicates": int(n_replicates),
        "observed_mean_gain": float(np.mean(observed)),
        "bootstrap_mean_gain": float(macro.mean()),
        "ci_lower": float(np.quantile(macro, 0.025)),
        "ci_upper": float(np.quantile(macro, 0.975)),
        "support_positive_gain": float(np.mean(macro > 0.0)),
    }


def development_selection(
    predictions: pd.DataFrame,
    environment_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, object]] = []
    bootstraps: dict[float, dict] = {}
    for index, weight in enumerate(BLEND_WEIGHTS):
        metrics = environment_metrics.loc[
            environment_metrics["timing_weight"].eq(weight)
        ]
        selected_predictions = predictions.loc[
            predictions["timing_weight"].eq(weight)
        ]
        bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_bge_replace",
            group_column="session_id",
            n_replicates=n_bootstrap,
            seed=SEED + index,
        )
        bootstraps[weight] = bootstrap
        delta_loss = metrics["delta_log_loss_vs_bge"].to_numpy(dtype=np.float64)
        row = {
            "timing_weight": weight,
            "mean_log_loss": float(metrics["log_loss"].mean()),
            "mean_log_loss_gain_vs_bge": float(-delta_loss.mean()),
            "improved_environments": int(np.sum(delta_loss < 0.0)),
            "worst_environment_delta_log_loss_vs_bge": float(delta_loss.max()),
            "mean_delta_roc_auc_vs_bge": float(
                metrics["delta_roc_auc_vs_bge"].mean()
            ),
            "mean_delta_brier_score_vs_bge": float(
                metrics["delta_brier_score_vs_bge"].mean()
            ),
            "mean_delta_ece_10_vs_bge": float(
                metrics["delta_ece_10_vs_bge"].mean()
            ),
            "session_bootstrap_support": bootstrap["support_positive_gain"],
        }
        rows.append(row)
    selection = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "timing_weight"], kind="mergesort"
    )
    selected = selection.iloc[0].to_dict()
    clauses = {
        "mean_gain_at_least_0_00075": (
            selected["mean_log_loss_gain_vs_bge"] >= 0.00075
        ),
        "improves_at_least_two_environments": selected["improved_environments"] >= 2,
        "worst_environment_regression_at_most_0_0005": (
            selected["worst_environment_delta_log_loss_vs_bge"] <= 0.0005
        ),
        "macro_auc_non_regression": selected["mean_delta_roc_auc_vs_bge"] >= 0.0,
        "macro_brier_non_regression": (
            selected["mean_delta_brier_score_vs_bge"] <= 0.0
        ),
        "macro_ece_non_regression": selected["mean_delta_ece_10_vs_bge"] <= 0.0,
        "session_bootstrap_support_at_least_0_90": (
            selected["session_bootstrap_support"] >= 0.90
        ),
    }
    decision = {
        "selected_weight": float(selected["timing_weight"]),
        "selected_row": selected,
        "clauses": clauses,
        "passes_development_gate": bool(all(clauses.values())),
        "bootstrap": bootstraps[float(selected["timing_weight"])],
        "V_final_accessed": False,
    }
    return selection.reset_index(drop=True), decision


def _runtime_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_timing_dynamics")


def run_development(
    project_root: str | Path,
    *,
    run_id: str | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict:
    root = Path(project_root).resolve()
    run_id = run_id or _new_run_id()
    run_dir = root / "experiments" / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"E510 run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    timing, timing_metadata = build_or_load_timing_cache(root)
    prediction_frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for environment in DEVELOPMENT_ENVIRONMENTS:
        print(f"E510 development: {environment}", flush=True)
        component = load_component_oof(root, environment)
        source_hashes[environment] = EXPECTED_COMPONENT_SHA256[environment]
        timing_values = align_timing_features(component, timing)
        timing_prediction = fit_timing_oof(component, timing_values)
        prediction_frames.append(
            build_prediction_rows(component, timing_prediction, BLEND_WEIGHTS)
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics, environment_metrics = _metric_tables(predictions)
    selection, gate = development_selection(
        predictions, environment_metrics, n_bootstrap=n_bootstrap
    )
    _atomic_parquet(run_dir / "development_predictions.parquet", predictions)
    _atomic_csv(run_dir / "development_fold_metrics.csv", fold_metrics)
    _atomic_csv(run_dir / "development_environment_metrics.csv", environment_metrics)
    _atomic_csv(run_dir / "development_selection.csv", selection)
    _atomic_json(run_dir / "development_gate.json", gate)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "stage": "development",
        "status": (
            "passed_development_gate"
            if gate["passes_development_gate"]
            else "rejected_development_gate"
        ),
        "run_id": run_id,
        "source_run_id": SOURCE_RUN_ID,
        "source_component_sha256": source_hashes,
        "timing_cache_sha256": timing_metadata["cache_sha256"],
        "timing_source_sha256": timing_metadata["source_sha256"],
        "feature_names": list(TIMING_FEATURE_NAMES),
        "probe": {
            "regularization_c": REGULARIZATION_C,
            "solver": "liblinear",
            "max_iter": 1_000,
            "seed": SEED,
            "fold_local_standardization": True,
            "session_purge": True,
        },
        "blend_weights": list(BLEND_WEIGHTS),
        "development_gate": gate,
        "runtime": _runtime_metadata(),
        "artifact_hashes": artifact_hashes,
        "V_final_accessed": False,
    }
    _atomic_json(run_dir / "report.json", report)
    if gate["passes_development_gate"]:
        lock = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "selected_weight": gate["selected_weight"],
            "development_predictions_sha256": artifact_hashes[
                "development_predictions.parquet"
            ],
            "development_gate_sha256": artifact_hashes["development_gate.json"],
            "source_component_sha256": source_hashes,
            "timing_cache_sha256": timing_metadata["cache_sha256"],
            "V_final_accessed": False,
        }
        lock["lock_sha256"] = hashlib.sha256(
            json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _atomic_json(run_dir / "locked_candidate.json", lock)
    return {"run_id": run_id, "run_dir": str(run_dir), **report}


def _promotion_gate(
    environment_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> dict:
    delta_bge = environment_metrics["delta_log_loss_vs_bge"].to_numpy(
        dtype=np.float64
    )
    delta_v02 = environment_metrics["delta_log_loss_vs_v02"].to_numpy(
        dtype=np.float64
    )
    session_bge = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_bge_replace",
        group_column="session_id",
        n_replicates=n_bootstrap,
        seed=SEED + 101,
    )
    session_v02 = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v02",
        group_column="session_id",
        n_replicates=n_bootstrap,
        seed=SEED + 102,
    )
    family_v02 = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v02",
        group_column="semantic_family",
        n_replicates=n_bootstrap,
        seed=SEED + 103,
    )
    backup_clauses = {
        "mean_gain_at_least_0_0010": float(-delta_bge.mean()) >= 0.0010,
        "improves_at_least_three_environments": int(np.sum(delta_bge < 0.0)) >= 3,
        "worst_environment_regression_at_most_0_0005": float(delta_bge.max())
        <= 0.0005,
        "macro_auc_non_regression": float(
            environment_metrics["delta_roc_auc_vs_bge"].mean()
        )
        >= 0.0,
        "macro_brier_non_regression": float(
            environment_metrics["delta_brier_score_vs_bge"].mean()
        )
        <= 0.0,
        "macro_ece_non_regression": float(
            environment_metrics["delta_ece_10_vs_bge"].mean()
        )
        <= 0.0,
        "session_bootstrap_support_at_least_0_90": session_bge[
            "support_positive_gain"
        ]
        >= 0.90,
    }
    top5_clauses = {
        "mean_gain_vs_v02_at_least_0_0043": float(-delta_v02.mean()) >= 0.0043,
        "improves_every_environment_vs_v02": bool(np.all(delta_v02 < 0.0)),
        "worst_fold_regression_vs_v02_at_most_0_0010": float(
            fold_metrics["delta_log_loss_vs_v02"].max()
        )
        <= 0.0010,
        "legal_subgroup_regression_audit_complete": False,
        "macro_auc_non_regression_vs_v02": float(
            environment_metrics["delta_roc_auc_vs_v02"].mean()
        )
        >= 0.0,
        "macro_brier_non_regression_vs_v02": float(
            environment_metrics["delta_brier_score_vs_v02"].mean()
        )
        <= 0.0,
        "macro_ece_non_regression_vs_v02": float(
            environment_metrics["delta_ece_10_vs_v02"].mean()
        )
        <= 0.0,
        "session_bootstrap_support_at_least_0_95": session_v02[
            "support_positive_gain"
        ]
        >= 0.95,
        "semantic_family_bootstrap_support_at_least_0_95": family_v02[
            "support_positive_gain"
        ]
        >= 0.95,
        "locked_joint_confirmation_complete": True,
        "V_final_gain_at_least_0_0035": False,
    }
    return {
        "mean_gain_vs_bge": float(-delta_bge.mean()),
        "mean_gain_vs_v02": float(-delta_v02.mean()),
        "projected_public_log_loss": float(0.6081 + delta_v02.mean()),
        "conservative_public_log_loss": float(0.6081 + 0.60 * delta_v02.mean()),
        "backup_clauses": backup_clauses,
        "passes_backup_gate": bool(all(backup_clauses.values())),
        "top5_clauses": top5_clauses,
        "passes_top5_gate": bool(all(top5_clauses.values())),
        "session_bootstrap_vs_bge": session_bge,
        "session_bootstrap_vs_v02": session_v02,
        "semantic_family_bootstrap_vs_v02": family_v02,
        "V_final_accessed": False,
    }


def run_joint(
    project_root: str | Path,
    *,
    run_id: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict:
    root = Path(project_root).resolve()
    run_dir = root / "experiments" / "runs" / run_id
    lock_path = run_dir / "locked_candidate.json"
    report_path = run_dir / "report.json"
    if not lock_path.exists() or not report_path.exists():
        raise FileNotFoundError("E510 joint stage requires a passed development lock.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        lock.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != "passed_development_gate"
        or lock.get("development_predictions_sha256")
        != sha256_file(run_dir / "development_predictions.parquet")
    ):
        raise ValueError("E510 development lock is invalid or stale.")
    timing, timing_metadata = build_or_load_timing_cache(root)
    component = load_component_oof(root, JOINT_ENVIRONMENT)
    timing_values = align_timing_features(component, timing)
    timing_prediction = fit_timing_oof(component, timing_values)
    weight = float(lock["selected_weight"])
    joint_predictions = build_prediction_rows(
        component, timing_prediction, weights=(weight,)
    )
    development = pd.read_parquet(run_dir / "development_predictions.parquet")
    development = development.loc[development["timing_weight"].eq(weight)]
    predictions = pd.concat([development, joint_predictions], ignore_index=True)
    fold_metrics, environment_metrics = _metric_tables(predictions)
    gate = _promotion_gate(
        environment_metrics,
        fold_metrics,
        predictions,
        n_bootstrap=n_bootstrap,
    )
    _atomic_parquet(run_dir / "all_predictions.parquet", predictions)
    _atomic_csv(run_dir / "all_fold_metrics.csv", fold_metrics)
    _atomic_csv(run_dir / "all_environment_metrics.csv", environment_metrics)
    _atomic_json(run_dir / "promotion_gate.json", gate)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "report.json"
    }
    final_report = {
        **report,
        "stage": "joint",
        "status": (
            "passed_backup_gate"
            if gate["passes_backup_gate"]
            else "rejected_promotion_gate"
        ),
        "selected_weight": weight,
        "joint_source_component_sha256": EXPECTED_COMPONENT_SHA256[
            JOINT_ENVIRONMENT
        ],
        "timing_cache_sha256": timing_metadata["cache_sha256"],
        "promotion_gate": gate,
        "artifact_hashes": artifact_hashes,
        "V_final_accessed": False,
    }
    _atomic_json(report_path, final_report)
    return {"run_id": run_id, "run_dir": str(run_dir), **final_report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen E510 timing-dynamics validation without V_final access."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage", choices=("development", "joint"), default="development")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "joint":
        if not args.run_id:
            raise ValueError("E510 joint stage requires --run-id.")
        result = run_joint(
            args.project_root,
            run_id=args.run_id,
            n_bootstrap=args.bootstrap_replicates,
        )
    else:
        result = run_development(
            args.project_root,
            run_id=args.run_id,
            n_bootstrap=args.bootstrap_replicates,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
