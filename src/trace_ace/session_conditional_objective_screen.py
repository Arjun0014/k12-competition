from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from trace_ace.bge_base_multiview_screen import (
    _current_rss_bytes,
    assert_runtime,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E750_session_conditional_objective_v1"
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
SEED = 20260728
REGULARIZATION_C = 0.1
MAX_ITERATIONS = 400
TOLERANCE = 1e-5
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6
MAX_PROJECTED_SECONDS = 3_600.0
MAX_RSS_BYTES = 8 * 1024**3
EXPECTED_SOURCE_SHA256 = {
    "modeling_base.parquet": (
        "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
    ),
    "bge_base_context_256.npy": (
        "b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a"
    ),
    "bge_base_objective_256.npy": (
        "6cc754a287f9ec303aa5f81dd98ab0321c7c5299b2a4e251d6f86f63afe6eb27"
    ),
    "component_oof_V_seen.parquet": (
        "1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af"
    ),
    "component_oof_V_objective.parquet": (
        "c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7"
    ),
    "component_oof_V_style.parquet": (
        "6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038"
    ),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_dir = paths.experiments_dir / "runs" / SOURCE_RUN_ID
    return {
        "modeling_base.parquet": paths.cache_dir / "modeling_base.parquet",
        "bge_base_context_256.npy": paths.cache_dir / "bge_base_context_256.npy",
        "bge_base_objective_256.npy": (
            paths.cache_dir / "bge_base_objective_256.npy"
        ),
        **{
            f"component_oof_{environment}.parquet": (
                source_dir / f"component_oof_{environment}.parquet"
            )
            for environment in SELECTION_ENVIRONMENTS
        },
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    source_paths = _source_paths(project_root)
    actual = {name: _sha256(path) for name, path in source_paths.items()}
    if actual != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "E750 immutable source verification failed: "
            f"expected={EXPECTED_SOURCE_SHA256}, actual={actual}"
        )
    return actual


def align_features(
    component: pd.DataFrame,
    modeling: pd.DataFrame,
    features: np.ndarray,
) -> np.ndarray:
    if (
        features.ndim != 2
        or len(modeling) != len(features)
        or modeling["response_id"].duplicated().any()
        or component["response_id"].duplicated().any()
    ):
        raise ValueError("E750 feature alignment inputs violate the frozen contract.")
    lookup = pd.Series(
        np.arange(len(modeling), dtype=np.int64),
        index=modeling["response_id"].astype(str),
    )
    positions = component["response_id"].astype(str).map(lookup)
    if positions.isna().any() or positions.duplicated().any():
        raise ValueError("E750 could not align every component response exactly once.")
    aligned = features[positions.to_numpy(dtype=np.int64)]
    if aligned.shape != (len(component), features.shape[1]):
        raise RuntimeError("E750 aligned feature shape is invalid.")
    return aligned


def build_session_pairs(
    frame: pd.DataFrame,
    training_mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    if (
        training_mask.shape != (len(frame),)
        or labels.shape != (len(frame),)
        or not {"response_id", "session_id"}.issubset(frame.columns)
        or set(np.unique(labels[training_mask]).tolist()) != {0, 1}
    ):
        raise ValueError("E750 pair inputs violate the frozen contract.")
    training_rows = np.flatnonzero(training_mask)
    ordered = (
        frame.iloc[training_rows]
        .assign(_row=training_rows)
        .sort_values(["session_id", "response_id"], kind="mergesort")
    )
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    mixed_sessions = 0
    for _, group in ordered.groupby("session_id", sort=True):
        rows = group["_row"].to_numpy(dtype=np.int64)
        positive = rows[labels[rows] == 1]
        negative = rows[labels[rows] == 0]
        if not len(positive) or not len(negative):
            continue
        mixed_sessions += 1
        session_positive = np.repeat(positive, len(negative))
        session_negative = np.tile(negative, len(positive))
        session_pairs = len(session_positive)
        positive_rows.append(session_positive)
        negative_rows.append(session_negative)
        pair_weights.append(
            np.full(session_pairs, 0.5 / session_pairs, dtype=np.float64)
        )
    if not positive_rows:
        raise ValueError("E750 found no mixed-outcome legal training session.")
    positive = np.concatenate(positive_rows)
    negative = np.concatenate(negative_rows)
    weights = np.concatenate(pair_weights)
    positive_sessions = frame.iloc[positive]["session_id"].astype(str).to_numpy()
    negative_sessions = frame.iloc[negative]["session_id"].astype(str).to_numpy()
    if (
        positive.shape != negative.shape
        or weights.shape != positive.shape
        or np.any(labels[positive] != 1)
        or np.any(labels[negative] != 0)
        or not np.array_equal(positive_sessions, negative_sessions)
        or not np.isclose(2.0 * weights.sum(), mixed_sessions)
    ):
        raise RuntimeError("E750 session-pair audit failed.")
    audit: dict[str, int | float] = {
        "training_rows": int(training_mask.sum()),
        "training_sessions": int(
            frame.loc[training_mask, "session_id"].astype(str).nunique()
        ),
        "mixed_outcome_sessions": mixed_sessions,
        "positive_negative_pairs": len(positive),
        "symmetric_examples": 2 * len(positive),
        "symmetric_total_sample_weight": float(2.0 * weights.sum()),
    }
    return positive, negative, weights, audit


def fit_conditional_head(
    features: np.ndarray,
    pair_positive: np.ndarray,
    pair_negative: np.ndarray,
    pair_weights: np.ndarray,
    validation_features: np.ndarray,
    *,
    training_prior: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if (
        features.ndim != 2
        or validation_features.ndim != 2
        or features.shape[1] != validation_features.shape[1]
        or pair_positive.shape != pair_negative.shape
        or pair_positive.shape != pair_weights.shape
        or pair_positive.ndim != 1
        or not len(pair_positive)
        or pair_positive.min() < 0
        or pair_negative.min() < 0
        or pair_positive.max() >= len(features)
        or pair_negative.max() >= len(features)
        or not np.isfinite(pair_weights).all()
        or np.any(pair_weights <= 0)
        or not 0.0 < training_prior < 1.0
    ):
        raise ValueError("E750 fit arrays violate the frozen contract.")
    difference = np.asarray(
        features[pair_positive] - features[pair_negative], dtype=np.float32
    )
    examples = np.vstack([difference, -difference])
    labels = np.concatenate(
        [
            np.ones(len(difference), dtype=np.int8),
            np.zeros(len(difference), dtype=np.int8),
        ]
    )
    sample_weight = np.concatenate([pair_weights, pair_weights])
    model = LogisticRegression(
        C=REGULARIZATION_C,
        solver="lbfgs",
        fit_intercept=False,
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE,
        random_state=SEED,
    )
    model.fit(examples, labels, sample_weight=sample_weight)
    if model.classes_.tolist() != [0, 1] or model.coef_.shape != (
        1,
        features.shape[1],
    ):
        raise RuntimeError("E750 conditional-head class or coefficient audit failed.")
    conditional_score = np.asarray(
        validation_features @ model.coef_[0], dtype=np.float64
    )
    prior_logit = np.log(training_prior / (1.0 - training_prior))
    prediction = np.clip(
        expit(prior_logit + conditional_score),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )
    if (
        prediction.shape != (len(validation_features),)
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError("E750 conditional-head predictions are invalid.")
    summary: dict[str, int | float] = {
        "feature_width": int(features.shape[1]),
        "pairs": int(len(pair_positive)),
        "symmetric_examples": int(len(examples)),
        "total_sample_weight": float(sample_weight.sum()),
        "training_prior": float(training_prior),
        "iterations": int(model.n_iter_[0]),
        "coefficient_norm": float(np.linalg.norm(model.coef_)),
        "conditional_score_mean": float(conditional_score.mean()),
        "conditional_score_std": float(conditional_score.std()),
        "prediction_mean": float(prediction.mean()),
    }
    return prediction, summary


def conditional_oof(
    component: pd.DataFrame,
    features: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target = component["target"].to_numpy(dtype=np.int8)
    folds = component["fold"].to_numpy(dtype=np.int8)
    prediction = np.full(len(component), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        training_mask, validation_mask = _fold_masks(component, folds, fold)
        training_sessions = set(
            component.loc[training_mask, "session_id"].astype(str)
        )
        validation_sessions = set(
            component.loc[validation_mask, "session_id"].astype(str)
        )
        if training_sessions.intersection(validation_sessions):
            raise RuntimeError(f"E750 session leakage in fold {fold}.")
        positive, negative, weights, pair_audit = build_session_pairs(
            component, training_mask, target
        )
        fold_prediction, fit_summary = fit_conditional_head(
            features,
            positive,
            negative,
            weights,
            features[validation_mask],
            training_prior=float(target[training_mask].mean()),
        )
        prediction[validation_mask] = fold_prediction
        summaries.append(
            {
                "fold": fold,
                **pair_audit,
                **fit_summary,
                "validation_rows": int(validation_mask.sum()),
            }
        )
        if _current_rss_bytes() >= MAX_RSS_BYTES:
            raise MemoryError("E750 exceeded the frozen 8 GiB RSS gate.")
    if not np.isfinite(prediction).all():
        raise RuntimeError("E750 OOF predictions are incomplete.")
    return prediction, summaries


def _baseline_prediction(component: pd.DataFrame) -> np.ndarray:
    return (
        0.25 * component["pred_full"].to_numpy(dtype=np.float64)
        + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
        + 0.50 * component["pred_bge_base"].to_numpy(dtype=np.float64)
    )


def build_prediction_rows(
    component: pd.DataFrame,
    conditional_prediction: np.ndarray,
    weights: Sequence[float],
) -> pd.DataFrame:
    if conditional_prediction.shape != (len(component),):
        raise ValueError("E750 conditional prediction is misaligned.")
    baseline = _baseline_prediction(component)
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
        candidate["conditional_weight"] = float(weight)
        candidate["pred_v05_raw"] = baseline
        candidate["pred_session_conditional"] = conditional_prediction
        candidate["prediction"] = np.clip(
            (1.0 - weight) * baseline + weight * conditional_prediction,
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        rows.append(candidate)
    return pd.concat(rows, ignore_index=True)


def _metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (environment, weight, fold), group in predictions.groupby(
        ["environment", "conditional_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        candidate = binary_metrics(target, scored["prediction"])
        row: dict[str, object] = {
            "environment": str(environment),
            "conditional_weight": float(weight),
            "fold": int(fold),
            "rows": int(len(scored)),
        }
        for name, value in candidate.items():
            row[name] = value
            row[f"baseline_{name}"] = baseline[name]
            row[f"delta_{name}_vs_v05_raw"] = value - baseline[name]
        rows.append(row)
    folds = pd.DataFrame(rows)
    metric_names = (
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "prediction_mean",
    )
    aggregate_columns = [
        *metric_names,
        *[f"baseline_{name}" for name in metric_names],
        *[f"delta_{name}_vs_v05_raw" for name in metric_names],
    ]
    environments = (
        folds.groupby(
            ["environment", "conditional_weight"], as_index=False, sort=True
        )[aggregate_columns]
        .mean()
        .sort_values(
            ["environment", "conditional_weight"], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def select_candidate(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    environment_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    bootstraps: dict[str, dict[str, object]] = {}
    for index, weight in enumerate(BLEND_WEIGHTS):
        metrics = environment_metrics.loc[
            environment_metrics["conditional_weight"].eq(weight)
        ]
        selected_predictions = predictions.loc[
            predictions["conditional_weight"].eq(weight)
        ]
        session_bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="session_id",
            n_replicates=n_bootstrap,
            seed=SEED + index,
        )
        family_bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="semantic_family",
            n_replicates=n_bootstrap,
            seed=SEED + 100 + index,
        )
        bootstraps[f"{weight:.2f}"] = {
            "session": session_bootstrap,
            "semantic_family": family_bootstrap,
        }
        delta_loss = metrics[
            "delta_log_loss_vs_v05_raw"
        ].to_numpy(dtype=np.float64)
        weight_folds = fold_metrics.loc[
            fold_metrics["conditional_weight"].eq(weight)
        ]
        rows.append(
            {
                "conditional_weight": weight,
                "mean_log_loss": float(metrics["log_loss"].mean()),
                "mean_log_loss_gain_vs_v05_raw": float(-delta_loss.mean()),
                "improved_environments": int(np.sum(delta_loss < 0.0)),
                "worst_environment_delta_log_loss_vs_v05_raw": float(
                    delta_loss.max()
                ),
                "worst_fold_delta_log_loss_vs_v05_raw": float(
                    weight_folds["delta_log_loss_vs_v05_raw"].max()
                ),
                "mean_delta_roc_auc_vs_v05_raw": float(
                    metrics["delta_roc_auc_vs_v05_raw"].mean()
                ),
                "mean_delta_brier_score_vs_v05_raw": float(
                    metrics["delta_brier_score_vs_v05_raw"].mean()
                ),
                "mean_delta_ece_10_vs_v05_raw": float(
                    metrics["delta_ece_10_vs_v05_raw"].mean()
                ),
                "session_bootstrap_support": session_bootstrap[
                    "support_positive_gain"
                ],
                "semantic_family_bootstrap_support": family_bootstrap[
                    "support_positive_gain"
                ],
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "conditional_weight"], kind="mergesort"
    )
    selected = selection.iloc[0].to_dict()
    clauses = {
        "mean_log_loss_gain_at_least_0_0016": (
            selected["mean_log_loss_gain_vs_v05_raw"] >= 0.0016
        ),
        "all_three_environments_improve": (
            selected["improved_environments"] == len(SELECTION_ENVIRONMENTS)
        ),
        "no_environment_log_loss_regression": (
            selected["worst_environment_delta_log_loss_vs_v05_raw"] <= 0.0
        ),
        "worst_fold_regression_at_most_0_0005": (
            selected["worst_fold_delta_log_loss_vs_v05_raw"] <= 0.0005
        ),
        "macro_auroc_non_regression": (
            selected["mean_delta_roc_auc_vs_v05_raw"] >= 0.0
        ),
        "macro_brier_non_regression": (
            selected["mean_delta_brier_score_vs_v05_raw"] <= 0.0
        ),
        "macro_ece_non_regression": (
            selected["mean_delta_ece_10_vs_v05_raw"] <= 0.0
        ),
        "session_bootstrap_support_at_least_0_95": (
            selected["session_bootstrap_support"] >= 0.95
        ),
        "semantic_family_bootstrap_support_at_least_0_95": (
            selected["semantic_family_bootstrap_support"] >= 0.95
        ),
    }
    selected_key = f"{selected['conditional_weight']:.2f}"
    decision: dict[str, object] = {
        "selected_weight": float(selected["conditional_weight"]),
        "selected_row": selected,
        "clauses": clauses,
        "passes_screen": bool(all(clauses.values())),
        "selected_bootstrap": bootstraps[selected_key],
        "all_preregistered_bootstraps": bootstraps,
        "V_joint_accessed": False,
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


def _benchmark_path(project_root: str | Path) -> Path:
    return (
        discover_project_paths(project_root).cache_dir
        / "session_conditional_e750_benchmark.json"
    )


def _synthetic_labels(response_ids: pd.Series) -> np.ndarray:
    return np.fromiter(
        (
            int(
                hashlib.sha256(str(response_id).encode("utf-8")).hexdigest()[:8],
                16,
            )
            % 2
            for response_id in response_ids
        ),
        dtype=np.int8,
        count=len(response_ids),
    )


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    paths = discover_project_paths(project_root)
    modeling = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features, _ = prepare_bge_base_features(paths.cache_dir, modeling)
    component = load_component_oof(project_root, "V_seen")
    features = align_features(component, modeling, base_features)
    synthetic = _synthetic_labels(component["response_id"])
    folds = component["fold"].to_numpy(dtype=np.int8)
    training_mask, validation_mask = _fold_masks(component, folds, 0)
    positive, negative, weights, pair_audit = build_session_pairs(
        component, training_mask, synthetic
    )
    started = time.perf_counter()
    _, fit_summary = fit_conditional_head(
        features,
        positive,
        negative,
        weights,
        features[validation_mask],
        training_prior=float(synthetic[training_mask].mean()),
    )
    elapsed = time.perf_counter() - started
    projected_seconds = elapsed * 15.0
    peak_rss = _current_rss_bytes()
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": (
            "one V_seen fold with deterministic response-id-hash synthetic labels"
        ),
        "elapsed_seconds": elapsed,
        "projected_fifteen_fold_seconds": projected_seconds,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_SECONDS
            and peak_rss < MAX_RSS_BYTES
        ),
        "pair_audit": pair_audit,
        "fit_summary": fit_summary,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    path = _benchmark_path(project_root)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **result,
        "benchmark_path": str(path),
        "benchmark_sha256": _sha256(path),
    }


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    path = _benchmark_path(project_root)
    if not path.exists():
        raise FileNotFoundError("Run the frozen E750 benchmark before validation.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol_id") != PROTOCOL_ID
        or result.get("source_hashes") != verify_sources(project_root)
        or result.get("proceed") is not True
        or result.get("competition_outcomes_accessed") is not False
        or result.get("V_joint_accessed") is not False
        or result.get("V_final_accessed") is not False
    ):
        raise ValueError("E750 benchmark binding or resource gate is invalid.")
    return {
        **result,
        "benchmark_path": str(path),
        "benchmark_sha256": _sha256(path),
    }


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    benchmark_result = _load_benchmark(project_root)
    paths = discover_project_paths(project_root)
    modeling = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features, _ = prepare_bge_base_features(paths.cache_dir, modeling)
    prediction_frames: list[pd.DataFrame] = []
    fit_summaries: dict[str, list[dict[str, object]]] = {}
    peak_rss = _current_rss_bytes()
    for environment in SELECTION_ENVIRONMENTS:
        print(f"E750 validation: {environment}", flush=True)
        component = load_component_oof(project_root, environment)
        features = align_features(component, modeling, base_features)
        conditional_prediction, summaries = conditional_oof(component, features)
        fit_summaries[environment] = summaries
        prediction_frames.append(
            build_prediction_rows(
                component, conditional_prediction, BLEND_WEIGHTS
            )
        )
        peak_rss = max(peak_rss, _current_rss_bytes())
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics, environment_metrics = _metric_tables(predictions)
    selection, decision = select_candidate(
        predictions,
        fold_metrics,
        environment_metrics,
        n_bootstrap=BOOTSTRAP_REPLICATES,
    )
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime(
        "%Y%m%dT%H%M%SZ_session_conditional_objective"
    )
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "development_predictions.parquet"
    fold_path = run_dir / "development_fold_metrics.csv"
    environment_path = run_dir / "development_environment_metrics.csv"
    selection_path = run_dir / "development_selection.csv"
    bootstrap_path = run_dir / "development_bootstraps.json"
    predictions.to_parquet(predictions_path, index=False)
    fold_metrics.to_csv(fold_path, index=False, lineterminator="\n")
    environment_metrics.to_csv(environment_path, index=False, lineterminator="\n")
    selection.to_csv(selection_path, index=False, lineterminator="\n")
    bootstrap_path.write_text(
        json.dumps(
            decision["all_preregistered_bootstraps"], indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_hashes = {
        path.name: _sha256(path)
        for path in (
            predictions_path,
            fold_path,
            environment_path,
            selection_path,
            bootstrap_path,
        )
    }
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E750_session_conditional_objective",
        "source_run_id": SOURCE_RUN_ID,
        "source_hashes": source_hashes,
        "lineage": {
            "estimator": "equal-session conditional positive-negative likelihood",
            "pairing_unit": "same legal training session",
            "feature_block": (
                "immutable 3072-wide BGE-base context/objective/interaction/"
                "absolute-difference block"
            ),
            "inference": (
                "fold-prior conditional probability; sample-independent at test"
            ),
            "raw_v05_weights": {
                "pred_full": 0.25,
                "pred_role": 0.25,
                "pred_bge_base": 0.50,
            },
            "conditional_blend_weights": list(BLEND_WEIGHTS),
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "fit_intercept": False,
            "max_iterations": MAX_ITERATIONS,
            "tolerance": TOLERANCE,
            "seed": SEED,
        },
        "preregistered_gate": decision,
        "fit_summaries": fit_summaries,
        "benchmark": benchmark_result,
        "runtime": runtime,
        "runtime_metadata": _runtime_metadata(),
        "validation_runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "artifact_sha256": artifact_hashes,
        "competition_outcomes_accessed": True,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passes_screen": decision["passes_screen"],
        "selected": decision["selected_row"],
        "clauses": decision["clauses"],
        "artifact_sha256": artifact_hashes,
        "report_sha256": _sha256(report_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E750 session-conditional objective screen."
    )
    parser.add_argument("stage", choices=("benchmark", "validate"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = (
        benchmark(args.project_root)
        if args.stage == "benchmark"
        else validate(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
