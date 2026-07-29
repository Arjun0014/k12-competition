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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trace_ace.hard_validation import _fold_masks
from trace_ace.metrics import binary_metrics
from trace_ace.productive_numeric_elaboration_discovery import (
    FEATURE_NAMES,
    MIN_STUDENT_WORDS,
    SEED,
    assert_runtime,
)
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E810_productive_numeric_elaboration_validation_v1"
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
EXPECTED_FEATURE_CACHE_SHA256 = (
    "560e82ceb07dc22ef0511da2b92d361f6e52564be997a2364a282e8c6c1eca27"
)
EXPECTED_DISCOVERY_REPORT_SHA256 = (
    "ed202b92697ca7d11c1115378f1b8825761901d9ee60092526ff56c5a445afac"
)
REGULARIZATION_C = 1.0
MAX_ITERATIONS = 1_000
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6
V05_WEIGHTS = {
    "pred_full": 0.25,
    "pred_role": 0.25,
    "pred_bge_base": 0.50,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "features": (
            root
            / "data_cache"
            / "productive_numeric_elaboration_e810_target_free.parquet"
        ),
        "discovery_report": (
            root
            / "data_cache"
            / "productive_numeric_elaboration_e810_target_free_report.json"
        ),
        "experiments": root / "experiments",
    }


def verify_target_free_discovery(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    if _sha256(paths["features"]) != EXPECTED_FEATURE_CACHE_SHA256:
        raise ValueError("E810 target-free feature cache SHA-256 changed.")
    if _sha256(paths["discovery_report"]) != EXPECTED_DISCOVERY_REPORT_SHA256:
        raise ValueError("E810 target-free discovery report SHA-256 changed.")
    report = json.loads(paths["discovery_report"].read_text(encoding="utf-8"))
    required = {
        "passes_target_free_gate": True,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        "feature_cache_sha256": EXPECTED_FEATURE_CACHE_SHA256,
    }
    observed = {name: report.get(name) for name in required}
    if observed != required:
        raise ValueError(f"E810 target-free discovery contract changed: {observed}")
    return report


def load_numeric_features(project_root: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(_paths(project_root)["features"])
    if list(frame.columns) != ["session_id", *FEATURE_NAMES]:
        raise ValueError("E810 feature-cache columns changed.")
    if frame["session_id"].duplicated().any() or len(frame) != 22_821:
        raise ValueError("E810 feature cache must contain 22,821 unique sessions.")
    values = frame[list(FEATURE_NAMES)]
    complete = values.notna().all(axis=1)
    if (
        int(complete.sum()) != 22_816
        or not np.isfinite(values.loc[complete].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("E810 feature-cache missingness contract changed.")
    return frame


def align_numeric_features(
    component: pd.DataFrame,
    features: pd.DataFrame,
) -> np.ndarray:
    aligned = component[["session_id"]].merge(
        features,
        on="session_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(aligned) != len(component):
        raise ValueError("E810 feature alignment changed row count.")
    values = aligned[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    missing = np.isnan(values)
    if not np.array_equal(missing[:, 0], missing[:, 1]) or not np.array_equal(
        missing[:, 0], missing[:, 2]
    ):
        raise ValueError("E810 aligned feature missingness differs by column.")
    if not np.isfinite(values[~missing]).all():
        raise ValueError("E810 aligned features contain non-finite values.")
    return values


def fit_numeric_oof(
    component: pd.DataFrame,
    feature_values: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if feature_values.shape != (len(component), len(FEATURE_NAMES)):
        raise ValueError("E810 aligned feature shape changed.")
    target = component["target"].to_numpy(dtype=np.int8)
    folds = component["fold"].to_numpy(dtype=np.int8)
    prediction = np.full(len(component), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    feature_eligible = np.isfinite(feature_values).all(axis=1) & (
        feature_values[:, 0] >= MIN_STUDENT_WORDS
    )
    for fold in range(5):
        training_mask, validation_mask = _fold_masks(component, folds, fold)
        model_training = training_mask & feature_eligible
        model_validation = validation_mask & feature_eligible
        quiet_validation = validation_mask & ~feature_eligible
        training_sessions = set(component.loc[training_mask, "session_id"].astype(str))
        validation_sessions = set(
            component.loc[validation_mask, "session_id"].astype(str)
        )
        if training_sessions & validation_sessions:
            raise RuntimeError(f"E810 session leakage in fold {fold}.")
        if model_training.sum() < 100 or len(np.unique(target[model_training])) != 2:
            raise ValueError(f"E810 fold {fold} has invalid model training rows.")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=REGULARIZATION_C,
                solver="lbfgs",
                max_iter=MAX_ITERATIONS,
                random_state=SEED,
            ),
        )
        model.fit(feature_values[model_training], target[model_training])
        prediction[model_validation] = model.predict_proba(
            feature_values[model_validation]
        )[:, 1]
        fold_prior = float(target[training_mask].mean())
        prediction[quiet_validation] = fold_prior
        logistic = model.named_steps["logisticregression"]
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(training_mask.sum()),
                "model_training_rows": int(model_training.sum()),
                "validation_rows": int(validation_mask.sum()),
                "modeled_validation_rows": int(model_validation.sum()),
                "quiet_validation_rows": int(quiet_validation.sum()),
                "training_sessions": len(training_sessions),
                "validation_sessions": len(validation_sessions),
                "quiet_fallback": "fold training prior",
                "fold_training_prior": fold_prior,
                "standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(
                        FEATURE_NAMES, logistic.coef_[0], strict=True
                    )
                },
                "intercept": float(logistic.intercept_[0]),
                "iterations": int(logistic.n_iter_[0]),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("E810 OOF predictions are incomplete.")
    return prediction, summaries


def _baseline_prediction(component: pd.DataFrame) -> np.ndarray:
    return sum(
        weight * component[name].to_numpy(dtype=np.float64)
        for name, weight in V05_WEIGHTS.items()
    )


def build_prediction_rows(
    component: pd.DataFrame,
    numeric_prediction: np.ndarray,
    weights: Sequence[float],
) -> pd.DataFrame:
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
        candidate["numeric_weight"] = float(weight)
        candidate["pred_v05_raw"] = baseline
        candidate["pred_numeric_elaboration"] = numeric_prediction
        candidate["prediction"] = np.clip(
            (1.0 - weight) * baseline + weight * numeric_prediction,
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
        ["environment", "numeric_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        candidate = binary_metrics(target, scored["prediction"])
        row: dict[str, object] = {
            "environment": str(environment),
            "numeric_weight": float(weight),
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
    columns = [
        *metric_names,
        *[f"baseline_{name}" for name in metric_names],
        *[f"delta_{name}_vs_v05_raw" for name in metric_names],
    ]
    environments = (
        folds.groupby(["environment", "numeric_weight"], as_index=False, sort=True)[
            columns
        ]
        .mean()
        .sort_values(["environment", "numeric_weight"], kind="mergesort")
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def select_candidate(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    environment_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    bootstraps: dict[str, dict[str, object]] = {}
    for index, weight in enumerate(BLEND_WEIGHTS):
        metrics = environment_metrics.loc[
            environment_metrics["numeric_weight"].eq(weight)
        ]
        selected_predictions = predictions.loc[predictions["numeric_weight"].eq(weight)]
        session_bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="session_id",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + index,
        )
        family_bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="semantic_family",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + 100 + index,
        )
        bootstraps[f"{weight:.2f}"] = {
            "session": session_bootstrap,
            "semantic_family": family_bootstrap,
        }
        delta_loss = metrics["delta_log_loss_vs_v05_raw"].to_numpy(dtype=np.float64)
        weight_folds = fold_metrics.loc[fold_metrics["numeric_weight"].eq(weight)]
        rows.append(
            {
                "numeric_weight": weight,
                "mean_log_loss": float(metrics["log_loss"].mean()),
                "mean_log_loss_gain_vs_v05_raw": float(-delta_loss.mean()),
                "improved_environments": int(np.sum(delta_loss < 0.0)),
                "worst_environment_delta_log_loss_vs_v05_raw": float(delta_loss.max()),
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
                "session_bootstrap_support": session_bootstrap["support_positive_gain"],
                "semantic_family_bootstrap_support": family_bootstrap[
                    "support_positive_gain"
                ],
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "numeric_weight"], kind="mergesort"
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
        "macro_ece_non_regression": (selected["mean_delta_ece_10_vs_v05_raw"] <= 0.0),
        "session_bootstrap_support_at_least_0_95": (
            selected["session_bootstrap_support"] >= 0.95
        ),
        "semantic_family_bootstrap_support_at_least_0_95": (
            selected["semantic_family_bootstrap_support"] >= 0.95
        ),
    }
    selected_key = f"{selected['numeric_weight']:.2f}"
    return selection.reset_index(drop=True), {
        "selected_weight": float(selected["numeric_weight"]),
        "selected_row": selected,
        "clauses": clauses,
        "passes_selection_gate": bool(all(clauses.values())),
        "selected_bootstrap": bootstraps[selected_key],
        "all_preregistered_bootstraps": bootstraps,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def _confirm_joint(
    project_root: str | Path,
    features: pd.DataFrame,
    selected_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    component = load_component_oof(project_root, CONFIRMATION_ENVIRONMENT)
    values = align_numeric_features(component, features)
    numeric, fits = fit_numeric_oof(component, values)
    predictions = build_prediction_rows(component, numeric, [selected_weight])
    folds, environments = _metric_tables(predictions)
    row = environments.iloc[0]
    session_bootstrap = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v05_raw",
        group_column="session_id",
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1_000,
    )
    family_bootstrap = _macro_cluster_bootstrap(
        predictions,
        candidate_column="prediction",
        baseline_column="pred_v05_raw",
        group_column="semantic_family",
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1_100,
    )
    gain = -float(row["delta_log_loss_vs_v05_raw"])
    clauses = {
        "joint_log_loss_gain_at_least_0_0010": gain >= 0.0010,
        "joint_worst_fold_regression_at_most_0_0005": (
            float(folds["delta_log_loss_vs_v05_raw"].max()) <= 0.0005
        ),
        "joint_auroc_non_regression": (float(row["delta_roc_auc_vs_v05_raw"]) >= 0.0),
        "joint_brier_non_regression": (
            float(row["delta_brier_score_vs_v05_raw"]) <= 0.0
        ),
        "joint_ece_non_regression": (float(row["delta_ece_10_vs_v05_raw"]) <= 0.0),
        "joint_session_bootstrap_support_at_least_0_95": (
            session_bootstrap["support_positive_gain"] >= 0.95
        ),
        "joint_semantic_family_bootstrap_support_at_least_0_95": (
            family_bootstrap["support_positive_gain"] >= 0.95
        ),
    }
    return (
        predictions,
        folds,
        {
            "environment_metrics": row.to_dict(),
            "fit_summaries": fits,
            "session_bootstrap": session_bootstrap,
            "semantic_family_bootstrap": family_bootstrap,
            "clauses": clauses,
            "passes_confirmation_gate": bool(all(clauses.values())),
            "V_joint_accessed": True,
            "V_final_accessed": False,
        },
    )


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime(project_root)
    discovery = verify_target_free_discovery(project_root)
    features = load_numeric_features(project_root)
    prediction_frames: list[pd.DataFrame] = []
    fit_summaries: dict[str, list[dict[str, object]]] = {}
    for environment in SELECTION_ENVIRONMENTS:
        component = load_component_oof(project_root, environment)
        values = align_numeric_features(component, features)
        numeric, summaries = fit_numeric_oof(component, values)
        fit_summaries[environment] = summaries
        prediction_frames.append(
            build_prediction_rows(component, numeric, BLEND_WEIGHTS)
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics, environment_metrics = _metric_tables(predictions)
    selection, decision = select_candidate(
        predictions, fold_metrics, environment_metrics
    )

    joint_predictions: pd.DataFrame | None = None
    joint_folds: pd.DataFrame | None = None
    joint: dict[str, object] = {
        "reason": "Selection gate failed; V_joint remained sealed.",
        "passes_confirmation_gate": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    if decision["passes_selection_gate"]:
        joint_predictions, joint_folds, joint = _confirm_joint(
            project_root,
            features,
            float(decision["selected_weight"]),
        )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_productive_numeric_elaboration")
    run_dir = _paths(project_root)["experiments"] / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, pd.DataFrame | dict[str, object]] = {
        "development_predictions.parquet": predictions,
        "development_fold_metrics.csv": fold_metrics,
        "development_environment_metrics.csv": environment_metrics,
        "development_selection.csv": selection,
        "development_bootstraps.json": decision["all_preregistered_bootstraps"],
    }
    if joint_predictions is not None and joint_folds is not None:
        artifacts.update(
            {
                "joint_confirmation_predictions.parquet": joint_predictions,
                "joint_confirmation_fold_metrics.csv": joint_folds,
                "joint_confirmation.json": joint,
            }
        )
    artifact_hashes: dict[str, str] = {}
    for name, value in artifacts.items():
        path = run_dir / name
        if name.endswith(".parquet"):
            assert isinstance(value, pd.DataFrame)
            value.to_parquet(path, index=False)
        elif name.endswith(".csv"):
            assert isinstance(value, pd.DataFrame)
            value.to_csv(path, index=False, lineterminator="\n")
        else:
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        artifact_hashes[name] = _sha256(path)

    accepted = bool(
        decision["passes_selection_gate"] and joint["passes_confirmation_gate"]
    )
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E810_productive_numeric_elaboration",
        "decision": "accept" if accepted else "reject",
        "passes_all_frozen_gates": accepted,
        "source_run_id": SOURCE_RUN_ID,
        "target_free_discovery_sha256": EXPECTED_DISCOVERY_REPORT_SHA256,
        "feature_cache_sha256": EXPECTED_FEATURE_CACHE_SHA256,
        "target_free_discovery": {
            "passes_target_free_gate": discovery["passes_target_free_gate"],
            "predictability": discovery["predictability"],
            "design_rank": discovery["design_rank"],
            "v05_prediction_correlations": discovery["v05_prediction_correlations"],
        },
        "lineage": {
            "features": list(FEATURE_NAMES),
            "quiet_threshold_student_words": MIN_STUDENT_WORDS,
            "quiet_validation_fallback": "fold training prior",
            "estimator": ("fold-local StandardScaler plus LogisticRegression"),
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "max_iterations": MAX_ITERATIONS,
            "seed": SEED,
            "raw_v05_weights": V05_WEIGHTS,
            "numeric_blend_weights": list(BLEND_WEIGHTS),
            "selection": (
                "lowest equal-environment, equal-fold macro log loss; "
                "ties use smaller blend weight"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_groups": ["session_id", "semantic_family"],
        },
        "fit_summaries": fit_summaries,
        "selection_gate": decision,
        "joint_confirmation": joint,
        "runtime": runtime,
        "runtime_metadata": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "validation_runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": artifact_hashes,
        "competition_outcomes_accessed": True,
        "V_joint_accessed": bool(joint["V_joint_accessed"]),
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "decision": report["decision"],
        "passes_all_frozen_gates": accepted,
        "selection_gate": decision,
        "joint_confirmation": joint,
        "artifact_sha256": artifact_hashes,
        "report_sha256": _sha256(report_path),
        "validation_runtime_seconds": report["validation_runtime_seconds"],
        "V_joint_accessed": report["V_joint_accessed"],
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E810 productive numeric elaboration validation."
    )
    parser.add_argument("stage", choices=("validate",))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
