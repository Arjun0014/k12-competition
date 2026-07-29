from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
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

from trace_ace.curriculum_progression_discovery import STAGE_LABELS
from trace_ace.hard_validation import _fold_masks
from trace_ace.metrics import binary_metrics
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E830_curriculum_progression_competition_screen_v1"
TARGET_FREE_PROTOCOL_ID = "E830_curriculum_progression_target_free_v1"
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
EXPECTED_FEATURE_SHA256 = (
    "2391a8f5d807323c2802086e8b8f78d126b0115f4ad71531deaac267c453ec19"
)
EXPECTED_REPORT_SHA256 = (
    "5b53d3f6fd3d34179c4a2f885d6592538d82d77090a74f1682bdb53d93ca9a2b"
)
SEED = 20260730
REGULARIZATION_C = 0.1
MAX_ITERATIONS = 1_000
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6
V05_WEIGHTS = {
    "pred_full": 0.25,
    "pred_role": 0.25,
    "pred_bge_base": 0.50,
}
FEATURE_COLUMNS = (
    *(f"curriculum_score_{label}" for label in STAGE_LABELS),
    *(f"curriculum_probability_{label}" for label in STAGE_LABELS),
    "curriculum_expected_stage",
    "curriculum_max_similarity",
    "curriculum_margin",
    "curriculum_entropy",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "root": root,
        "features": root / "data_cache" / "curriculum_progression_e830_target_free.parquet",
        "target_free_report": (
            root
            / "data_cache"
            / "curriculum_progression_e830_target_free_report.json"
        ),
        "experiments": root / "experiments",
    }


def assert_runtime(project_root: str | Path) -> dict[str, str]:
    expected = (
        Path(project_root).resolve() / ".venv" / "Scripts" / "python.exe"
    ).resolve()
    observed = Path(sys.executable).resolve()
    if observed != expected:
        raise RuntimeError(f"E830 requires {expected}; observed {observed}.")
    if platform.python_version() != "3.12.8":
        raise RuntimeError("E830 requires Python 3.12.8.")
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError("E830 requires scikit-learn 1.8.0.")
    return {
        "python_executable": str(observed),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }


def verify_target_free(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    if _sha256(paths["features"]) != EXPECTED_FEATURE_SHA256:
        raise ValueError("E830 target-free feature SHA-256 changed.")
    if _sha256(paths["target_free_report"]) != EXPECTED_REPORT_SHA256:
        raise ValueError("E830 target-free report SHA-256 changed.")
    report = json.loads(paths["target_free_report"].read_text(encoding="utf-8"))
    required = {
        "protocol_id": TARGET_FREE_PROTOCOL_ID,
        "passes_target_free_gate": True,
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    observed = {key: report.get(key) for key in required}
    if observed != required:
        raise ValueError(f"E830 target-free contract changed: {observed}")
    if not all(report["gate_clauses"].values()):
        raise ValueError("E830 target-free report contains a failed clause.")
    return report


def load_curriculum_features(project_root: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(_paths(project_root)["features"])
    required = {"learning_objective_id", *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"E830 feature cache is missing columns: {missing}")
    if len(frame) != 398 or frame["learning_objective_id"].duplicated().any():
        raise ValueError("E830 feature cache must contain 398 unique objectives.")
    values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("E830 feature cache contains non-finite values.")
    return frame[["learning_objective_id", *FEATURE_COLUMNS]].copy()


def align_curriculum_features(
    component: pd.DataFrame,
    features: pd.DataFrame,
) -> np.ndarray:
    aligned = component[["learning_objective_id"]].merge(
        features,
        on="learning_objective_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(aligned) != len(component):
        raise ValueError("E830 feature alignment changed row count.")
    values = aligned[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("At least one component row lacks E830 features.")
    return values


def _objective_equal_weights(objective_ids: pd.Series) -> np.ndarray:
    counts = objective_ids.value_counts()
    weights = objective_ids.map(counts).rdiv(1.0).to_numpy(dtype=np.float64)
    return weights / weights.mean()


def fit_curriculum_oof(
    component: pd.DataFrame,
    values: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if values.shape != (len(component), len(FEATURE_COLUMNS)):
        raise ValueError("E830 aligned feature shape changed.")
    folds = component["fold"].to_numpy(dtype=np.int8)
    target = component["target"].to_numpy(dtype=np.int8)
    prediction = np.full(len(component), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        training_mask, validation_mask = _fold_masks(component, folds, fold)
        training_sessions = set(component.loc[training_mask, "session_id"].astype(str))
        validation_sessions = set(
            component.loc[validation_mask, "session_id"].astype(str)
        )
        if training_sessions & validation_sessions:
            raise RuntimeError(f"E830 session leakage in fold {fold}.")
        training_objectives = component.loc[
            training_mask, "learning_objective_id"
        ].astype(str)
        sample_weight = _objective_equal_weights(training_objectives)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=REGULARIZATION_C,
                solver="lbfgs",
                max_iter=MAX_ITERATIONS,
                random_state=SEED,
            ),
        )
        model.fit(
            values[training_mask],
            target[training_mask],
            logisticregression__sample_weight=sample_weight,
        )
        prediction[validation_mask] = model.predict_proba(
            values[validation_mask]
        )[:, 1]
        logistic = model.named_steps["logisticregression"]
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(training_mask.sum()),
                "training_sessions": len(training_sessions),
                "training_objectives": int(training_objectives.nunique()),
                "validation_rows": int(validation_mask.sum()),
                "validation_sessions": len(validation_sessions),
                "validation_objectives": int(
                    component.loc[
                        validation_mask, "learning_objective_id"
                    ].nunique()
                ),
                "sample_weight_min": float(sample_weight.min()),
                "sample_weight_max": float(sample_weight.max()),
                "sample_weight_mean": float(sample_weight.mean()),
                "standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(
                        FEATURE_COLUMNS, logistic.coef_[0], strict=True
                    )
                },
                "intercept": float(logistic.intercept_[0]),
                "iterations": int(logistic.n_iter_[0]),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("E830 OOF predictions are incomplete.")
    return np.clip(prediction, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP), summaries


def _baseline_prediction(component: pd.DataFrame) -> np.ndarray:
    return sum(
        weight * component[name].to_numpy(dtype=np.float64)
        for name, weight in V05_WEIGHTS.items()
    )


def build_prediction_rows(
    component: pd.DataFrame,
    curriculum_prediction: np.ndarray,
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
        candidate["curriculum_weight"] = float(weight)
        candidate["pred_v05_raw"] = baseline
        candidate["pred_curriculum"] = curriculum_prediction
        candidate["prediction"] = np.clip(
            (1.0 - weight) * baseline + weight * curriculum_prediction,
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
        ["environment", "curriculum_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        candidate = binary_metrics(target, scored["prediction"])
        row: dict[str, object] = {
            "environment": str(environment),
            "curriculum_weight": float(weight),
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
        folds.groupby(
            ["environment", "curriculum_weight"], as_index=False, sort=True
        )[columns]
        .mean()
        .sort_values(["environment", "curriculum_weight"], kind="mergesort")
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
            environment_metrics["curriculum_weight"].eq(weight)
        ]
        selected_predictions = predictions.loc[
            predictions["curriculum_weight"].eq(weight)
        ]
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
        weight_folds = fold_metrics.loc[
            fold_metrics["curriculum_weight"].eq(weight)
        ]
        rows.append(
            {
                "curriculum_weight": weight,
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
        ["mean_log_loss", "curriculum_weight"], kind="mergesort"
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
    key = f"{selected['curriculum_weight']:.2f}"
    return selection.reset_index(drop=True), {
        "selected_weight": float(selected["curriculum_weight"]),
        "selected_row": selected,
        "clauses": clauses,
        "passes_selection_gate": bool(all(clauses.values())),
        "selected_bootstrap": bootstraps[key],
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
    values = align_curriculum_features(component, features)
    curriculum, fits = fit_curriculum_oof(component, values)
    predictions = build_prediction_rows(component, curriculum, [selected_weight])
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
        "joint_auroc_non_regression": (
            float(row["delta_roc_auc_vs_v05_raw"]) >= 0.0
        ),
        "joint_brier_non_regression": (
            float(row["delta_brier_score_vs_v05_raw"]) <= 0.0
        ),
        "joint_ece_non_regression": (
            float(row["delta_ece_10_vs_v05_raw"]) <= 0.0
        ),
        "joint_session_bootstrap_support_at_least_0_95": (
            session_bootstrap["support_positive_gain"] >= 0.95
        ),
        "joint_semantic_family_bootstrap_support_at_least_0_95": (
            family_bootstrap["support_positive_gain"] >= 0.95
        ),
    }
    return predictions, folds, {
        "environment_metrics": row.to_dict(),
        "fit_summaries": fits,
        "session_bootstrap": session_bootstrap,
        "semantic_family_bootstrap": family_bootstrap,
        "clauses": clauses,
        "passes_confirmation_gate": bool(all(clauses.values())),
        "V_joint_accessed": True,
        "V_final_accessed": False,
    }


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime(project_root)
    discovery = verify_target_free(project_root)
    features = load_curriculum_features(project_root)
    prediction_frames: list[pd.DataFrame] = []
    fit_summaries: dict[str, list[dict[str, object]]] = {}
    for environment in SELECTION_ENVIRONMENTS:
        component = load_component_oof(project_root, environment)
        values = align_curriculum_features(component, features)
        curriculum, summaries = fit_curriculum_oof(component, values)
        fit_summaries[environment] = summaries
        prediction_frames.append(
            build_prediction_rows(component, curriculum, BLEND_WEIGHTS)
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_curriculum_progression")
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
        "candidate": "E830_curriculum_progression_objective_head",
        "decision": "accept" if accepted else "reject",
        "passes_all_frozen_gates": accepted,
        "source_run_id": SOURCE_RUN_ID,
        "target_free_feature_sha256": EXPECTED_FEATURE_SHA256,
        "target_free_report_sha256": EXPECTED_REPORT_SHA256,
        "target_free_evidence": {
            "passes_target_free_gate": discovery["passes_target_free_gate"],
            "curriculum_leave_one_out": discovery["curriculum_leave_one_out"],
            "anchor_audit": discovery["anchor_audit"],
            "bge_redundancy_audit": discovery["bge_redundancy_audit"],
            "synthetic_progression_benchmark": discovery[
                "synthetic_progression_benchmark"
            ],
        },
        "lineage": {
            "features": list(FEATURE_COLUMNS),
            "estimator": "fold-local StandardScaler plus LogisticRegression",
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "max_iterations": MAX_ITERATIONS,
            "sample_weight": "inverse training-objective frequency, mean one",
            "seed": SEED,
            "raw_v05_weights": V05_WEIGHTS,
            "curriculum_blend_weights": list(BLEND_WEIGHTS),
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
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        description="Run the frozen E830 curriculum-progression validation."
    )
    parser.add_argument("stage", choices=("validate",))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
