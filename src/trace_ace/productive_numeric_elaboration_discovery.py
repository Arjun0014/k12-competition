from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trace_ace.io import discover_project_paths
from trace_ace.role_hard_validation import BEHAVIOR_COLUMNS


PROTOCOL_ID = "E810_productive_numeric_elaboration_target_free_v1"
SOURCE_URL = "https://blog.drivendata.org/blog/productive-math-talk-reference"
SEED = 20260729
FOLDS = 5
MIN_STUDENT_WORDS = 100
BENCHMARK_BATCHES = 8
BATCH_SIZE = 65_536
MAX_PROJECTED_SECONDS = 600.0
FEATURE_NAMES = (
    "n_student_words",
    "numeric_turns_per_word",
    "digit_chars_per_word",
)
RATIO_FEATURES = (
    "numeric_turns_per_word",
    "digit_chars_per_word",
)
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
DIGIT_RE = re.compile(r"\d")
EXPECTED_PUBLISHED_RESPONSE_SUMMARY = {
    "count": 35_062,
    "mean": {
        "n_student_words": 1003.214534,
        "numeric_turns_per_word": 0.044799,
        "digit_chars_per_word": 0.200411,
    },
    "std": {
        "n_student_words": 430.687493,
        "numeric_turns_per_word": 0.019231,
        "digit_chars_per_word": 0.084626,
    },
    "quiet_sessions": 107,
    "quiet_responses": 159,
}
PUBLISHED_STANDARDIZED_COEFFICIENTS = np.asarray(
    [0.068, -0.211, 0.076], dtype=np.float64
)
PUBLISHED_MEANS = np.asarray([1003.214534, 0.044799, 0.200411], dtype=np.float64)
PUBLISHED_STDS = np.asarray([430.687493, 0.019231, 0.084626], dtype=np.float64)
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
V05_WEIGHTS = {
    "pred_full": 0.25,
    "pred_role": 0.25,
    "pred_bge_base": 0.50,
}
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {"target", "is_correct", "correct", "label", "outcome"}
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_outcomes(frame: pd.DataFrame, label: str) -> None:
    forbidden = FORBIDDEN_OUTCOME_COLUMNS.intersection(
        str(column).lower() for column in frame.columns
    )
    if forbidden:
        raise ValueError(f"{label} unexpectedly exposes outcomes: {sorted(forbidden)}")


def assert_runtime(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    expected_prefix = (root / ".venv").resolve()
    observed_prefix = Path(sys.prefix).resolve()
    observed = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python_prefix": str(observed_prefix),
    }
    if (
        observed["python"] != "3.12.8"
        or observed["scikit_learn"] != "1.8.0"
        or observed_prefix != expected_prefix
    ):
        raise RuntimeError(f"E810 requires the frozen project .venv: {observed}")
    return observed


def _fold(session_id: object) -> int:
    digest = hashlib.sha256(f"E810|target-free|{session_id}".encode()).hexdigest()
    return int(digest[:16], 16) % FOLDS


def count_student_turn(content: object) -> tuple[int, int, int]:
    text = "" if content is None else str(content)
    return (
        len(WORD_RE.findall(text)),
        int(bool(DIGIT_RE.search(text))),
        len(DIGIT_RE.findall(text)),
    )


def _accumulate_batch(
    batch: pa.RecordBatch,
    counts: dict[str, list[int]],
) -> int:
    student = batch.filter(
        pc.equal(batch.column("role"), pa.scalar("student", pa.string()))
    )
    session_ids = student.column("session_id").to_pylist()
    contents = student.column("content").to_pylist()
    for session_id, content in zip(session_ids, contents, strict=True):
        words, numeric_turn, digit_chars = count_student_turn(content)
        values = counts.setdefault(str(session_id), [0, 0, 0])
        values[0] += words
        values[1] += numeric_turn
        values[2] += digit_chars
    return len(student)


def extract_session_features(
    utterance_path: str | Path,
    session_ids: Sequence[object],
    *,
    max_batches: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, list[int]] = {}
    parquet = pq.ParquetFile(utterance_path)
    scanned_rows = 0
    student_turns = 0
    batches = 0
    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["session_id", "role", "content"],
    ):
        scanned_rows += len(batch)
        student_turns += _accumulate_batch(batch, counts)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    rows: list[dict[str, object]] = []
    for session_id in session_ids:
        words, numeric_turns, digit_chars = counts.get(str(session_id), [0, 0, 0])
        denominator = float(words) if words else math.nan
        rows.append(
            {
                "session_id": str(session_id),
                "n_student_words": denominator,
                "numeric_turns_per_word": numeric_turns / denominator,
                "digit_chars_per_word": digit_chars / denominator,
            }
        )
    frame = pd.DataFrame(rows)
    _assert_no_outcomes(frame, "E810 extracted features")
    return frame, {
        "parquet_rows": parquet.metadata.num_rows,
        "scanned_rows": scanned_rows,
        "student_turns": student_turns,
        "batches": batches,
    }


def _v05_dense_behavior(
    behavior: pd.DataFrame,
    session_features: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    merged = behavior.merge(
        session_features,
        on="session_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(behavior):
        raise ValueError("E810 existing feature caches do not align one-to-one.")
    values = merged[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float64)
    transformed = np.log1p(np.maximum(values, 0.0))
    eps = 1.0
    student_words = merged["student_words"].to_numpy(dtype=np.float64)
    tutor_words = merged["tutor_words"].to_numpy(dtype=np.float64)
    closing_student = merged["closing_student_words"].to_numpy(dtype=np.float64)
    closing_tutor = merged["closing_tutor_words"].to_numpy(dtype=np.float64)
    ratios = np.column_stack(
        [
            student_words / (tutor_words + eps),
            closing_student / (student_words + eps),
            closing_tutor / (tutor_words + eps),
            merged["student_questions"].to_numpy(dtype=np.float64)
            / (student_words + eps),
            merged["tutor_affirmations"].to_numpy(dtype=np.float64)
            / (tutor_words + eps),
            merged["tutor_corrections"].to_numpy(dtype=np.float64)
            / (tutor_words + eps),
            merged["student_uncertainty"].to_numpy(dtype=np.float64)
            / (student_words + eps),
            merged["student_reasoning_cues"].to_numpy(dtype=np.float64)
            / (student_words + eps),
        ]
    )
    session_columns = [
        "n_utterances",
        "duration_seconds",
        "content_chars",
        "tutor_utterances",
        "student_utterances",
        "background_utterances",
        "other_role_utterances",
        "unclear_utterances",
        "empty_utterances",
    ]
    session_values = np.log1p(
        np.maximum(
            merged[session_columns].to_numpy(dtype=np.float64),
            0.0,
        )
    )
    names = [
        *[f"log1p_{name}" for name in BEHAVIOR_COLUMNS],
        "student_tutor_word_ratio",
        "closing_student_share",
        "closing_tutor_share",
        "student_question_rate",
        "tutor_affirmation_rate",
        "tutor_correction_rate",
        "student_uncertainty_rate",
        "student_reasoning_rate",
        *[f"log1p_{name}" for name in session_columns],
    ]
    matrix = np.column_stack([transformed, ratios, session_values])
    if matrix.shape != (len(merged), len(names)) or not np.isfinite(matrix).all():
        raise RuntimeError("E810 existing dense feature matrix is invalid.")
    return matrix, names


def _crossfit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(FOLDS):
        training = folds != fold
        validation = ~training
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(matrix[training], target[training])
        prediction[validation] = model.predict(matrix[validation])
    if not np.isfinite(prediction).all():
        raise RuntimeError("E810 target-free Ridge cross-fit is incomplete.")
    return prediction


def _crossfit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(FOLDS):
        training = folds != fold
        validation = ~training
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1_000,
                random_state=SEED,
            ),
        )
        model.fit(matrix[training], target[training])
        prediction[validation] = model.predict_proba(matrix[validation])[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("E810 synthetic logistic cross-fit is incomplete.")
    return prediction


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    keep = scale > 1e-12
    return (matrix[:, keep] - mean[keep]) / scale[keep]


def _published_summary(
    response_features: pd.DataFrame,
    session_features: pd.DataFrame,
) -> tuple[dict[str, object], bool]:
    describe = response_features[list(FEATURE_NAMES)].describe()
    means = {name: float(describe.loc["mean", name]) for name in FEATURE_NAMES}
    stds = {name: float(describe.loc["std", name]) for name in FEATURE_NAMES}
    quiet_sessions = int(
        (session_features["n_student_words"].fillna(0.0) < MIN_STUDENT_WORDS).sum()
    )
    quiet_ids = set(
        session_features.loc[
            session_features["n_student_words"].fillna(0.0) < MIN_STUDENT_WORDS,
            "session_id",
        ]
    )
    quiet_responses = int(response_features["session_id"].isin(quiet_ids).sum())
    rounded_match = bool(
        int(describe.loc["count", FEATURE_NAMES[0]])
        == EXPECTED_PUBLISHED_RESPONSE_SUMMARY["count"]
        and quiet_sessions == EXPECTED_PUBLISHED_RESPONSE_SUMMARY["quiet_sessions"]
        and quiet_responses == EXPECTED_PUBLISHED_RESPONSE_SUMMARY["quiet_responses"]
        and all(
            round(means[name], 6) == EXPECTED_PUBLISHED_RESPONSE_SUMMARY["mean"][name]
            and round(stds[name], 6) == EXPECTED_PUBLISHED_RESPONSE_SUMMARY["std"][name]
            for name in FEATURE_NAMES
        )
    )
    return {
        "response_count": len(response_features),
        "nonmissing_response_count": int(
            response_features[FEATURE_NAMES[0]].notna().sum()
        ),
        "session_count": len(session_features),
        "means": means,
        "stds": stds,
        "quiet_sessions": quiet_sessions,
        "quiet_responses": quiet_responses,
    }, rounded_match


def _prediction_correlations(
    project_root: str | Path,
    response_features: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    root = Path(project_root).resolve()
    usable = response_features.loc[
        response_features["n_student_words"].ge(MIN_STUDENT_WORDS),
        ["response_id", *FEATURE_NAMES],
    ].copy()
    standardized = (
        usable[list(FEATURE_NAMES)].to_numpy(dtype=np.float64) - PUBLISHED_MEANS
    ) / PUBLISHED_STDS
    usable["published_numeric_score"] = (
        standardized @ PUBLISHED_STANDARDIZED_COEFFICIENTS
    )
    source = root / "experiments" / "runs" / SOURCE_RUN_ID
    results: dict[str, dict[str, float | int]] = {}
    for environment in SELECTION_ENVIRONMENTS:
        path = source / f"component_oof_{environment}.parquet"
        columns = ["response_id", *V05_WEIGHTS]
        component = pd.read_parquet(path, columns=columns)
        _assert_no_outcomes(component, f"E810 {environment} predictions")
        component["pred_v05_raw"] = sum(
            weight * component[name].to_numpy(dtype=np.float64)
            for name, weight in V05_WEIGHTS.items()
        )
        aligned = usable.merge(
            component[["response_id", "pred_v05_raw"]],
            on="response_id",
            how="inner",
            validate="one_to_one",
        )
        results[environment] = {
            "rows": len(aligned),
            "pearson": float(
                aligned["published_numeric_score"].corr(
                    aligned["pred_v05_raw"], method="pearson"
                )
            ),
            "spearman": float(
                aligned["published_numeric_score"].corr(
                    aligned["pred_v05_raw"], method="spearman"
                )
            ),
        }
    return results


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime(project_root)
    paths = discover_project_paths(project_root)
    features = pd.read_csv(
        paths.train_features,
        usecols=["session_id"],
        dtype={"session_id": "string"},
    )
    session_ids = features["session_id"].drop_duplicates().tolist()
    started = time.perf_counter()
    _, scan = extract_session_features(
        paths.cache_dir / "utterances.parquet",
        session_ids,
        max_batches=BENCHMARK_BATCHES,
    )
    elapsed = time.perf_counter() - started
    projected = elapsed * scan["parquet_rows"] / scan["scanned_rows"]
    result = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "scanned_rows": scan["scanned_rows"],
        "total_rows": scan["parquet_rows"],
        "student_turns_in_sample": scan["student_turns"],
        "elapsed_seconds": elapsed,
        "projected_full_seconds": projected,
        "proceed": bool(projected <= MAX_PROJECTED_SECONDS),
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    output = paths.cache_dir / "productive_numeric_elaboration_e810_benchmark.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["benchmark_path"] = str(output)
    result["benchmark_sha256"] = _sha256(output)
    return result


def audit(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime(project_root)
    paths = discover_project_paths(project_root)
    response_index = pd.read_csv(
        paths.train_features,
        usecols=["response_id", "session_id"],
        dtype={"response_id": "string", "session_id": "string"},
    )
    _assert_no_outcomes(response_index, "E810 response index")
    session_ids = response_index["session_id"].drop_duplicates().tolist()
    session_numeric, scan = extract_session_features(
        paths.cache_dir / "utterances.parquet",
        session_ids,
    )
    response_numeric = response_index.merge(
        session_numeric,
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    published, published_match = _published_summary(response_numeric, session_numeric)

    behavior = pd.read_parquet(paths.cache_dir / "session_behavior.parquet")
    basic = pd.read_parquet(paths.cache_dir / "session_features.parquet")
    _assert_no_outcomes(behavior, "E810 role behavior")
    _assert_no_outcomes(basic, "E810 session features")
    existing, existing_names = _v05_dense_behavior(behavior, basic)
    existing_order = behavior["session_id"].astype(str)
    numeric = (
        session_numeric.set_index("session_id").loc[list(existing_order)].reset_index()
    )
    usable = numeric["n_student_words"].notna().to_numpy()
    existing_usable = existing[usable]
    new_usable = numeric.loc[usable, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    new_design = new_usable.copy()
    new_design[:, 0] = np.log1p(new_design[:, 0])
    folds = np.asarray(
        [_fold(value) for value in numeric.loc[usable, "session_id"]],
        dtype=np.int8,
    )

    predictability: dict[str, dict[str, float | str]] = {}
    residual_columns: list[np.ndarray] = []
    for offset, name in enumerate(FEATURE_NAMES):
        target = new_design[:, offset]
        prediction = _crossfit_ridge(existing_usable, target, folds)
        residual_columns.append(target - prediction)
        correlations = [
            pd.Series(target).corr(
                pd.Series(existing_usable[:, column]), method="spearman"
            )
            for column in range(existing_usable.shape[1])
        ]
        finite = [
            (abs(float(value)), existing_names[column], float(value))
            for column, value in enumerate(correlations)
            if pd.notna(value)
        ]
        nearest = max(finite, key=lambda item: item[0])
        predictability[name] = {
            "crossfit_r2_from_existing_dense": float(r2_score(target, prediction)),
            "nearest_existing_feature": nearest[1],
            "nearest_abs_spearman": nearest[0],
            "nearest_signed_spearman": nearest[2],
        }

    existing_standardized = _standardize(existing_usable)
    augmented_standardized = np.column_stack(
        [existing_standardized, _standardize(new_design)]
    )
    existing_rank = int(np.linalg.matrix_rank(existing_standardized))
    augmented_rank = int(np.linalg.matrix_rank(augmented_standardized))
    rank_gain = augmented_rank - existing_rank

    residual = np.column_stack(residual_columns)
    residual = _standardize(residual)
    synthetic_logit = (
        0.25 * residual[:, 0] - 1.00 * residual[:, 1] + 0.65 * residual[:, 2]
    )
    synthetic_probability = expit(np.clip(synthetic_logit, -5.0, 5.0))
    rng = np.random.default_rng(SEED)
    synthetic_target = rng.binomial(1, synthetic_probability).astype(np.int8)
    synthetic_baseline = _crossfit_logistic(existing_usable, synthetic_target, folds)
    synthetic_candidate = _crossfit_logistic(
        np.column_stack([existing_usable, new_design]),
        synthetic_target,
        folds,
    )
    synthetic = {
        "rows": len(synthetic_target),
        "positive_rate": float(synthetic_target.mean()),
        "existing_dense_log_loss": float(
            log_loss(synthetic_target, synthetic_baseline)
        ),
        "augmented_log_loss": float(log_loss(synthetic_target, synthetic_candidate)),
    }
    synthetic["log_loss_gain"] = (
        synthetic["existing_dense_log_loss"] - synthetic["augmented_log_loss"]
    )

    prediction_correlations = _prediction_correlations(project_root, response_numeric)
    ratio_nonredundant = any(
        predictability[name]["crossfit_r2_from_existing_dense"] <= 0.90
        and predictability[name]["nearest_abs_spearman"] <= 0.95
        for name in RATIO_FEATURES
    )
    mean_abs_v05_spearman = float(
        np.mean(
            [
                abs(float(values["spearman"]))
                for values in prediction_correlations.values()
            ]
        )
    )
    clauses = {
        "published_summary_reproduced_to_six_decimals": published_match,
        "response_feature_coverage_at_least_0_995": (
            published["nonmissing_response_count"] / published["response_count"]
            >= 0.995
        ),
        "augmented_design_adds_at_least_two_rank_dimensions": rank_gain >= 2,
        "at_least_one_numeric_ratio_is_nonredundant": ratio_nonredundant,
        "synthetic_residual_log_loss_gain_at_least_0_002": (
            synthetic["log_loss_gain"] >= 0.002
        ),
        "mean_abs_spearman_with_v05_at_most_0_50": (mean_abs_v05_spearman <= 0.50),
    }

    feature_path = (
        paths.cache_dir / "productive_numeric_elaboration_e810_target_free.parquet"
    )
    session_numeric.to_parquet(feature_path, index=False)
    source_paths = {
        "train_features": paths.train_features,
        "utterances": paths.cache_dir / "utterances.parquet",
        "session_behavior": paths.cache_dir / "session_behavior.parquet",
        "session_features": paths.cache_dir / "session_features.parquet",
        **{
            f"component_oof_{environment}": (
                paths.experiments_dir
                / "runs"
                / SOURCE_RUN_ID
                / f"component_oof_{environment}.parquet"
            )
            for environment in SELECTION_ENVIRONMENTS
        },
    }
    timestamp = datetime.now(timezone.utc)
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": timestamp.isoformat(),
        "source_url": SOURCE_URL,
        "runtime": runtime,
        "lineage": {
            "feature_names": list(FEATURE_NAMES),
            "word_regex": WORD_RE.pattern,
            "digit_regex": DIGIT_RE.pattern,
            "min_student_words": MIN_STUDENT_WORDS,
            "target_free_folds": FOLDS,
            "target_free_seed": SEED,
            "ridge_alpha": 1.0,
            "synthetic_logit": (
                "0.25*residual_log_words - 1.00*residual_numeric_turns_per_word "
                "+ 0.65*residual_digit_chars_per_word"
            ),
            "source_hashes": {
                name: _sha256(path) for name, path in source_paths.items()
            },
        },
        "scan": scan,
        "published_reproduction": published,
        "existing_dense_feature_count": len(existing_names),
        "existing_dense_feature_names": existing_names,
        "predictability": predictability,
        "design_rank": {
            "existing": existing_rank,
            "augmented": augmented_rank,
            "gain": rank_gain,
        },
        "synthetic_benchmark": synthetic,
        "v05_prediction_correlations": prediction_correlations,
        "mean_abs_v05_spearman": mean_abs_v05_spearman,
        "clauses": clauses,
        "passes_target_free_gate": bool(all(clauses.values())),
        "runtime_seconds": time.perf_counter() - started,
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = (
        paths.cache_dir / "productive_numeric_elaboration_e810_target_free_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "passes_target_free_gate": report["passes_target_free_gate"],
        "clauses": clauses,
        "published_reproduction": published,
        "predictability": predictability,
        "design_rank": report["design_rank"],
        "synthetic_benchmark": synthetic,
        "v05_prediction_correlations": prediction_correlations,
        "runtime_seconds": report["runtime_seconds"],
        "feature_cache_sha256": report["feature_cache_sha256"],
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run target-free E810 productive numeric elaboration discovery."
    )
    parser.add_argument("stage", choices=("benchmark", "audit"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = (
        benchmark(args.project_root)
        if args.stage == "benchmark"
        else audit(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
