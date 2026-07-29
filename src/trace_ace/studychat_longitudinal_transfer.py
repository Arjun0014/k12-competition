from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROTOCOL_ID = "E840_studychat_longitudinal_assessment_transfer_v1"
SEED = 20260730
RIDGE_ALPHA = 10.0
BOOTSTRAP_REPLICATES = 5_000
SOURCE_REVISION = "24d7987d9fbb30d9da12acc53455a10f1cdd2d7f"
EXPECTED_HASHES = {
    "data.jsonl": "67927f73327639904c417c2f6e6200e11427920a8435b39cb7c87b5c6601e130",
    "scores/f24_grades_released_normalized.csv": (
        "d16d54c8eaa9f3723755362f0dcb651740d070bdcf5c739c7f9a27a77db5a244"
    ),
    "scores/s25_grades_released_normalized.csv": (
        "06f1a4041e1631a9787c5eac842f04e63dad1480609ac5376fcab351a1fad5ec"
    ),
    "LICENSE": "0caefd1999143ddb9e3eb93dfa802680a8ab924d95d2acc1fee434da6d66d6fc",
    "README.md": "86a5926deec67d5e848c930363829d53339bb601c7cc5d6064bdf917abd1e5c0",
}
EXAM_TOPIC_LIMITS = {"e1": 2, "e2": 4, "e3": 7}

QUESTION_RE = re.compile(r"\?|^\s*(?:how|why|what|when|where|which|can|could|should|is|are|do|does)\b", re.I)
EXPLANATION_RE = re.compile(r"\b(?:explain|understand|concept|meaning|intuition|walk me through|break down)\b", re.I)
WHY_HOW_RE = re.compile(r"\b(?:why|how)\b", re.I)
VERIFY_RE = re.compile(r"\b(?:is this right|am i right|check|verify|correct\?|does this work|review)\b", re.I)
DIRECT_RE = re.compile(r"\b(?:give me (?:the )?answer|just (?:give|tell|write)|solve it|do it for me|complete this)\b", re.I)
WRITING_RE = re.compile(r"\b(?:write (?:my|the|a) report|rewrite|paraphrase|make this sound|report section|essay)\b", re.I)
CODE_REQUEST_RE = re.compile(r"\b(?:write|generate|fix|debug|implement|code|function|class|python)\b", re.I)
CONFUSION_RE = re.compile(r"\b(?:confus|stuck|don'?t understand|not sure|lost|help me)\b", re.I)
SELF_EXPLANATION_RE = re.compile(r"\b(?:i think|because|my reasoning|i understand|so basically|what i did)\b", re.I)
METACOGNITIVE_RE = re.compile(r"\b(?:key takeaway|learn|strategy|approach|next step|mistake|where did i go wrong)\b", re.I)
TUTOR_EXPLANATION_RE = re.compile(r"\b(?:because|this means|for example|in other words|the key|step \d|here'?s how)\b", re.I)
CODE_MARK_RE = re.compile(r"```|\b(?:def|class|import|return|for|while|if)\b|[{}();=]", re.I)
NUMBER_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z0-9_']+")

BEHAVIOR_FEATURES = (
    "student_question_rate",
    "student_explanation_request_rate",
    "student_why_how_rate",
    "student_verification_rate",
    "student_direct_answer_rate",
    "student_writing_request_rate",
    "student_code_request_rate",
    "student_confusion_rate",
    "student_self_explanation_rate",
    "student_metacognitive_rate",
    "student_numeric_rate",
    "student_code_mark_rate",
    "tutor_question_rate",
    "tutor_explanation_rate",
    "tutor_code_mark_rate",
    "student_lexical_diversity",
    "mean_student_words",
    "mean_tutor_words",
    "tutor_student_word_ratio",
    "multi_turn_fraction",
)
BASELINE_FEATURES = (
    "semester_s25",
    "exam_e2",
    "exam_e3",
    "prior_assignment_mean",
    "prior_assignment_std",
    "prior_assignment_min",
    "prior_assignment_last",
    "log_interactions",
    "log_chats",
    "topic_coverage",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _privacy_hash(semester: str, user_id: str) -> str:
    payload = f"E840|StudyChat|{semester}|{user_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _words(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(str(text))]


def dialogue_features(
    student_texts: list[str],
    tutor_texts: list[str],
    multi_turn_fraction: float,
) -> dict[str, float]:
    student = [str(value) for value in student_texts if str(value).strip()]
    tutor = [str(value) for value in tutor_texts if str(value).strip()]
    if not student or not tutor:
        raise ValueError("E840 dialogue features require both roles.")
    student_words = [_words(text) for text in student]
    tutor_words = [_words(text) for text in tutor]

    def rate(pattern: re.Pattern[str], texts: list[str]) -> float:
        return float(np.mean([bool(pattern.search(text)) for text in texts]))

    flat_student = [word for row in student_words for word in row]
    mean_student = float(np.mean([len(row) for row in student_words]))
    mean_tutor = float(np.mean([len(row) for row in tutor_words]))
    return {
        "student_question_rate": rate(QUESTION_RE, student),
        "student_explanation_request_rate": rate(EXPLANATION_RE, student),
        "student_why_how_rate": rate(WHY_HOW_RE, student),
        "student_verification_rate": rate(VERIFY_RE, student),
        "student_direct_answer_rate": rate(DIRECT_RE, student),
        "student_writing_request_rate": rate(WRITING_RE, student),
        "student_code_request_rate": rate(CODE_REQUEST_RE, student),
        "student_confusion_rate": rate(CONFUSION_RE, student),
        "student_self_explanation_rate": rate(SELF_EXPLANATION_RE, student),
        "student_metacognitive_rate": rate(METACOGNITIVE_RE, student),
        "student_numeric_rate": rate(NUMBER_RE, student),
        "student_code_mark_rate": rate(CODE_MARK_RE, student),
        "tutor_question_rate": rate(QUESTION_RE, tutor),
        "tutor_explanation_rate": rate(TUTOR_EXPLANATION_RE, tutor),
        "tutor_code_mark_rate": rate(CODE_MARK_RE, tutor),
        "student_lexical_diversity": (
            float(len(set(flat_student)) / len(flat_student)) if flat_student else 0.0
        ),
        "mean_student_words": mean_student,
        "mean_tutor_words": mean_tutor,
        "tutor_student_word_ratio": mean_tutor / max(mean_student, 1.0),
        "multi_turn_fraction": float(multi_turn_fraction),
    }


def _studychat_root(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "Datasets"
        / "StudyChat"
        / SOURCE_REVISION
    )


def verify_source(project_root: str | Path) -> dict[str, object]:
    root = _studychat_root(project_root)
    observed = {name: _sha256(root / name) for name in EXPECTED_HASHES}
    if observed != EXPECTED_HASHES:
        raise ValueError(f"E840 StudyChat source hashes changed: {observed}")
    return {
        "revision": SOURCE_REVISION,
        "license": "CC BY 4.0",
        "hashes": observed,
    }


def _load_interactions(project_root: str | Path) -> list[dict[str, object]]:
    source = _studychat_root(project_root) / "data.jsonl"
    rows: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 16_851:
        raise ValueError("E840 expects 16,851 StudyChat interactions.")
    return rows


def build_external_samples(project_root: str | Path) -> pd.DataFrame:
    interactions = _load_interactions(project_root)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in interactions:
        key = (str(row["semester"]), str(row["userId"]), str(row["topic"]))
        grouped[key].append(row)
    rows: list[dict[str, object]] = []
    root = _studychat_root(project_root)
    for semester in ("f24", "s25"):
        grades = pd.read_csv(
            root / "scores" / f"{semester}_grades_released_normalized.csv",
            dtype={"userId": "string"},
        )
        for grade in grades.itertuples(index=False):
            user_id = str(grade.userId)
            for exam, topic_limit in EXAM_TOPIC_LIMITS.items():
                topics = [f"a{index}" for index in range(1, topic_limit + 1)]
                eligible = [
                    item
                    for topic in topics
                    for item in grouped.get((semester, user_id, topic), [])
                ]
                if not eligible:
                    continue
                eligible.sort(
                    key=lambda item: (
                        int(item["timestamp"]),
                        str(item["chatId"]),
                        int(item["interactionCount"]),
                    )
                )
                student_texts = [str(item["prompt"]) for item in eligible]
                tutor_texts = [str(item["response"]) for item in eligible]
                chats = {str(item["chatId"]) for item in eligible}
                multi_turn = np.mean(
                    [int(item["interactionCount"]) > 0 for item in eligible]
                )
                behavior = dialogue_features(
                    student_texts, tutor_texts, float(multi_turn)
                )
                assignments = np.asarray(
                    [float(getattr(grade, topic)) for topic in topics],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "user_hash": _privacy_hash(semester, user_id),
                        "semester": semester,
                        "exam": exam,
                        "target_exam_score": float(getattr(grade, exam)),
                        "semester_s25": float(semester == "s25"),
                        "exam_e2": float(exam == "e2"),
                        "exam_e3": float(exam == "e3"),
                        "prior_assignment_mean": float(assignments.mean()),
                        "prior_assignment_std": float(assignments.std()),
                        "prior_assignment_min": float(assignments.min()),
                        "prior_assignment_last": float(assignments[-1]),
                        "log_interactions": math.log1p(len(eligible)),
                        "log_chats": math.log1p(len(chats)),
                        "topic_coverage": float(
                            sum(
                                bool(grouped.get((semester, user_id, topic)))
                                for topic in topics
                            )
                            / len(topics)
                        ),
                        **behavior,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["user_hash", "exam"], kind="mergesort"
    ).reset_index(drop=True)
    if len(frame) != 507 or frame["user_hash"].nunique() != 175:
        raise ValueError("E840 external sample lineage changed.")
    return frame


def build_competition_transfer_features(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    utterances = pd.read_parquet(
        root / "data_cache" / "utterances.parquet",
        columns=["session_id", "utterance_id", "role", "content"],
    ).sort_values(["session_id", "utterance_id"], kind="mergesort")
    rows: list[dict[str, object]] = []
    for session_id, group in utterances.groupby("session_id", sort=True):
        roles = group["role"].astype(str).str.casefold()
        student = group.loc[roles.eq("student"), "content"].astype(str).tolist()
        tutor = group.loc[roles.eq("tutor"), "content"].astype(str).tolist()
        if not student or not tutor:
            continue
        behavior = dialogue_features(
            student,
            tutor,
            float(max(len(student) - 1, 0) / max(len(student), 1)),
        )
        rows.append({"session_id": str(session_id), **behavior})
    return pd.DataFrame(rows).sort_values("session_id", kind="mergesort").reset_index(
        drop=True
    )


def transfer_distribution_audit(
    external: pd.DataFrame,
    competition: pd.DataFrame,
) -> dict[str, object]:
    nonconstant_external = int(
        sum(external[name].nunique() > 1 for name in BEHAVIOR_FEATURES)
    )
    nonconstant_competition = int(
        sum(competition[name].nunique() > 1 for name in BEHAVIOR_FEATURES)
    )
    shifts: dict[str, float] = {}
    for name in BEHAVIOR_FEATURES:
        source = external[name].to_numpy(dtype=np.float64)
        target = competition[name].to_numpy(dtype=np.float64)
        q25, median, q75 = np.quantile(source, [0.25, 0.50, 0.75])
        scale = max(float(q75 - q25), 1e-3)
        shifts[name] = abs(float(np.median(target)) - float(median)) / scale
    within = int(sum(value <= 2.5 for value in shifts.values()))
    return {
        "external_rows": int(len(external)),
        "competition_sessions": int(len(competition)),
        "nonconstant_external_features": nonconstant_external,
        "nonconstant_competition_features": nonconstant_competition,
        "median_shift_iqr_units": shifts,
        "features_with_median_shift_at_most_2_5_iqr": within,
        "feature_overlap_fraction": within / len(BEHAVIOR_FEATURES),
    }


def _fit_oof(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    baseline_values = frame[list(BASELINE_FEATURES)].to_numpy(dtype=np.float64)
    candidate_values = frame[
        [*BASELINE_FEATURES, *BEHAVIOR_FEATURES]
    ].to_numpy(dtype=np.float64)
    target = frame["target_exam_score"].to_numpy(dtype=np.float64)
    groups = frame["user_hash"].astype(str).to_numpy()
    baseline_prediction = np.full(len(frame), np.nan, dtype=np.float64)
    candidate_prediction = np.full(len(frame), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (training, validation) in enumerate(
        splitter.split(candidate_values, target, groups)
    ):
        baseline = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        baseline.fit(baseline_values[training], target[training])
        candidate.fit(candidate_values[training], target[training])
        baseline_prediction[validation] = baseline.predict(
            baseline_values[validation]
        )
        candidate_prediction[validation] = candidate.predict(
            candidate_values[validation]
        )
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(len(training)),
                "validation_rows": int(len(validation)),
                "training_users": int(len(set(groups[training]))),
                "validation_users": int(len(set(groups[validation]))),
                "user_overlap": int(
                    len(set(groups[training]).intersection(groups[validation]))
                ),
                "candidate_standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(
                        [*BASELINE_FEATURES, *BEHAVIOR_FEATURES],
                        candidate.named_steps["ridge"].coef_,
                        strict=True,
                    )
                },
            }
        )
    return baseline_prediction, candidate_prediction, summaries


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "mae": float(mean_absolute_error(target, prediction)),
        "spearman": float(spearmanr(target, prediction).statistic),
    }


def _user_bootstrap(
    scored: pd.DataFrame,
) -> dict[str, float]:
    grouped = {
        name: group.index.to_numpy()
        for name, group in scored.groupby("user_hash", sort=True)
    }
    names = np.asarray(sorted(grouped))
    rng = np.random.default_rng(SEED)
    gains = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    target = scored["target_exam_score"].to_numpy(dtype=np.float64)
    baseline = scored["baseline_prediction"].to_numpy(dtype=np.float64)
    candidate = scored["candidate_prediction"].to_numpy(dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([grouped[name] for name in selected])
        gains[replicate] = (
            mean_squared_error(target[indices], baseline[indices]) ** 0.5
            - mean_squared_error(target[indices], candidate[indices]) ** 0.5
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "group_column": "user_hash",
        "observed_rmse_gain": float(
            mean_squared_error(target, baseline) ** 0.5
            - mean_squared_error(target, candidate) ** 0.5
        ),
        "bootstrap_mean_gain": float(gains.mean()),
        "ci_lower": float(np.quantile(gains, 0.025)),
        "ci_upper": float(np.quantile(gains, 0.975)),
        "support_positive_gain": float(np.mean(gains > 0.0)),
    }


def evaluate_external(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    baseline, candidate, fits = _fit_oof(frame)
    scored = frame[
        ["user_hash", "semester", "exam", "target_exam_score"]
    ].copy()
    scored["baseline_prediction"] = baseline
    scored["candidate_prediction"] = candidate
    pooled_baseline = _regression_metrics(
        scored["target_exam_score"].to_numpy(), baseline
    )
    pooled_candidate = _regression_metrics(
        scored["target_exam_score"].to_numpy(), candidate
    )
    cells: list[dict[str, object]] = []
    for (semester, exam), group in scored.groupby(["semester", "exam"], sort=True):
        target = group["target_exam_score"].to_numpy(dtype=np.float64)
        base_metrics = _regression_metrics(
            target, group["baseline_prediction"].to_numpy(dtype=np.float64)
        )
        candidate_metrics = _regression_metrics(
            target, group["candidate_prediction"].to_numpy(dtype=np.float64)
        )
        cells.append(
            {
                "semester": semester,
                "exam": exam,
                "rows": int(len(group)),
                **{f"baseline_{key}": value for key, value in base_metrics.items()},
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                "rmse_gain": base_metrics["rmse"] - candidate_metrics["rmse"],
                "mae_gain": base_metrics["mae"] - candidate_metrics["mae"],
                "spearman_gain": (
                    candidate_metrics["spearman"] - base_metrics["spearman"]
                ),
            }
        )
    cells_frame = pd.DataFrame(cells)
    bootstrap = _user_bootstrap(scored)
    summary: dict[str, object] = {
        "rows": int(len(scored)),
        "users": int(scored["user_hash"].nunique()),
        "pooled_baseline": pooled_baseline,
        "pooled_candidate": pooled_candidate,
        "pooled_rmse_gain": pooled_baseline["rmse"] - pooled_candidate["rmse"],
        "pooled_mae_gain": pooled_baseline["mae"] - pooled_candidate["mae"],
        "pooled_spearman_gain": (
            pooled_candidate["spearman"] - pooled_baseline["spearman"]
        ),
        "improved_rmse_cells": int((cells_frame["rmse_gain"] > 0.0).sum()),
        "worst_cell_rmse_gain": float(cells_frame["rmse_gain"].min()),
        "user_bootstrap": bootstrap,
    }
    return scored, summary, fits


def _gate(
    external: dict[str, object],
    transfer: dict[str, object],
) -> dict[str, bool]:
    return {
        "pooled_rmse_gain_at_least_0_010": (
            float(external["pooled_rmse_gain"]) >= 0.010
        ),
        "pooled_mae_gain_at_least_0_005": (
            float(external["pooled_mae_gain"]) >= 0.005
        ),
        "pooled_spearman_non_regression": (
            float(external["pooled_spearman_gain"]) >= 0.0
        ),
        "at_least_five_of_six_cells_improve_rmse": (
            int(external["improved_rmse_cells"]) >= 5
        ),
        "worst_cell_rmse_regression_at_most_0_005": (
            float(external["worst_cell_rmse_gain"]) >= -0.005
        ),
        "user_bootstrap_support_at_least_0_95": (
            float(external["user_bootstrap"]["support_positive_gain"]) >= 0.95
        ),
        "all_behavior_features_nonconstant_external": (
            int(transfer["nonconstant_external_features"]) == len(BEHAVIOR_FEATURES)
        ),
        "all_behavior_features_nonconstant_competition": (
            int(transfer["nonconstant_competition_features"])
            == len(BEHAVIOR_FEATURES)
        ),
        "at_least_75_percent_distribution_overlap": (
            float(transfer["feature_overlap_fraction"]) >= 0.75
        ),
    }


def run(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    source = verify_source(project_root)
    external = build_external_samples(project_root)
    competition = build_competition_transfer_features(project_root)
    transfer = transfer_distribution_audit(external, competition)
    scored, external_summary, fits = evaluate_external(external)
    clauses = _gate(external_summary, transfer)

    root = Path(project_root).resolve()
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_studychat_longitudinal"
    )
    run_dir = root / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    scored_path = run_dir / "external_oof_predictions.parquet"
    cells_path = run_dir / "external_cell_metrics.csv"
    transfer_path = run_dir / "competition_target_free_features.parquet"
    report_path = run_dir / "report.json"
    scored.to_parquet(scored_path, index=False)

    cell_rows: list[dict[str, object]] = []
    for (semester, exam), group in scored.groupby(["semester", "exam"], sort=True):
        target = group["target_exam_score"].to_numpy(dtype=np.float64)
        baseline = _regression_metrics(
            target, group["baseline_prediction"].to_numpy(dtype=np.float64)
        )
        candidate = _regression_metrics(
            target, group["candidate_prediction"].to_numpy(dtype=np.float64)
        )
        cell_rows.append(
            {
                "semester": semester,
                "exam": exam,
                "rows": len(group),
                **{f"baseline_{key}": value for key, value in baseline.items()},
                **{f"candidate_{key}": value for key, value in candidate.items()},
                "rmse_gain": baseline["rmse"] - candidate["rmse"],
                "mae_gain": baseline["mae"] - candidate["mae"],
                "spearman_gain": candidate["spearman"] - baseline["spearman"],
            }
        )
    pd.DataFrame(cell_rows).to_csv(cells_path, index=False, lineterminator="\n")
    competition.to_parquet(transfer_path, index=False)
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "configuration": {
            "seed": SEED,
            "ridge_alpha": RIDGE_ALPHA,
            "baseline_features": list(BASELINE_FEATURES),
            "behavior_features": list(BEHAVIOR_FEATURES),
            "split": "five-fold shuffled GroupKFold by anonymized learner",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "external_validation": external_summary,
        "fit_summaries": fits,
        "transfer_distribution_audit": transfer,
        "gate_clauses": clauses,
        "passes_external_and_transfer_gate": bool(all(clauses.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "external_oof_predictions": {
                "path": str(scored_path.relative_to(root)),
                "sha256": _sha256(scored_path),
            },
            "external_cell_metrics": {
                "path": str(cells_path.relative_to(root)),
                "sha256": _sha256(cells_path),
            },
            "competition_target_free_features": {
                "path": str(transfer_path.relative_to(root)),
                "sha256": _sha256(transfer_path),
            },
        },
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passes_external_and_transfer_gate": report[
            "passes_external_and_transfer_gate"
        ],
        "gate_clauses": clauses,
        "external_validation": external_summary,
        "transfer_distribution_audit": transfer,
        "artifact_sha256": {
            name: value["sha256"] for name, value in report["artifacts"].items()
        },
        "report_sha256": _sha256(report_path),
        "runtime_seconds": report["runtime_seconds"],
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen E840 StudyChat longitudinal external validation."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
