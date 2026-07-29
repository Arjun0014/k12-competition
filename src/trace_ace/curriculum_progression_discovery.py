from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trace_ace.io import discover_project_paths


PROTOCOL_ID: Final = "E830_curriculum_progression_target_free_v1"
SEED: Final = 20260730
EXPECTED_SOURCE_SHA256: Final = (
    "d8499da572edb7f1dbf9c42284dae265e121a8215692cff204c3a5d73cd6c490"
)
SOURCE_URL: Final = (
    "https://www.gov.uk/government/publications/"
    "national-curriculum-in-england-mathematics-programmes-of-study/"
    "national-curriculum-in-england-mathematics-programmes-of-study"
)
STAGE_LABELS: Final = (
    "year_1",
    "year_2",
    "year_3",
    "year_4",
    "year_5",
    "year_6",
    "key_stage_3",
    "key_stage_4",
)
STAGE_TITLES: Final = {
    "year 1 programme of study": "year_1",
    "year 2 programme of study": "year_2",
    "year 3 programme of study": "year_3",
    "year 4 programme of study": "year_4",
    "year 5 programme of study": "year_5",
    "year 6 programme of study": "year_6",
    "key stage 3": "key_stage_3",
    "key stage 4": "key_stage_4",
}
STAGE_INDEX: Final = {label: index + 1 for index, label in enumerate(STAGE_LABELS)}
TEMPERATURE: Final = 0.08
WORD_WEIGHT: Final = 0.55
CHAR_WEIGHT: Final = 0.45


@dataclass(frozen=True)
class Anchor:
    objective: str
    minimum_stage: int
    maximum_stage: int


ANCHORS: Final = (
    Anchor("Recognising and finding a half", 1, 2),
    Anchor("Subtracting within 20.", 1, 2),
    Anchor("Using addition and subtraction facts within 20.", 1, 2),
    Anchor("Using 2x, 5x and 10x table.", 2, 2),
    Anchor("Telling the time to five minutes.", 2, 2),
    Anchor("Comparing numbers up to 1,000.", 3, 3),
    Anchor("Adding and subtracting multiples of 10 up to 1,000", 3, 3),
    Anchor("Adding 4-digit numbers using the column method", 4, 4),
    Anchor("Comparing numbers up to 10,000", 4, 4),
    Anchor("Using multiplication and division facts up to 12 x 12.", 4, 4),
    Anchor("Comparing numbers up to 1,000,000", 5, 5),
    Anchor("Knowing the value of each digit in numbers with up to 3 decimal places", 5, 6),
    Anchor("Comparing numbers up to 10,000,000", 6, 6),
    Anchor("Adding and subtracting fractions with different denominators.", 6, 6),
    Anchor("Dividing using long division", 6, 6),
    Anchor("Forming and Solving Linear Equations.", 7, 7),
    Anchor("Working with y=mx+c", 7, 7),
    Anchor("Applying Pythagoras' Theorem", 7, 8),
    Anchor("Expanding Double Brackets", 7, 8),
    Anchor("Completing the Square", 8, 8),
    Anchor("Trigonometry: Using SOHCAHTOA.", 8, 8),
    Anchor("Using Circle Theorems", 8, 8),
    Anchor("Simplifying Surds", 8, 8),
    Anchor("Drawing and Reading from Histograms", 8, 8),
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _CurriculumParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stage: str | None = None
        self.domain = "overview"
        self._headings: list[tuple[str, list[str]]] = []
        self._blocks: list[tuple[str, list[str], str, str]] = []
        self.rows: list[dict[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in {"h2", "h3"}:
            self._headings.append((tag, []))
        if tag in {"li", "p"} and self.stage is not None:
            self._blocks.append((tag, [], self.stage, self.domain))

    def handle_data(self, data: str) -> None:
        for _, buffer in self._headings:
            buffer.append(data)
        for _, buffer, _, _ in self._blocks:
            buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._headings and self._headings[-1][0] == tag:
            _, buffer = self._headings.pop()
            title = _normalize(" ".join(buffer))
            if tag == "h2":
                self.stage = STAGE_TITLES.get(title.casefold())
                self.domain = "overview"
            elif self.stage is not None:
                self.domain = title or "overview"
        if self._blocks and self._blocks[-1][0] == tag:
            _, buffer, stage, domain = self._blocks.pop()
            text = _normalize(" ".join(buffer))
            if len(text) >= 20:
                self.rows.append(
                    {"stage": stage, "domain": domain, "statement": text}
                )


def parse_curriculum(path: str | Path) -> pd.DataFrame:
    parser = _CurriculumParser()
    parser.feed(Path(path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(parser.rows).drop_duplicates(
        ["stage", "domain", "statement"]
    )
    frame["stage_index"] = frame["stage"].map(STAGE_INDEX).astype(np.int8)
    return frame.sort_values(
        ["stage_index", "domain", "statement"], kind="mergesort"
    ).reset_index(drop=True)


def _vectorizers(corpus: list[str]) -> tuple[object, object]:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.995,
        sublinear_tf=True,
        norm="l2",
    ).fit(corpus)
    char = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    ).fit(corpus)
    return word, char


def _combined_similarity(
    left_text: list[str],
    right_text: list[str],
    word: TfidfVectorizer,
    char: TfidfVectorizer,
) -> np.ndarray:
    left_word = word.transform(left_text)
    right_word = word.transform(right_text)
    left_char = char.transform(left_text)
    right_char = char.transform(right_text)
    return (
        WORD_WEIGHT * (left_word @ right_word.T).toarray()
        + CHAR_WEIGHT * (left_char @ right_char.T).toarray()
    )


def _stage_scores(
    similarity: np.ndarray,
    standard_stages: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((similarity.shape[0], len(STAGE_LABELS)), dtype=np.float64)
    for stage in range(1, len(STAGE_LABELS) + 1):
        scores[:, stage - 1] = similarity[:, standard_stages == stage].max(axis=1)
    return scores


def _posterior_features(stage_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior = softmax(stage_scores / TEMPERATURE, axis=1)
    expected = posterior @ np.arange(1, len(STAGE_LABELS) + 1, dtype=np.float64)
    return posterior, expected


def curriculum_leave_one_out(
    standards: pd.DataFrame,
    similarity: np.ndarray,
) -> dict[str, float]:
    if similarity.shape != (len(standards), len(standards)):
        raise ValueError("Curriculum similarity matrix shape changed.")
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    stages = standards["stage_index"].to_numpy(dtype=np.int8)
    scores = _stage_scores(work, stages)
    posterior, expected = _posterior_features(scores)
    predicted = np.argmax(scores, axis=1) + 1
    correlation = float(spearmanr(stages, expected).statistic)
    return {
        "rows": int(len(standards)),
        "top1_accuracy": float(np.mean(predicted == stages)),
        "within_one_accuracy": float(np.mean(np.abs(predicted - stages) <= 1)),
        "mean_absolute_stage_error": float(np.mean(np.abs(expected - stages))),
        "spearman_true_vs_expected_stage": correlation,
        "posterior_mean_entropy": float(
            np.mean(-np.sum(posterior * np.log(np.clip(posterior, 1e-12, 1.0)), axis=1))
        ),
    }


def build_objective_features(
    objectives: pd.DataFrame,
    standards: pd.DataFrame,
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    scores = _stage_scores(
        similarity,
        standards["stage_index"].to_numpy(dtype=np.int8),
    )
    posterior, expected = _posterior_features(scores)
    predicted = np.argmax(scores, axis=1) + 1
    sorted_scores = np.sort(scores, axis=1)
    output = objectives.copy()
    for index, label in enumerate(STAGE_LABELS):
        output[f"curriculum_score_{label}"] = scores[:, index]
        output[f"curriculum_probability_{label}"] = posterior[:, index]
    output["curriculum_expected_stage"] = expected
    output["curriculum_best_stage"] = predicted.astype(np.int8)
    output["curriculum_max_similarity"] = sorted_scores[:, -1]
    output["curriculum_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    output["curriculum_entropy"] = -np.sum(
        posterior * np.log(np.clip(posterior, 1e-12, 1.0)), axis=1
    )
    best_standard = np.argmax(similarity, axis=1)
    output["nearest_curriculum_stage"] = standards.iloc[best_standard][
        "stage"
    ].to_numpy()
    output["nearest_curriculum_domain"] = standards.iloc[best_standard][
        "domain"
    ].to_numpy()
    output["nearest_curriculum_statement"] = standards.iloc[best_standard][
        "statement"
    ].to_numpy()
    return output, scores


def evaluate_anchors(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    normalized = {
        _normalize(str(row.learning_objective)).casefold(): row
        for row in features.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for anchor in ANCHORS:
        key = _normalize(anchor.objective).casefold()
        if key not in normalized:
            raise ValueError(f"Frozen E830 anchor not found: {anchor.objective}")
        match = normalized[key]
        best = int(match.curriculum_best_stage)
        expected = float(match.curriculum_expected_stage)
        distance = max(anchor.minimum_stage - best, 0, best - anchor.maximum_stage)
        rows.append(
            {
                "learning_objective": anchor.objective,
                "minimum_stage": anchor.minimum_stage,
                "maximum_stage": anchor.maximum_stage,
                "anchor_midpoint": (
                    anchor.minimum_stage + anchor.maximum_stage
                )
                / 2.0,
                "predicted_best_stage": best,
                "predicted_expected_stage": expected,
                "distance_from_allowed_range": int(distance),
                "in_allowed_range": bool(distance == 0),
                "within_one_stage": bool(distance <= 1),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "anchors": int(len(frame)),
        "in_range_fraction": float(frame["in_allowed_range"].mean()),
        "within_one_fraction": float(frame["within_one_stage"].mean()),
        "spearman_midpoint_vs_expected_stage": float(
            spearmanr(
                frame["anchor_midpoint"],
                frame["predicted_expected_stage"],
            ).statistic
        ),
    }


def bge_redundancy_audit(
    features: pd.DataFrame,
    raw_features: pd.DataFrame,
    embedding_path: str | Path,
) -> dict[str, float]:
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape[0] != len(raw_features):
        raise ValueError("BGE objective cache row alignment changed.")
    first_rows = (
        raw_features.reset_index(names="response_row")
        .drop_duplicates("learning_objective_id", keep="first")
        .set_index("learning_objective_id")["response_row"]
    )
    objective_ids = features["learning_objective_id"].astype(str)
    if set(objective_ids) != set(first_rows.index.astype(str)):
        raise ValueError("Objective ID set differs between E830 features and BGE cache.")
    row_lookup = {str(key): int(value) for key, value in first_rows.items()}
    matrix = np.asarray(
        embeddings[[row_lookup[value] for value in objective_ids]], dtype=np.float64
    )
    target = features["curriculum_expected_stage"].to_numpy(dtype=np.float64)
    prediction = np.full(len(features), np.nan, dtype=np.float64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for training, validation in splitter.split(matrix):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0),
        )
        model.fit(matrix[training], target[training])
        prediction[validation] = model.predict(matrix[validation])
    return {
        "objectives": int(len(features)),
        "embedding_dimensions": int(matrix.shape[1]),
        "fixed_ridge_alpha": 100.0,
        "five_fold_cv_r2": float(r2_score(target, prediction)),
        "five_fold_cv_mae": float(np.mean(np.abs(target - prediction))),
        "spearman_observed_vs_predicted": float(
            spearmanr(target, prediction).statistic
        ),
    }


def synthetic_progression_benchmark(
    standards: pd.DataFrame,
    similarity: np.ndarray,
) -> dict[str, float]:
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    stages = standards["stage_index"].to_numpy(dtype=np.int8)
    scores = _stage_scores(work, stages)
    posterior, expected = _posterior_features(scores)
    rng = np.random.default_rng(SEED)
    probability = 1.0 / (1.0 + np.exp(-(-1.0 + 0.32 * stages)))
    target = rng.binomial(1, probability).astype(np.int8)
    text = standards["statement"].astype(str)
    baseline_features = np.column_stack(
        [
            text.str.len().to_numpy(dtype=np.float64),
            text.str.split().str.len().to_numpy(dtype=np.float64),
            text.str.count(r"\d").to_numpy(dtype=np.float64),
        ]
    )
    candidate_features = np.column_stack(
        [baseline_features, scores, posterior, expected]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline_prediction = np.full(len(standards), np.nan, dtype=np.float64)
    candidate_prediction = np.full(len(standards), np.nan, dtype=np.float64)
    for training, validation in splitter.split(candidate_features, target):
        baseline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2_000,
                solver="lbfgs",
                random_state=SEED,
            ),
        )
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                max_iter=2_000,
                solver="lbfgs",
                random_state=SEED,
            ),
        )
        baseline.fit(baseline_features[training], target[training])
        candidate.fit(candidate_features[training], target[training])
        baseline_prediction[validation] = baseline.predict_proba(
            baseline_features[validation]
        )[:, 1]
        candidate_prediction[validation] = candidate.predict_proba(
            candidate_features[validation]
        )[:, 1]
    baseline_loss = float(log_loss(target, baseline_prediction))
    candidate_loss = float(log_loss(target, candidate_prediction))
    return {
        "rows": int(len(standards)),
        "positive_rate": float(target.mean()),
        "baseline_log_loss": baseline_loss,
        "candidate_log_loss": candidate_loss,
        "log_loss_gain": baseline_loss - candidate_loss,
    }


def _gate(
    standards: pd.DataFrame,
    features: pd.DataFrame,
    leave_one_out: dict[str, float],
    anchor_summary: dict[str, float],
    redundancy: dict[str, float],
    synthetic: dict[str, float],
) -> dict[str, bool]:
    stage_counts = standards.groupby("stage", sort=True).size()
    best_stage_count = int(features["curriculum_best_stage"].nunique())
    return {
        "all_eight_stages_parsed": set(stage_counts.index) == set(STAGE_LABELS),
        "at_least_15_statements_per_stage": int(stage_counts.min()) >= 15,
        "curriculum_loo_spearman_at_least_0_30": (
            leave_one_out["spearman_true_vs_expected_stage"] >= 0.30
        ),
        "curriculum_loo_mae_at_most_2_00": (
            leave_one_out["mean_absolute_stage_error"] <= 2.00
        ),
        "curriculum_loo_within_one_at_least_0_45": (
            leave_one_out["within_one_accuracy"] >= 0.45
        ),
        "curriculum_loo_top1_at_least_0_18": (
            leave_one_out["top1_accuracy"] >= 0.18
        ),
        "objective_median_similarity_at_least_0_20": (
            float(features["curriculum_max_similarity"].median()) >= 0.20
        ),
        "objective_p10_similarity_at_least_0_10": (
            float(features["curriculum_max_similarity"].quantile(0.10)) >= 0.10
        ),
        "anchors_in_range_at_least_0_60": (
            anchor_summary["in_range_fraction"] >= 0.60
        ),
        "anchors_within_one_at_least_0_80": (
            anchor_summary["within_one_fraction"] >= 0.80
        ),
        "anchor_spearman_at_least_0_70": (
            anchor_summary["spearman_midpoint_vs_expected_stage"] >= 0.70
        ),
        "at_least_six_best_stages_used": best_stage_count >= 6,
        "bge_cv_r2_below_0_85": redundancy["five_fold_cv_r2"] < 0.85,
        "synthetic_log_loss_gain_at_least_0_01": (
            synthetic["log_loss_gain"] >= 0.01
        ),
    }


def run_discovery(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    paths = discover_project_paths(project_root)
    source = (
        paths.root
        / "Datasets"
        / "UK_National_Curriculum"
        / "national_curriculum_mathematics_programmes_of_study.html"
    )
    if _sha256(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Official curriculum source SHA-256 changed.")

    raw_features = pd.read_csv(
        paths.train_features,
        dtype={
            "response_id": "string",
            "learning_objective_id": "string",
            "learning_objective": "string",
        },
        usecols=[
            "response_id",
            "learning_objective_id",
            "learning_objective",
        ],
    )
    objectives = (
        raw_features[["learning_objective_id", "learning_objective"]]
        .drop_duplicates()
        .sort_values("learning_objective_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(objectives) != 398:
        raise ValueError("E830 expects exactly 398 unique objectives.")

    standards = parse_curriculum(source)
    corpus = (
        standards["statement"].astype(str).tolist()
        + objectives["learning_objective"].astype(str).tolist()
    )
    word, char = _vectorizers(corpus)
    standard_text = standards["statement"].astype(str).tolist()
    objective_text = objectives["learning_objective"].astype(str).tolist()
    standard_similarity = _combined_similarity(
        standard_text, standard_text, word, char
    )
    objective_similarity = _combined_similarity(
        objective_text, standard_text, word, char
    )

    leave_one_out = curriculum_leave_one_out(standards, standard_similarity)
    objective_features, _ = build_objective_features(
        objectives, standards, objective_similarity
    )
    anchor_rows, anchor_summary = evaluate_anchors(objective_features)
    redundancy = bge_redundancy_audit(
        objective_features,
        raw_features,
        paths.cache_dir / "bge_base_objective_256.npy",
    )
    synthetic = synthetic_progression_benchmark(standards, standard_similarity)
    clauses = _gate(
        standards,
        objective_features,
        leave_one_out,
        anchor_summary,
        redundancy,
        synthetic,
    )

    output_dir = paths.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "curriculum_progression_e830_target_free.parquet"
    standard_path = output_dir / "curriculum_progression_e830_standards.parquet"
    anchor_path = output_dir / "curriculum_progression_e830_anchors.parquet"
    report_path = output_dir / "curriculum_progression_e830_target_free_report.json"
    objective_features.to_parquet(feature_path, index=False)
    standards.to_parquet(standard_path, index=False)
    anchor_rows.to_parquet(anchor_path, index=False)

    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "seed": SEED,
        "source": {
            "url": SOURCE_URL,
            "license": "Open Government Licence v3.0",
            "path": str(source.relative_to(paths.root)),
            "sha256": _sha256(source),
            "bytes": int(source.stat().st_size),
            "retrieved_local_date": "2026-07-30",
        },
        "train_feature_source": {
            "path": str(paths.train_features.relative_to(paths.root)),
            "sha256": _sha256(paths.train_features),
            "rows": int(len(raw_features)),
            "columns_read": [
                "response_id",
                "learning_objective_id",
                "learning_objective",
            ],
            "outcome_columns_read": [],
        },
        "configuration": {
            "stage_labels": list(STAGE_LABELS),
            "temperature": TEMPERATURE,
            "word_weight": WORD_WEIGHT,
            "char_weight": CHAR_WEIGHT,
            "anchors": len(ANCHORS),
        },
        "standards": {
            "rows": int(len(standards)),
            "stage_counts": {
                key: int(value)
                for key, value in standards.groupby("stage", sort=True).size().items()
            },
            "domains": int(standards["domain"].nunique()),
        },
        "curriculum_leave_one_out": leave_one_out,
        "objectives": {
            "rows": int(len(objective_features)),
            "median_max_similarity": float(
                objective_features["curriculum_max_similarity"].median()
            ),
            "p10_max_similarity": float(
                objective_features["curriculum_max_similarity"].quantile(0.10)
            ),
            "best_stage_counts": {
                str(int(key)): int(value)
                for key, value in objective_features.groupby(
                    "curriculum_best_stage", sort=True
                ).size().items()
            },
            "mean_expected_stage": float(
                objective_features["curriculum_expected_stage"].mean()
            ),
        },
        "anchor_audit": anchor_summary,
        "bge_redundancy_audit": redundancy,
        "synthetic_progression_benchmark": synthetic,
        "gate_clauses": clauses,
        "passes_target_free_gate": bool(all(clauses.values())),
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        "artifacts": {},
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["artifacts"] = {
        "features": {
            "path": str(feature_path.relative_to(paths.root)),
            "sha256": _sha256(feature_path),
        },
        "standards": {
            "path": str(standard_path.relative_to(paths.root)),
            "sha256": _sha256(standard_path),
        },
        "anchors": {
            "path": str(anchor_path.relative_to(paths.root)),
            "sha256": _sha256(anchor_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E830 target-free curriculum-progression discovery."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = run_discovery(args.project_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
