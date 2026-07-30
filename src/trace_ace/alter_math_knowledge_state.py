from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from trace_ace.source_robust_validation import _current_rss_bytes


PROTOCOL_ID = "E910_alter_math_knowledge_state_v1"
SEED = 20260730
SOURCE_REVISION = "efaaa64e2dd08c67d9ebef3ab3141d7de28c5d40"
SOURCE_SHA256 = "50370bde7e0bb4ed691ca3a1bcf7533f7494966b63f3d6086224c66136d7d63b"
SOURCE_BYTES = 23_995_917
RAW_ROWS = 24_116
SOURCE_COLUMNS = (
    "id",
    "id2",
    "content",
    "Success",
    "confirmatory feedback",
    "negative feedback",
    "correcting",
    "giving instruction",
    "giving explanation",
    "providing further references",
    "questioning",
    "asking for elaboration",
    "praising and encouraging",
    "managing frustration",
    "managing discussions",
    "giving answers",
    "encouraging peer tutoring",
    "guiding peer tutoring",
    "acknowledging tutor issue",
    "other",
    "irrelevant statement",
    "computational skill",
    "linguistic knowledge",
    "conceptual knowledge",
    "strategic knowledge",
    "affective control",
)
LABELS = (
    "computational skill",
    "conceptual knowledge",
    "strategic knowledge",
)
EXPECTED_ROWS = 8_573
EXPECTED_SESSIONS = 2_304
EXPECTED_POSITIVES = {
    "computational skill": 252,
    "conceptual knowledge": 486,
    "strategic knowledge": 130,
}
ORDERED_INPUT_SHA256 = "02da071a600d751e10644eb0ae9048a89cc855ad899ddd8dafb0c91d417d2c0a"
CANONICAL_SHA256 = "72cc15248e0446593e2419b848a722cd6d2764f901f6af205447b12962ed06d1"
EXPECTED_FOLDS = {
    0: {
        "rows": 1_709,
        "sessions": 445,
        "positives": (35, 90, 19),
    },
    1: {
        "rows": 1_607,
        "sessions": 448,
        "positives": (34, 81, 17),
    },
    2: {
        "rows": 1_788,
        "sessions": 488,
        "positives": (51, 110, 26),
    },
    3: {
        "rows": 1_697,
        "sessions": 449,
        "positives": (55, 111, 29),
    },
    4: {
        "rows": 1_772,
        "sessions": 474,
        "positives": (77, 94, 39),
    },
}
REGULARIZATION_C = 1.0
MAX_ITERATIONS = 2_000
BOOTSTRAP_REPLICATES = 5_000
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
CONFIRMATION_ENVIRONMENT = "V_joint"
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
PROBABILITY_CLIP = 1e-6
FEATURE_AGGREGATES = ("mean", "early_mean", "late_mean", "late_minus_early")
COMPETITION_FEATURES = tuple(
    f"alter_{label.replace(' ', '_')}_{aggregate}"
    for label in LABELS
    for aggregate in FEATURE_AGGREGATES
)


def _paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "root": root,
        "raw": (
            root
            / "Datasets"
            / "ALTER_Math"
            / "LEVI_tutoring_dataset_round1.csv"
        ),
        "canonical": root / "data_cache" / "alter_math_e910_utterances.parquet",
        "audit": root / "data_cache" / "alter_math_e910_source_audit.json",
        "benchmark": root / "data_cache" / "alter_math_e910_benchmark.json",
        "oof": root / "data_cache" / "alter_math_e910_oof.parquet",
        "model": root / "models" / "alter_math_e910_knowledge_models.joblib",
        "model_metadata": (
            root / "models" / "alter_math_e910_knowledge_models.metadata.json"
        ),
        "features": (
            root / "data_cache" / "alter_math_e910_competition_features.parquet"
        ),
        "runs": root / "experiments" / "runs",
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_runtime() -> dict[str, str]:
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    expected = {
        "python": "3.12.8",
        "numpy": "2.5.1",
        "pandas": "2.3.3",
        "scikit_learn": "1.8.0",
    }
    if observed != expected:
        raise RuntimeError(
            f"E910 runtime mismatch: observed={observed}, expected={expected}"
        )
    return observed


def source_fold(session_id: str | int) -> int:
    digest = hashlib.sha256(f"E910|{session_id}".encode()).hexdigest()
    return int(digest[:16], 16) % 5


def _current_utterance(value: object) -> tuple[str, str]:
    raw = str(value)
    if "*document*" not in raw:
        raise ValueError("E910 source row lacks the *document* delimiter.")
    current = re.sub(r"\s+", " ", raw.split("*document*", maxsplit=1)[0]).strip()
    role, separator, text = current.partition(":")
    if not separator:
        raise ValueError("E910 source current utterance lacks a role delimiter.")
    return role.strip().lower(), re.sub(r"\s+", " ", text).strip()


def _normalized_utterance(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _ordered_input_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = frame.sort_values(["session_id", "turn_id"], kind="stable")
    for row in ordered[["session_id", "turn_id", "text"]].itertuples(
        index=False, name=None
    ):
        digest.update(("\t".join(map(str, row)) + "\n").encode())
    return digest.hexdigest()


def _canonicalize_source(
    raw: pd.DataFrame,
    *,
    verify_frozen_hash: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"id", "id2", "content", *LABELS}
    if set(raw.columns) != required:
        raise ValueError("E910 selected source schema changed.")
    if raw[["id", "id2"]].duplicated().any():
        raise ValueError("E910 source identity is duplicated.")
    roles: list[str] = []
    texts: list[str] = []
    for value in raw["content"]:
        role, text = _current_utterance(value)
        roles.append(role)
        texts.append(text)
    frame = raw.assign(_role=roles, text=texts)
    frame = frame[frame["_role"].str.startswith("u")].copy()
    frame["normalized_text"] = frame["text"].map(_normalized_utterance)
    cross_session_counts = frame.groupby("normalized_text", sort=False)["id"].nunique()
    repeated = set(cross_session_counts[cross_session_counts > 1].index)
    removed = frame[frame["normalized_text"].isin(repeated)]
    frame = frame[~frame["normalized_text"].isin(repeated)].copy()
    frame = frame.rename(columns={"id": "session_id", "id2": "turn_id"})
    frame["session_id"] = frame["session_id"].astype(str)
    frame["turn_id"] = frame["turn_id"].astype(str)
    frame["fold"] = frame["session_id"].map(source_fold).astype(np.int8)
    for label in LABELS:
        values = pd.to_numeric(frame[label], errors="raise").astype(np.int8)
        if not set(values.unique()).issubset({0, 1}):
            raise ValueError(f"E910 label {label!r} is not binary.")
        frame[label] = values
    frame = frame[
        [
            "session_id",
            "turn_id",
            "text",
            "normalized_text",
            "fold",
            *LABELS,
        ]
    ].reset_index(drop=True)
    if len(frame) != EXPECTED_ROWS or frame["session_id"].nunique() != EXPECTED_SESSIONS:
        raise ValueError("E910 canonical row/session count changed.")
    positives = {label: int(frame[label].sum()) for label in LABELS}
    if positives != EXPECTED_POSITIVES:
        raise ValueError(f"E910 label counts changed: {positives}")
    if not frame["normalized_text"].ne("").all():
        raise ValueError("E910 retained an empty student utterance.")
    if (frame.groupby("normalized_text")["session_id"].nunique() > 1).any():
        raise RuntimeError("E910 cross-session exact-text leakage remains.")
    ordered_input_sha256 = _ordered_input_hash(frame)
    if verify_frozen_hash and ordered_input_sha256 != ORDERED_INPUT_SHA256:
        raise ValueError("E910 ordered source input changed.")
    fold_summary: dict[int, dict[str, object]] = {}
    for fold in range(5):
        part = frame[frame["fold"].eq(fold)]
        summary = {
            "rows": len(part),
            "sessions": part["session_id"].nunique(),
            "positives": tuple(int(part[label].sum()) for label in LABELS),
        }
        if summary != EXPECTED_FOLDS[fold]:
            raise ValueError(f"E910 fold {fold} changed: {summary}")
        fold_summary[fold] = summary
    audit = {
        "protocol": PROTOCOL_ID,
        "raw_student_rows": int(len(frame) + len(removed)),
        "removed_cross_session_duplicate_rows": int(len(removed)),
        "removed_cross_session_duplicate_texts": int(len(repeated)),
        "rows": len(frame),
        "sessions": frame["session_id"].nunique(),
        "positives": positives,
        "folds": fold_summary,
        "ordered_input_sha256": ordered_input_sha256,
        "success_column_read": False,
        "model_score_computed": False,
    }
    return frame, audit


def prepare_source(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = _paths(project_root)
    raw_path = paths["raw"]
    if raw_path.stat().st_size != SOURCE_BYTES or _sha256(raw_path) != SOURCE_SHA256:
        raise ValueError("E910 raw ALTER-Math source changed.")
    header = pd.read_csv(raw_path, nrows=0)
    if tuple(header.columns) != SOURCE_COLUMNS:
        raise ValueError("E910 full source header changed.")
    selected = pd.read_csv(
        raw_path,
        usecols=["id", "id2", "content", *LABELS],
    )
    if len(selected) != RAW_ROWS:
        raise ValueError("E910 raw row count changed.")
    frame, audit = _canonicalize_source(selected)
    paths["canonical"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths["canonical"], index=False)
    canonical_sha256 = _sha256(paths["canonical"])
    if canonical_sha256 != CANONICAL_SHA256:
        raise ValueError("E910 canonical Parquet changed.")
    audit.update(
        {
            "runtime": runtime,
            "raw_sha256": SOURCE_SHA256,
            "raw_bytes": SOURCE_BYTES,
            "source_revision": SOURCE_REVISION,
            "canonical_sha256": canonical_sha256,
        }
    )
    _json_write(paths["audit"], audit)
    audit["audit_sha256"] = _sha256(paths["audit"])
    return audit


def _vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 1.0},
    )


def _classifier() -> LogisticRegression:
    return LogisticRegression(
        C=REGULARIZATION_C,
        class_weight=None,
        solver="liblinear",
        l1_ratio=0,
        max_iter=MAX_ITERATIONS,
        random_state=SEED,
    )


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = _paths(project_root)
    if not paths["canonical"].exists():
        prepare_source(project_root)
    frame = pd.read_parquet(paths["canonical"])
    order = frame["session_id"].str.cat(frame["turn_id"], sep="|").map(
        lambda value: hashlib.sha256(f"E910-benchmark|{value}".encode()).hexdigest()
    )
    sample = frame.assign(_order=order).sort_values("_order").head(512)
    synthetic = sample["session_id"].str.cat(sample["turn_id"], sep="|").map(
        lambda value: int(
            hashlib.sha256(f"E910-synthetic|{value}".encode()).hexdigest()[:8],
            16,
        )
        % 2
    )
    started = time.perf_counter()
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(sample["text"])
    probabilities: list[np.ndarray] = []
    for label_index in range(len(LABELS)):
        y = (synthetic.to_numpy() ^ (label_index % 2)).astype(np.int8)
        probabilities.append(_classifier().fit(matrix, y).predict_proba(matrix)[:, 1])
    elapsed = time.perf_counter() - started
    prediction_matrix = np.column_stack(probabilities)
    projected = elapsed * EXPECTED_ROWS / len(sample) * 5
    report = {
        "protocol": PROTOCOL_ID,
        "runtime": runtime,
        "sample_rows": len(sample),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "probability_shape": list(prediction_matrix.shape),
        "finite": bool(np.isfinite(prediction_matrix).all()),
        "elapsed_seconds": elapsed,
        "projected_external_seconds": projected,
        "peak_rss_bytes": _current_rss_bytes(),
        "synthetic_labels_only": True,
        "source_labels_read": False,
        "success_column_read": False,
        "gate": bool(
            matrix.shape[0] == 512
            and matrix.shape[1] > 1_000
            and prediction_matrix.shape == (512, len(LABELS))
            and np.isfinite(prediction_matrix).all()
            and projected < 3_600
            and _current_rss_bytes() < 8 * 1024**3
        ),
    }
    _json_write(paths["benchmark"], report)
    report["benchmark_sha256"] = _sha256(paths["benchmark"])
    if not report["gate"]:
        raise RuntimeError("E910 target-free synthetic benchmark failed.")
    return report


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    result = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            result += mask.mean() * abs(
                probabilities[mask].mean() - y_true[mask].mean()
            )
    return float(result)


def _label_metrics(
    y_true: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
) -> dict[str, float]:
    return {
        "prevalence": float(y_true.mean()),
        "candidate_log_loss": float(log_loss(y_true, candidate, labels=[0, 1])),
        "comparator_log_loss": float(log_loss(y_true, comparator, labels=[0, 1])),
        "log_loss_gain": float(
            log_loss(y_true, comparator, labels=[0, 1])
            - log_loss(y_true, candidate, labels=[0, 1])
        ),
        "candidate_brier": float(brier_score_loss(y_true, candidate)),
        "comparator_brier": float(brier_score_loss(y_true, comparator)),
        "brier_gain": float(
            brier_score_loss(y_true, comparator)
            - brier_score_loss(y_true, candidate)
        ),
        "auroc": float(roc_auc_score(y_true, candidate)),
        "average_precision": float(average_precision_score(y_true, candidate)),
        "ece_10": _ece(y_true, candidate),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
    comparator_columns: Sequence[str],
) -> dict[str, float | int | list[float]]:
    gains = np.zeros(len(frame), dtype=np.float64)
    for label, candidate_column, comparator_column in zip(
        LABELS,
        candidate_columns,
        comparator_columns,
        strict=True,
    ):
        y = frame[label].to_numpy(dtype=np.float64)
        candidate = np.clip(
            frame[candidate_column].to_numpy(dtype=np.float64),
            PROBABILITY_CLIP,
            1 - PROBABILITY_CLIP,
        )
        comparator = np.clip(
            frame[comparator_column].to_numpy(dtype=np.float64),
            PROBABILITY_CLIP,
            1 - PROBABILITY_CLIP,
        )
        gains += (
            -(y * np.log(comparator) + (1 - y) * np.log(1 - comparator))
            + (y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
        ) / len(LABELS)
    grouped = pd.DataFrame(
        {"session_id": frame["session_id"], "gain": gains}
    ).groupby("session_id", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(grouped), size=len(grouped))
        draws[replicate] = sums[indices].sum() / counts[indices].sum()
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "clusters": len(grouped),
        "observed_gain": float(gains.mean()),
        "mean_gain": float(draws.mean()),
        "interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "positive_support": float((draws > 0).mean()),
    }


def validate_external(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = _paths(project_root)
    if not paths["canonical"].exists():
        raise RuntimeError("Run the frozen E910 source audit before validation.")
    frame = pd.read_parquet(paths["canonical"])
    candidate_columns = [f"pred_{index}" for index in range(len(LABELS))]
    comparator_columns = [f"prior_{index}" for index in range(len(LABELS))]
    for column in [*candidate_columns, *comparator_columns]:
        frame[column] = np.nan
    fold_metrics: list[dict[str, object]] = []
    started = time.perf_counter()
    for fold in range(5):
        train = frame[~frame["fold"].eq(fold)]
        validation = frame[frame["fold"].eq(fold)]
        vectorizer = _vectorizer()
        train_matrix = vectorizer.fit_transform(train["text"])
        validation_matrix = vectorizer.transform(validation["text"])
        for label_index, label in enumerate(LABELS):
            classifier = _classifier()
            classifier.fit(train_matrix, train[label])
            candidate = classifier.predict_proba(validation_matrix)[:, 1]
            prior = float(train[label].mean())
            comparator = np.full(len(validation), prior, dtype=np.float64)
            frame.loc[validation.index, candidate_columns[label_index]] = candidate
            frame.loc[validation.index, comparator_columns[label_index]] = comparator
            metrics = _label_metrics(
                validation[label].to_numpy(),
                candidate,
                comparator,
            )
            fold_metrics.append({"fold": fold, "label": label, **metrics})
    if frame[[*candidate_columns, *comparator_columns]].isna().any().any():
        raise RuntimeError("E910 OOF predictions are incomplete.")
    label_metrics: dict[str, dict[str, float]] = {}
    for label_index, label in enumerate(LABELS):
        label_metrics[label] = _label_metrics(
            frame[label].to_numpy(),
            frame[candidate_columns[label_index]].to_numpy(),
            frame[comparator_columns[label_index]].to_numpy(),
        )
    macro = {
        key: float(np.mean([metrics[key] for metrics in label_metrics.values()]))
        for key in (
            "candidate_log_loss",
            "comparator_log_loss",
            "log_loss_gain",
            "candidate_brier",
            "comparator_brier",
            "brier_gain",
            "auroc",
            "average_precision",
            "ece_10",
        )
    }
    bootstrap = _cluster_bootstrap(
        frame,
        candidate_columns,
        comparator_columns,
    )
    fold_gains = [float(item["log_loss_gain"]) for item in fold_metrics]
    clauses = {
        "each_label_auroc_at_least_0_70": all(
            metrics["auroc"] >= 0.70 for metrics in label_metrics.values()
        ),
        "each_label_ap_at_least_3x_prevalence": all(
            metrics["average_precision"] >= 3 * metrics["prevalence"]
            for metrics in label_metrics.values()
        ),
        "macro_ap_gain_at_least_0_08": (
            macro["average_precision"]
            - np.mean([metrics["prevalence"] for metrics in label_metrics.values()])
            >= 0.08
        ),
        "each_label_log_loss_gain_at_least_0_005": all(
            metrics["log_loss_gain"] >= 0.005
            for metrics in label_metrics.values()
        ),
        "macro_log_loss_gain_at_least_0_015": macro["log_loss_gain"] >= 0.015,
        "each_label_brier_gain_positive": all(
            metrics["brier_gain"] > 0 for metrics in label_metrics.values()
        ),
        "macro_brier_gain_at_least_0_003": macro["brier_gain"] >= 0.003,
        "each_label_ece_at_most_0_03": all(
            metrics["ece_10"] <= 0.03 for metrics in label_metrics.values()
        ),
        "at_least_13_of_15_fold_gains_positive": (
            sum(gain > 0 for gain in fold_gains) >= 13
        ),
        "worst_fold_gain_at_least_minus_0_003": min(fold_gains) >= -0.003,
        "bootstrap_support_at_least_0_99": (
            bootstrap["positive_support"] >= 0.99
        ),
        "bootstrap_lower_bound_at_least_0_0075": (
            bootstrap["interval"][0] >= 0.0075
        ),
    }
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_alter_knowledge_state"
    )
    run_dir = paths["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    oof_path = run_dir / "e910_external_oof.parquet"
    report_path = run_dir / "e910_external_report.json"
    frame.to_parquet(oof_path, index=False)
    report = {
        "protocol": PROTOCOL_ID,
        "run_id": run_id,
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(frame),
        "sessions": frame["session_id"].nunique(),
        "labels": label_metrics,
        "macro": macro,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "clauses": clauses,
        "passed": bool(all(clauses.values())),
        "success_column_read": False,
        "competition_text_accessed": False,
        "competition_outcomes_accessed": False,
    }
    _json_write(report_path, report)
    report["oof_sha256"] = _sha256(oof_path)
    report["report_sha256"] = _sha256(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def fit_all_source(project_root: str | Path, external_report: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    paths = _paths(project_root)
    report = json.loads(Path(external_report).read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL_ID or not report.get("passed"):
        raise RuntimeError("E910 all-source fit requires its first passing external gate.")
    frame = pd.read_parquet(paths["canonical"])
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(frame["text"])
    models: dict[str, LogisticRegression] = {}
    priors: dict[str, float] = {}
    for label in LABELS:
        models[label] = _classifier().fit(matrix, frame[label])
        priors[label] = float(frame[label].mean())
    payload = {
        "protocol": PROTOCOL_ID,
        "vectorizer": vectorizer,
        "models": models,
        "priors": priors,
        "external_report_sha256": _sha256(external_report),
    }
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, paths["model"], compress=3)
    metadata = {
        "protocol": PROTOCOL_ID,
        "runtime": runtime,
        "model_sha256": _sha256(paths["model"]),
        "external_report_sha256": _sha256(external_report),
        "rows": len(frame),
        "labels": list(LABELS),
        "priors": priors,
    }
    _json_write(paths["model_metadata"], metadata)
    metadata["metadata_sha256"] = _sha256(paths["model_metadata"])
    return metadata


def _student_turns(context: object) -> list[str]:
    turns: list[str] = []
    for raw_line in str(context).splitlines():
        line = raw_line.strip()
        if line.startswith("[STUDENT]"):
            text = re.sub(r"\s+", " ", line[len("[STUDENT]") :]).strip()
            if text:
                turns.append(text)
    return turns


def aggregate_probabilities(
    probabilities: np.ndarray,
    priors: Sequence[float],
) -> dict[str, float]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError("E910 probability matrix shape changed.")
    values: dict[str, float] = {}
    if len(probabilities):
        split = max(1, (len(probabilities) + 1) // 2)
        early = probabilities[:split]
        late = probabilities[split:] if split < len(probabilities) else probabilities[-1:]
        overall_mean = probabilities.mean(axis=0)
        early_mean = early.mean(axis=0)
        late_mean = late.mean(axis=0)
    else:
        overall_mean = np.asarray(priors, dtype=np.float64)
        early_mean = np.asarray(priors, dtype=np.float64)
        late_mean = np.asarray(priors, dtype=np.float64)
    for label_index, label in enumerate(LABELS):
        prefix = f"alter_{label.replace(' ', '_')}"
        values[f"{prefix}_mean"] = float(overall_mean[label_index])
        values[f"{prefix}_early_mean"] = float(early_mean[label_index])
        values[f"{prefix}_late_mean"] = float(late_mean[label_index])
        values[f"{prefix}_late_minus_early"] = float(
            late_mean[label_index] - early_mean[label_index]
        )
    if tuple(values) != COMPETITION_FEATURES:
        raise RuntimeError("E910 competition feature order changed.")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen E910 ALTER-Math learner-state stages."
    )
    parser.add_argument(
        "stage",
        choices=("audit", "benchmark", "validate-external", "fit-all"),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--external-report")
    args = parser.parse_args(argv)
    if args.stage == "audit":
        result = prepare_source(args.project_root)
    elif args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "validate-external":
        result = validate_external(args.project_root)
    else:
        if not args.external_report:
            parser.error("--external-report is required for fit-all")
        result = fit_all_source(args.project_root, args.external_report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
