from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    train_features: Path
    train_labels: Path
    transcript_dir: Path
    submission_formats: tuple[Path, ...]
    cache_dir: Path
    models_dir: Path
    experiments_dir: Path


def _exactly_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {root}; found {len(matches)}."
        )
    return matches[0]


def discover_project_paths(project_root: str | Path) -> ProjectPaths:
    root = Path(project_root).resolve()
    transcript_dir = root / "Train transcripts"
    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    submission_formats = tuple(sorted(root.glob("submission_format_*.csv")))
    if len(submission_formats) != 2:
        raise FileNotFoundError(
            f"Expected the smoke and full submission formats; found {len(submission_formats)}."
        )

    return ProjectPaths(
        root=root,
        train_features=_exactly_one(root, "train_features_*.csv"),
        train_labels=_exactly_one(root, "train_labels_*.csv"),
        transcript_dir=transcript_dir,
        submission_formats=submission_formats,
        cache_dir=root / "data_cache",
        models_dir=root / "models",
        experiments_dir=root / "experiments",
    )


def load_training_tables(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_dtypes = {
        "response_id": "string",
        "session_id": "string",
        "learning_objective_id": "string",
        "learning_objective": "string",
    }
    features = pd.read_csv(paths.train_features, dtype=feature_dtypes)
    labels = pd.read_csv(paths.train_labels, dtype={"response_id": "string"})

    required_features = {
        "response_id",
        "session_id",
        "learning_objective_id",
        "learning_objective",
    }
    missing_features = required_features.difference(features.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")

    target_candidates = [column for column in ("is_correct", "correct") if column in labels]
    if len(target_candidates) != 1:
        raise ValueError("Labels must contain exactly one of 'is_correct' or 'correct'.")
    labels = labels.rename(columns={target_candidates[0]: "target"})
    labels["target"] = pd.to_numeric(labels["target"], errors="raise").astype("int8")

    if features["response_id"].duplicated().any():
        raise ValueError("Duplicate response_id values in training features.")
    if labels["response_id"].duplicated().any():
        raise ValueError("Duplicate response_id values in labels.")
    if features[list(required_features)].isna().any().any():
        raise ValueError("Training features contain null values in required columns.")
    if labels[["response_id", "target"]].isna().any().any():
        raise ValueError("Training labels contain null values.")
    if not labels["target"].isin([0, 1]).all():
        raise ValueError("Training targets must be binary.")

    feature_ids = set(features["response_id"])
    label_ids = set(labels["response_id"])
    if feature_ids != label_ids:
        raise ValueError("Feature and label response_id sets do not match exactly.")

    return features, labels


def validate_submission_formats(paths: ProjectPaths) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for path in paths.submission_formats:
        frame = pd.read_csv(path, dtype={"response_id": "string"})
        if list(frame.columns) != ["response_id", "probability"]:
            raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
        if frame["response_id"].duplicated().any():
            raise ValueError(f"Duplicate response_id values in {path.name}.")
        summaries.append({"path": path.name, "rows": int(len(frame))})
    return sorted(summaries, key=lambda item: int(item["rows"]))
