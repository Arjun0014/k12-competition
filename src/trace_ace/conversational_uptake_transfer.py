from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from trace_ace.bge_base_multiview_screen import assert_runtime
from trace_ace.io import discover_project_paths


PROTOCOL_ID = "E760_conversational_uptake_transfer_v1"
SOURCE_REPOSITORY = "https://github.com/ddemszky/conversational-uptake.git"
SOURCE_COMMIT = "67fb30cf7c3ea6f619487e33d2f99692dca107d0"
SOURCE_DATA_SHA256 = (
    "c6bb9e5ff69b6c8ae8981b9d613c41bda1c1ed5685e9e8a74f97d3c1bddb642d"
)
SOURCE_LICENSE_SHA256 = (
    "c4fcfe38abef13391b7546f3a3ec9f8c8fc7a480255f888655a57e49076c0429"
)
SOURCE_README_SHA256 = (
    "2a2895477a4308d4214a5010dfa162797bc236667f4b02a1e1f508cdc1bdc0a7"
)
EXPECTED_SOURCE_ROWS = 2_246
EXPECTED_LABELED_ROWS = 1_998
EXPECTED_GROUPS = 774
FOLDS = 5
SEED = 20260728
HASH_DIMENSION = 2**17
MAX_ROLE_TOKENS = 48
MAX_CROSS_TOKENS = 24
RIDGE_ALPHA = 10.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260728
MAX_PROJECTED_SECONDS = 3_600.0
CANONICAL_CONTENT_SHA256 = (
    "c2ef8d92db3f22247e1a16ec8c81d5acb2058e3eec6aa889d65ead1629546d82"
)
CANONICAL_PARQUET_SHA256 = (
    "368b5cee4035251838bd50be09904374024c2e81f1cb2fe504df9acadb30d542"
)
GATE_THRESHOLDS = {
    "candidate_pearson": 0.30,
    "candidate_spearman": 0.30,
    "spearman_gain_over_repetition": 0.05,
    "rmse_gain_over_repetition": 0.015,
    "quadratic_weighted_kappa": 0.15,
    "positive_candidate_spearman_folds": 5,
    "improved_spearman_folds": 4,
    "group_bootstrap_support_positive_spearman_gain": 0.90,
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ACKNOWLEDGMENT_PATTERN = re.compile(
    r"^(?:okay|ok|yes|yeah|right|good|great|exactly|correct|well done)\b"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "Datasets" / "conversational-uptake"
    return {
        "source_root": source_root,
        "source": source_root / "data" / "uptake_data.csv",
        "license": source_root / "LICENSE",
        "readme": source_root / "README.md",
        "canonical": paths.cache_dir / "conversational_uptake_e760.parquet",
        "canonical_metadata": (
            paths.cache_dir / "conversational_uptake_e760.metadata.json"
        ),
        "benchmark": (
            paths.cache_dir / "conversational_uptake_e760_benchmark.json"
        ),
        "external_report": (
            paths.experiments_dir
            / "reports"
            / "conversational_uptake_e760_external.json"
        ),
    }


def verify_sources(project_root: str | Path) -> dict[str, object]:
    paths = _paths(project_root)
    hashes = {
        "uptake_data.csv": _sha256(paths["source"]),
        "LICENSE": _sha256(paths["license"]),
        "README.md": _sha256(paths["readme"]),
    }
    expected = {
        "uptake_data.csv": SOURCE_DATA_SHA256,
        "LICENSE": SOURCE_LICENSE_SHA256,
        "README.md": SOURCE_README_SHA256,
    }
    if hashes != expected:
        raise ValueError(
            f"E760 source hash audit failed: expected={expected}, actual={hashes}"
        )
    return {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "hashes": hashes,
        "license": "MIT repository root LICENSE; notice must be retained",
        "hugging_face_checkpoint_used": False,
    }


def _tokenize(text: object, *, limit: int = MAX_ROLE_TOKENS) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())[:limit]


def repetition_features(
    student_texts: Sequence[object], teacher_texts: Sequence[object]
) -> np.ndarray:
    if len(student_texts) != len(teacher_texts):
        raise ValueError("E760 pair text arrays are misaligned.")
    rows: list[list[float]] = []
    for student_text, teacher_text in zip(student_texts, teacher_texts):
        student = _tokenize(student_text)
        teacher = _tokenize(teacher_text)
        student_set = set(student)
        teacher_set = set(teacher)
        shared = student_set & teacher_set
        union = student_set | teacher_set
        student_raw = str(student_text).lower().strip()
        teacher_raw = str(teacher_text).lower().strip()
        rows.append(
            [
                math.log1p(len(student)),
                math.log1p(len(teacher)),
                math.log1p(len(shared)),
                len(shared) / max(1, len(union)),
                len(shared) / max(1, len(student_set)),
                len(shared) / max(1, len(teacher_set)),
                float("?" in student_raw),
                float("?" in teacher_raw),
                float(bool(ACKNOWLEDGMENT_PATTERN.search(teacher_raw))),
                float(
                    bool(student_raw)
                    and len(student_raw) >= 8
                    and student_raw in teacher_raw
                ),
            ]
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(student_texts), 10) or not np.isfinite(matrix).all():
        raise RuntimeError("E760 repetition feature contract failed.")
    return matrix


def sparse_pair_records(
    student_texts: Sequence[object], teacher_texts: Sequence[object]
) -> list[dict[str, float]]:
    if len(student_texts) != len(teacher_texts):
        raise ValueError("E760 pair text arrays are misaligned.")
    records: list[dict[str, float]] = []
    for student_text, teacher_text in zip(student_texts, teacher_texts):
        student = _tokenize(student_text)
        teacher = _tokenize(teacher_text)
        student_unique = list(dict.fromkeys(student))
        teacher_unique = list(dict.fromkeys(teacher))
        features: dict[str, float] = {}
        for role, tokens in (("s", student), ("t", teacher)):
            for token in tokens:
                key = f"{role}:u:{token}"
                features[key] = features.get(key, 0.0) + 1.0
            for left, right in zip(tokens, tokens[1:]):
                key = f"{role}:b:{left}_{right}"
                features[key] = features.get(key, 0.0) + 1.0
        for token in sorted(set(student_unique) & set(teacher_unique)):
            features[f"shared:{token}"] = 1.0
        for student_token in student_unique[:MAX_CROSS_TOKENS]:
            for teacher_token in teacher_unique[:MAX_CROSS_TOKENS]:
                features[f"cross:{student_token}>{teacher_token}"] = 1.0
        records.append(features)
    return records


def hashed_pair_matrix(
    student_texts: Sequence[object], teacher_texts: Sequence[object]
) -> sparse.csr_matrix:
    hasher = FeatureHasher(
        n_features=HASH_DIMENSION,
        input_type="dict",
        alternate_sign=False,
    )
    matrix = hasher.transform(sparse_pair_records(student_texts, teacher_texts))
    matrix = matrix.tocsr().astype(np.float64)
    if matrix.shape != (len(student_texts), HASH_DIMENSION):
        raise RuntimeError("E760 hashed pair feature contract failed.")
    if matrix.nnz and float(matrix.data.min()) < 0.0:
        raise RuntimeError("E760 hashing must remain non-negative for sublinear TF-IDF.")
    return matrix


def _ordered_content_hash(frame: pd.DataFrame) -> str:
    columns = [
        "obs_id",
        "exchange_idx",
        "student_text",
        "teacher_text",
        "uptake_majority",
        "uptake_zscore",
        "fold",
    ]
    digest = hashlib.sha256()
    for values in frame[columns].itertuples(index=False, name=None):
        normalized = [
            None if pd.isna(value) else value
            for value in values
        ]
        digest.update(
            json.dumps(
                normalized, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _assign_group_folds(frame: pd.DataFrame) -> np.ndarray:
    folds = np.full(len(frame), -1, dtype=np.int8)
    labeled = frame["uptake_zscore"].notna().to_numpy()
    splitter = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    placeholder = np.zeros((int(labeled.sum()), 1), dtype=np.float32)
    groups = frame.loc[labeled, "obs_id"].astype(str).to_numpy()
    for fold, (_, validation_indices) in enumerate(
        splitter.split(placeholder, groups=groups)
    ):
        folds[np.flatnonzero(labeled)[validation_indices]] = fold
    if set(folds[labeled].tolist()) != set(range(FOLDS)):
        raise RuntimeError("E760 failed to construct five populated grouped folds.")
    return folds


def _fold_audit(frame: pd.DataFrame) -> list[dict[str, int]]:
    labeled = frame["fold"].ge(0)
    rows: list[dict[str, int]] = []
    for fold in range(FOLDS):
        validation = labeled & frame["fold"].eq(fold)
        training = labeled & ~frame["fold"].eq(fold)
        training_groups = set(frame.loc[training, "obs_id"].astype(str))
        validation_groups = set(frame.loc[validation, "obs_id"].astype(str))
        overlap = training_groups & validation_groups
        if overlap:
            raise ValueError(f"E760 fold {fold} leaks source observations.")
        rows.append(
            {
                "fold": fold,
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "training_groups": len(training_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
            }
        )
    return rows


def build_canonical(project_root: str | Path) -> dict[str, object]:
    source_audit = verify_sources(project_root)
    paths = _paths(project_root)
    source = pd.read_csv(paths["source"])
    required = {
        "obs_id",
        "exchange_idx",
        "student_text",
        "teacher_text",
        "uptake_majority",
        "uptake_zscore",
    }
    if not required.issubset(source.columns) or len(source) != EXPECTED_SOURCE_ROWS:
        raise ValueError("E760 source schema or row count changed.")
    frame = (
        source[list(required)]
        .sort_values(["obs_id", "exchange_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    frame = frame[
        [
            "obs_id",
            "exchange_idx",
            "student_text",
            "teacher_text",
            "uptake_majority",
            "uptake_zscore",
        ]
    ]
    frame["obs_id"] = frame["obs_id"].astype(str)
    frame["exchange_idx"] = frame["exchange_idx"].astype(np.int32)
    frame["student_text"] = frame["student_text"].astype(str)
    frame["teacher_text"] = frame["teacher_text"].astype(str)
    frame["uptake_majority"] = frame["uptake_majority"].astype("Float64")
    frame["uptake_zscore"] = frame["uptake_zscore"].astype("Float64")
    frame["fold"] = _assign_group_folds(frame)
    if (
        frame[["obs_id", "exchange_idx"]].duplicated().any()
        or frame["obs_id"].nunique() != EXPECTED_GROUPS
        or frame["uptake_zscore"].notna().sum() != EXPECTED_LABELED_ROWS
        or frame.loc[frame["uptake_zscore"].notna(), "student_text"]
        .map(lambda value: len(_tokenize(value, limit=10_000)))
        .min()
        < 5
    ):
        raise ValueError("E760 canonical identity, group, label, or text audit failed.")
    fold_audit = _fold_audit(frame)
    frame.to_parquet(paths["canonical"], index=False)
    metadata: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_audit,
        "rows": len(frame),
        "labeled_rows": int(frame["uptake_zscore"].notna().sum()),
        "groups": int(frame["obs_id"].nunique()),
        "fold_audit": fold_audit,
        "uptake_majority_counts": {
            str(key): int(value)
            for key, value in frame["uptake_majority"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "uptake_zscore_summary": {
            str(key): float(value)
            for key, value in frame["uptake_zscore"].describe().items()
        },
        "canonical_content_sha256": _ordered_content_hash(frame),
        "canonical_parquet_sha256": _sha256(paths["canonical"]),
        "splitter": {
            "name": "GroupKFold",
            "n_splits": FOLDS,
            "shuffle": True,
            "random_state": SEED,
            "group": "obs_id",
        },
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["canonical_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_canonical(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    if not CANONICAL_CONTENT_SHA256 or not CANONICAL_PARQUET_SHA256:
        raise RuntimeError("E760 canonical hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["canonical"]) != CANONICAL_PARQUET_SHA256:
        raise ValueError("E760 canonical Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["canonical"])
    if _ordered_content_hash(frame) != CANONICAL_CONTENT_SHA256:
        raise ValueError("E760 canonical ordered-content SHA-256 changed.")
    _fold_audit(frame)
    return json.loads(paths["canonical_metadata"].read_text(encoding="utf-8"))


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    verify_canonical(project_root)
    paths = _paths(project_root)
    frame = pd.read_parquet(
        paths["canonical"],
        columns=["obs_id", "exchange_idx", "student_text", "teacher_text"],
    )
    order = frame.apply(
        lambda row: hashlib.sha256(
            f"E760-benchmark|{row.obs_id}|{row.exchange_idx}".encode()
        ).hexdigest(),
        axis=1,
    )
    sample = frame.assign(_order=order).sort_values("_order").head(512)
    started = time.perf_counter()
    repetition = repetition_features(
        sample["student_text"].tolist(), sample["teacher_text"].tolist()
    )
    pair = hashed_pair_matrix(
        sample["student_text"].tolist(), sample["teacher_text"].tolist()
    )
    feature_seconds = time.perf_counter() - started
    synthetic_target = (
        0.6 * repetition[:, 4]
        + 0.2 * repetition[:, 8]
        - 0.1 * repetition[:, 0]
    )
    scaler = StandardScaler()
    scaled_repetition = scaler.fit_transform(repetition)
    transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
    pair_tfidf = transformer.fit_transform(pair)
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    candidate = sparse.hstack(
        [pair_tfidf, sparse.csr_matrix(scaled_repetition)], format="csr"
    )
    model.fit(candidate, synthetic_target)
    synthetic_prediction = model.predict(candidate)
    projected_seconds = feature_seconds * len(frame) / len(sample)
    clauses = {
        "runtime_is_python_3_12_8": runtime["python"] == "3.12.8",
        "runtime_is_sklearn_1_8_0": runtime["scikit_learn"] == "1.8.0",
        "features_are_finite": bool(
            np.isfinite(repetition).all()
            and np.isfinite(pair.data).all()
            and np.isfinite(synthetic_prediction).all()
        ),
        "synthetic_spearman_at_least_0_90": (
            float(spearmanr(synthetic_target, synthetic_prediction).statistic)
            >= 0.90
        ),
        "projected_feature_seconds_at_most_3600": (
            projected_seconds <= MAX_PROJECTED_SECONDS
        ),
    }
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rows": len(sample),
        "full_external_rows": len(frame),
        "hash_dimension": HASH_DIMENSION,
        "nonzero_pair_features": int(pair.nnz),
        "feature_seconds": feature_seconds,
        "projected_external_feature_seconds": projected_seconds,
        "synthetic_spearman": float(
            spearmanr(synthetic_target, synthetic_prediction).statistic
        ),
        "clauses": clauses,
        "proceed": bool(all(clauses.values())),
        "runtime": runtime,
        "competition_outcomes_accessed": False,
        "external_uptake_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["benchmark"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _continuous_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    if (
        target.shape != prediction.shape
        or target.ndim != 1
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("E760 metric inputs are invalid.")
    return {
        "pearson": float(pearsonr(target, prediction).statistic),
        "spearman": float(spearmanr(target, prediction).statistic),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
    }


def _ordinal_thresholds(
    target: np.ndarray, majority: np.ndarray
) -> tuple[float, float]:
    means = [
        float(np.mean(target[majority == level]))
        for level in (0.0, 1.0, 2.0)
    ]
    if not means[0] < means[1] < means[2]:
        raise ValueError("E760 ordinal label means are not strictly ordered.")
    return ((means[0] + means[1]) / 2.0, (means[1] + means[2]) / 2.0)


def _ordinal_prediction(
    prediction: np.ndarray, thresholds: tuple[float, float]
) -> np.ndarray:
    return np.digitize(prediction, bins=np.asarray(thresholds)).astype(np.int8)


def _group_bootstrap_support(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
) -> dict[str, float | int]:
    groups = frame["obs_id"].astype(str).to_numpy()
    target = frame["uptake_zscore"].to_numpy(dtype=np.float64)
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    gains = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in sampled])
        baseline_rho = float(
            spearmanr(target[indices], baseline[indices]).statistic
        )
        candidate_rho = float(
            spearmanr(target[indices], candidate[indices]).statistic
        )
        gains[replicate] = candidate_rho - baseline_rho
    return {
        "replicates": n_replicates,
        "seed": seed,
        "support_positive_spearman_gain": float(np.mean(gains > 0.0)),
        "gain_mean": float(np.mean(gains)),
        "gain_q025": float(np.quantile(gains, 0.025)),
        "gain_q50": float(np.quantile(gains, 0.50)),
        "gain_q975": float(np.quantile(gains, 0.975)),
    }


def external_validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    canonical_metadata = verify_canonical(project_root)
    paths = _paths(project_root)
    benchmark_result = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    if (
        benchmark_result.get("protocol_id") != PROTOCOL_ID
        or benchmark_result.get("proceed") is not True
        or benchmark_result.get("external_uptake_outcomes_accessed") is not False
        or benchmark_result.get("competition_outcomes_accessed") is not False
    ):
        raise ValueError("E760 target-free benchmark gate is not valid.")
    frame = pd.read_parquet(paths["canonical"])
    frame = frame.loc[frame["uptake_zscore"].notna()].reset_index(drop=True)
    target = frame["uptake_zscore"].to_numpy(dtype=np.float64)
    majority = frame["uptake_majority"].to_numpy(dtype=np.float64, na_value=np.nan)
    folds = frame["fold"].to_numpy(dtype=np.int8)
    repetition = repetition_features(
        frame["student_text"].tolist(), frame["teacher_text"].tolist()
    )
    hashed = hashed_pair_matrix(
        frame["student_text"].tolist(), frame["teacher_text"].tolist()
    )
    baseline_oof = np.full(len(frame), np.nan, dtype=np.float64)
    candidate_oof = np.full(len(frame), np.nan, dtype=np.float64)
    ordinal_oof = np.full(len(frame), -1, dtype=np.int8)
    fold_rows: list[dict[str, object]] = []
    for fold in range(FOLDS):
        validation = folds == fold
        training = ~validation
        train_groups = set(frame.loc[training, "obs_id"].astype(str))
        validation_groups = set(frame.loc[validation, "obs_id"].astype(str))
        if train_groups & validation_groups:
            raise RuntimeError(f"E760 fold {fold} leaks source observations.")
        scaler = StandardScaler()
        repetition_train = scaler.fit_transform(repetition[training])
        repetition_validation = scaler.transform(repetition[validation])
        baseline_model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
        baseline_model.fit(repetition_train, target[training])
        fold_baseline = baseline_model.predict(repetition_validation)
        transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
        hashed_train = transformer.fit_transform(hashed[training])
        hashed_validation = transformer.transform(hashed[validation])
        candidate_train = sparse.hstack(
            [hashed_train, sparse.csr_matrix(repetition_train)], format="csr"
        )
        candidate_validation = sparse.hstack(
            [hashed_validation, sparse.csr_matrix(repetition_validation)],
            format="csr",
        )
        candidate_model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
        candidate_model.fit(candidate_train, target[training])
        fold_candidate = candidate_model.predict(candidate_validation)
        baseline_oof[validation] = fold_baseline
        candidate_oof[validation] = fold_candidate
        ordinal_available_train = training & np.isfinite(majority)
        thresholds = _ordinal_thresholds(
            target[ordinal_available_train],
            majority[ordinal_available_train],
        )
        ordinal_oof[validation] = _ordinal_prediction(
            fold_candidate, thresholds
        )
        baseline_metrics = _continuous_metrics(
            target[validation], fold_baseline
        )
        candidate_metrics = _continuous_metrics(
            target[validation], fold_candidate
        )
        fold_rows.append(
            {
                "fold": fold,
                "training_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "training_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                **{
                    f"baseline_{name}": value
                    for name, value in baseline_metrics.items()
                },
                **{
                    f"candidate_{name}": value
                    for name, value in candidate_metrics.items()
                },
                "spearman_gain": (
                    candidate_metrics["spearman"] - baseline_metrics["spearman"]
                ),
                "rmse_gain": (
                    baseline_metrics["rmse"] - candidate_metrics["rmse"]
                ),
            }
        )
    if (
        not np.isfinite(baseline_oof).all()
        or not np.isfinite(candidate_oof).all()
        or np.any(ordinal_oof < 0)
    ):
        raise RuntimeError("E760 external OOF predictions are incomplete.")
    baseline_metrics = _continuous_metrics(target, baseline_oof)
    candidate_metrics = _continuous_metrics(target, candidate_oof)
    ordinal_mask = np.isfinite(majority)
    qwk = float(
        cohen_kappa_score(
            majority[ordinal_mask].astype(np.int8),
            ordinal_oof[ordinal_mask],
            weights="quadratic",
        )
    )
    folds_frame = pd.DataFrame(fold_rows)
    bootstrap = _group_bootstrap_support(
        frame,
        baseline_oof,
        candidate_oof,
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    summary = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "spearman_gain_over_repetition": (
            candidate_metrics["spearman"] - baseline_metrics["spearman"]
        ),
        "rmse_gain_over_repetition": (
            baseline_metrics["rmse"] - candidate_metrics["rmse"]
        ),
        "quadratic_weighted_kappa": qwk,
        "positive_candidate_spearman_folds": int(
            (folds_frame["candidate_spearman"] > 0.0).sum()
        ),
        "improved_spearman_folds": int(
            (folds_frame["spearman_gain"] > 0.0).sum()
        ),
    }
    clauses = {
        "candidate_pearson_at_least_0_30": (
            candidate_metrics["pearson"]
            >= GATE_THRESHOLDS["candidate_pearson"]
        ),
        "candidate_spearman_at_least_0_30": (
            candidate_metrics["spearman"]
            >= GATE_THRESHOLDS["candidate_spearman"]
        ),
        "spearman_gain_over_repetition_at_least_0_05": (
            summary["spearman_gain_over_repetition"]
            >= GATE_THRESHOLDS["spearman_gain_over_repetition"]
        ),
        "rmse_gain_over_repetition_at_least_0_015": (
            summary["rmse_gain_over_repetition"]
            >= GATE_THRESHOLDS["rmse_gain_over_repetition"]
        ),
        "quadratic_weighted_kappa_at_least_0_15": (
            qwk >= GATE_THRESHOLDS["quadratic_weighted_kappa"]
        ),
        "all_five_candidate_spearman_folds_positive": (
            summary["positive_candidate_spearman_folds"]
            >= GATE_THRESHOLDS["positive_candidate_spearman_folds"]
        ),
        "at_least_four_spearman_improvement_folds": (
            summary["improved_spearman_folds"]
            >= GATE_THRESHOLDS["improved_spearman_folds"]
        ),
        "group_bootstrap_support_at_least_0_90": (
            bootstrap["support_positive_spearman_gain"]
            >= GATE_THRESHOLDS[
                "group_bootstrap_support_positive_spearman_gain"
            ]
        ),
    }
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E760_directional_teacher_uptake",
        "source": verify_sources(project_root),
        "canonical": canonical_metadata,
        "lineage": {
            "target": "expert uptake_zscore only",
            "group": "obs_id",
            "split": "five-fold shuffled GroupKFold",
            "baseline": "10 fixed repetition and length features + ridge",
            "candidate": (
                "role-prefixed unigram/bigram/shared/directional-cross hashing, "
                "fold-local TF-IDF, the same repetition controls, and ridge"
            ),
            "hash_dimension": HASH_DIMENSION,
            "max_role_tokens": MAX_ROLE_TOKENS,
            "max_cross_tokens": MAX_CROSS_TOKENS,
            "ridge_alpha": RIDGE_ALPHA,
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "gate_thresholds": GATE_THRESHOLDS,
        "fold_metrics": fold_rows,
        "summary": summary,
        "group_bootstrap": bootstrap,
        "clauses": clauses,
        "passes_external_gate": bool(all(clauses.values())),
        "benchmark_sha256": _sha256(paths["benchmark"]),
        "runtime": runtime,
        "runtime_metadata": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.perf_counter() - started,
        "external_uptake_outcomes_accessed": True,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["external_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["external_report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "passes_external_gate": report["passes_external_gate"],
        "summary": summary,
        "clauses": clauses,
        "runtime_seconds": report["runtime_seconds"],
        "report": str(paths["external_report"]),
        "report_sha256": _sha256(paths["external_report"]),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E760 conversational-uptake transfer stages."
    )
    parser.add_argument(
        "stage", choices=("canonicalize", "benchmark", "external")
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "canonicalize":
        result = build_canonical(args.project_root)
    elif args.stage == "benchmark":
        result = benchmark(args.project_root)
    else:
        result = external_validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
