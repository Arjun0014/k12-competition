from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

from trace_ace.bge_base_dual_pooling import _load_pilot
from trace_ace.bge_base_multiview_screen import (
    _current_rss_bytes,
    _session_bootstrap,
    assert_runtime,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


PROTOCOL_ID = "E710_target_free_next_turn_coherence_v1"
PILOT_ROWS = 4_096
SSL_FOLDS = 5
HASH_FEATURES = 2**18
TOKEN_CAP = 16
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
SEED = 20260728
MAX_RSS_BYTES = 8 * 1024**3
TOKEN_RE = re.compile(r"[a-z0-9]+")
ACKNOWLEDGEMENTS = frozenset({"yes", "yeah", "yep", "okay", "ok", "mm", "mhm"})
UNCERTAINTY = frozenset({"maybe", "think", "guess", "unsure", "unclear", "dont", "know"})
AGGREGATE_NAMES = (
    "pair_count",
    "probability_mean",
    "probability_std",
    "probability_min",
    "probability_max",
    "probability_q25",
    "probability_q75",
    "probability_first_half_mean",
    "probability_second_half_mean",
    "probability_late_minus_early",
    "probability_slope",
    "probability_share_at_least_0_5",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_fold(session_id: object) -> int:
    value = hashlib.sha256(f"E710|ssl|{session_id}".encode()).hexdigest()
    return int(value[:16], 16) % SSL_FOLDS


def _tokens(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        if token not in seen:
            values.append(token)
            seen.add(token)
        if len(values) == TOKEN_CAP:
            break
    return values


def pair_features(tutor_text: str, student_text: str) -> dict[str, float]:
    tutor = _tokens(tutor_text)
    student = _tokens(student_text)
    tutor_set = set(tutor)
    student_set = set(student)
    shared = tutor_set & student_set
    union = tutor_set | student_set
    features: dict[str, float] = {}
    for token in tutor:
        features[f"t={token}"] = 1.0
    for token in student:
        features[f"s={token}"] = 1.0
    for token in sorted(shared):
        features[f"shared={token}"] = 1.0
    for left in tutor:
        for right in student:
            features[f"x={left}|{right}"] = 1.0
    features["n:tutor_log_words"] = float(np.log1p(len(tutor)))
    features["n:student_log_words"] = float(np.log1p(len(student)))
    features["n:jaccard"] = float(len(shared) / max(len(union), 1))
    features["n:student_tutor_ratio"] = float(
        min(len(student) / max(len(tutor), 1), 4.0) / 4.0
    )
    features["n:tutor_question"] = float("?" in tutor_text)
    features["n:student_question"] = float("?" in student_text)
    features["n:student_ack"] = float(bool(student_set & ACKNOWLEDGEMENTS))
    features["n:student_uncertain"] = float(bool(student_set & UNCERTAINTY))
    return features


def extract_pairs(
    frame: pd.DataFrame, contexts: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, float]], list[dict[str, float]]]:
    if list(frame["response_id"].astype(str)) != list(
        contexts["response_id"].astype(str)
    ):
        raise ValueError("E710 context rows do not align with the pilot.")
    rows: list[dict[str, object]] = []
    positive_features: list[dict[str, float]] = []
    negative_features: list[dict[str, float]] = []
    for row_index, row in contexts.iterrows():
        lines = str(row["objective_context"]).splitlines()
        pairs = [
            (lines[i][8:].strip(), lines[i + 1][10:].strip())
            for i in range(len(lines) - 1)
            if lines[i].startswith("[TUTOR]")
            and lines[i + 1].startswith("[STUDENT]")
        ]
        if len(pairs) < 2:
            continue
        for sequence, (tutor, student) in enumerate(pairs):
            negative_student = pairs[(sequence + 1) % len(pairs)][1]
            rows.append(
                {
                    "row_index": row_index,
                    "response_id": str(row["response_id"]),
                    "session_id": str(row["session_id"]),
                    "sequence": sequence,
                    "ssl_fold": _ssl_fold(row["session_id"]),
                }
            )
            positive_features.append(pair_features(tutor, student))
            negative_features.append(pair_features(tutor, negative_student))
    pairs = pd.DataFrame(rows)
    if (
        pairs.empty
        or len(pairs) != len(positive_features)
        or len(pairs) != len(negative_features)
        or pairs.groupby("session_id")["ssl_fold"].nunique().max() != 1
    ):
        raise ValueError("E710 pair construction failed its contract.")
    return pairs, positive_features, negative_features


def _aggregate_scores(
    pairs: pd.DataFrame, scores: np.ndarray, n_rows: int
) -> np.ndarray:
    result = np.zeros((n_rows, len(AGGREGATE_NAMES)), dtype=np.float32)
    scored = pairs.assign(score=scores)
    for row_index, group in scored.groupby("row_index", sort=False):
        values = group.sort_values("sequence", kind="mergesort")["score"].to_numpy(
            dtype=np.float64
        )
        midpoint = max(1, len(values) // 2)
        first = float(values[:midpoint].mean())
        second = float(values[midpoint:].mean()) if midpoint < len(values) else first
        x = np.arange(len(values), dtype=np.float64)
        slope = (
            float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0
        )
        result[int(row_index)] = np.asarray(
            [
                len(values),
                values.mean(),
                values.std(ddof=0),
                values.min(),
                values.max(),
                np.quantile(values, 0.25),
                np.quantile(values, 0.75),
                first,
                second,
                second - first,
                slope,
                np.mean(values >= 0.5),
            ],
            dtype=np.float32,
        )
    if not np.isfinite(result).all():
        raise ValueError("E710 aggregates contain non-finite values.")
    return result


def build_target_free_features(
    project_root: str | Path, *, benchmark_rows: int | None = None
) -> dict[str, object]:
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    frame, _, indices, _ = _load_pilot(project_root)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "session_id", "objective_context"],
    ).iloc[indices].reset_index(drop=True)
    if benchmark_rows is not None:
        frame = frame.iloc[:benchmark_rows].reset_index(drop=True)
        contexts = contexts.iloc[:benchmark_rows].reset_index(drop=True)
    pairs, positive_features, negative_features = extract_pairs(frame, contexts)
    hasher = FeatureHasher(
        n_features=HASH_FEATURES, input_type="dict", alternate_sign=True
    )
    positive = hasher.transform(positive_features).tocsr()
    negative = hasher.transform(negative_features).tocsr()
    pair_scores = np.full(len(pairs), np.nan, dtype=np.float64)
    started = time.perf_counter()
    fold_summaries: list[dict[str, object]] = []
    for fold in range(SSL_FOLDS):
        train_mask = pairs["ssl_fold"].to_numpy() != fold
        validation_mask = ~train_mask
        x_train = sparse.vstack(
            [positive[train_mask], negative[train_mask]], format="csr"
        )
        y_train = np.concatenate(
            [
                np.ones(int(train_mask.sum()), dtype=np.int8),
                np.zeros(int(train_mask.sum()), dtype=np.int8),
            ]
        )
        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            max_iter=20,
            tol=1e-4,
            random_state=SEED,
            class_weight=None,
        )
        model.fit(x_train, y_train)
        pair_scores[validation_mask] = model.predict_proba(
            positive[validation_mask]
        )[:, 1]
        fold_summaries.append(
            {
                "ssl_fold": fold,
                "training_examples": int(len(y_train)),
                "validation_positive_pairs": int(validation_mask.sum()),
                "iterations": int(model.n_iter_),
            }
        )
        if _current_rss_bytes() >= MAX_RSS_BYTES:
            raise MemoryError("E710 exceeded its 8 GiB RSS gate.")
    elapsed = time.perf_counter() - started
    if not np.isfinite(pair_scores).all():
        raise RuntimeError("E710 self-supervised cross-fit is incomplete.")
    aggregates = _aggregate_scores(pairs, pair_scores, len(frame))
    result = {
        "protocol_id": PROTOCOL_ID,
        "runtime": runtime,
        "rows": len(frame),
        "positive_pairs": len(pairs),
        "balanced_self_supervised_examples": 2 * len(pairs),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": _current_rss_bytes(),
        "fold_summaries": fold_summaries,
        "aggregate_names": list(AGGREGATE_NAMES),
        "pair_score_mean": float(pair_scores.mean()),
        "pair_score_std": float(pair_scores.std()),
    }
    if benchmark_rows is None:
        cache_path = paths.cache_dir / "target_free_coherence_e710_pilot.npy"
        metadata_path = paths.cache_dir / "target_free_coherence_e710.metadata.json"
        np.save(cache_path, aggregates, allow_pickle=False)
        result["cache_sha256"] = _sha256(cache_path)
        result["cache_path"] = str(cache_path)
        metadata_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["metadata_path"] = str(metadata_path)
        result["metadata_sha256"] = _sha256(metadata_path)
    return result


def benchmark(project_root: str | Path) -> dict[str, object]:
    result = build_target_free_features(project_root, benchmark_rows=512)
    projected = result["elapsed_seconds"] * PILOT_ROWS / result["rows"]
    result["projected_seconds"] = projected
    result["proceed"] = bool(projected <= 2 * 3600 and result["peak_rss_bytes"] < MAX_RSS_BYTES)
    paths = discover_project_paths(project_root)
    path = paths.cache_dir / "target_free_coherence_e710_benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["benchmark_path"] = str(path)
    result["benchmark_sha256"] = _sha256(path)
    return result


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    paths = discover_project_paths(project_root)
    frame, _, indices, folds = _load_pilot(project_root)
    metadata_path = paths.cache_dir / "target_free_coherence_e710.metadata.json"
    cache_path = paths.cache_dir / "target_free_coherence_e710_pilot.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["cache_sha256"] != _sha256(cache_path):
        raise ValueError("E710 cache hash changed.")
    coherence = np.load(cache_path)
    if coherence.shape != (PILOT_ROWS, len(AGGREGATE_NAMES)):
        raise ValueError("E710 cache shape changed.")
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    base_features, base_similarity = prepare_bge_base_features(paths.cache_dir, full)
    _, dense_full, dense_names = prepare_semantic_features(paths.cache_dir, full)
    dense = dense_full[indices].copy()
    dense[:, -1:] = base_similarity[indices]
    baseline = semantic_logistic_oof(
        frame,
        folds,
        base_features[indices],
        dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    candidate_dense = np.column_stack([coherence, dense])
    candidate = semantic_logistic_oof(
        frame,
        folds,
        np.empty((PILOT_ROWS, 0), dtype=np.float32),
        candidate_dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline_metrics = binary_metrics(target, baseline)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for weight in BLEND_WEIGHTS:
        prediction = (1 - weight) * baseline + weight * candidate
        metrics = binary_metrics(target, prediction)
        rows.append(
            {
                "blend_weight": weight,
                **metrics,
                **{
                    f"delta_{name}_vs_bge": metrics[name] - baseline_metrics[name]
                    for name in ("log_loss", "roc_auc", "brier_score", "ece_10")
                },
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "response_id": frame["response_id"],
                    "session_id": frame["session_id"],
                    "fold": folds,
                    "target": target,
                    "blend_weight": weight,
                    "pred_bge_base": baseline,
                    "pred_coherence": candidate,
                    "prediction": prediction,
                }
            )
        )
    metrics_frame = pd.DataFrame(rows).sort_values(
        ["log_loss", "roc_auc", "blend_weight"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = metrics_frame.iloc[0].to_dict()
    all_predictions = pd.concat(predictions, ignore_index=True)
    selected_predictions = all_predictions.loc[
        all_predictions["blend_weight"].eq(selected["blend_weight"])
    ].reset_index(drop=True)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = selected_predictions["fold"].to_numpy() == fold
        base_fold = binary_metrics(target[mask], baseline[mask])
        candidate_fold = binary_metrics(
            target[mask], selected_predictions.loc[mask, "prediction"].to_numpy()
        )
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "baseline_log_loss": base_fold["log_loss"],
                "candidate_log_loss": candidate_fold["log_loss"],
                "delta_log_loss_vs_bge": (
                    candidate_fold["log_loss"] - base_fold["log_loss"]
                ),
            }
        )
    bootstrap = _session_bootstrap(
        frame,
        baseline,
        selected_predictions["prediction"].to_numpy(),
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    fold_regression = max(row["delta_log_loss_vs_bge"] for row in fold_rows)
    clauses = {
        "log_loss_gain_at_least_0_0016": (
            selected["delta_log_loss_vs_bge"] <= -0.0016
        ),
        "max_fold_regression_at_most_0_0005": fold_regression <= 0.0005,
        "auroc_non_regression": selected["delta_roc_auc_vs_bge"] >= 0,
        "brier_non_regression": selected["delta_brier_score_vs_bge"] <= 0,
        "ece_regression_at_most_0_001": selected["delta_ece_10_vs_bge"] <= 0.001,
        "bootstrap_support_at_least_0_95": (
            bootstrap["support_positive_log_loss_gain"] >= 0.95
        ),
    }
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_target_free_coherence")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    selected_predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame([bootstrap]).to_csv(
        run_dir / "bootstrap.csv", index=False, lineterminator="\n"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "runtime": runtime,
        "python_platform": platform.platform(),
        "pilot_rows": PILOT_ROWS,
        "aggregate_names": list(AGGREGATE_NAMES),
        "dense_features": dense_names,
        "baseline_metrics": baseline_metrics,
        "selected": selected,
        "fold_metrics": fold_rows,
        "paired_session_bootstrap": bootstrap,
        "clauses": clauses,
        "passes_screen": bool(all(clauses.values())),
        "validation_elapsed_seconds": time.perf_counter() - started,
        "cache_sha256": metadata["cache_sha256"],
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["artifact_hashes"] = {
        name: _sha256(run_dir / name)
        for name in (
            "metrics.csv",
            "fold_metrics.csv",
            "oof_predictions.parquet",
            "bootstrap.csv",
            "report.json",
        )
    }
    return {"run_id": run_id, "run_dir": str(run_dir), "report": report}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run frozen E710 coherence screen.")
    parser.add_argument("stage", choices=("benchmark", "build-cache", "validate"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build-cache":
        result = build_target_free_features(args.project_root)
    else:
        result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
