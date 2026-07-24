from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn

from trace_ace.bge_base_dual_pooling import verify_sources
from trace_ace.bge_base_multiview_screen import (
    _configure_torch,
    _current_rss_bytes,
    _session_bootstrap,
    assert_runtime,
)
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.residual_head_screen import _fold_features, _load_features
from trace_ace.semantic_hard_validation import semantic_logistic_oof


PROTOCOL_ID = "E570_objective_conditional_pairwise_ranker_v1"
PILOT_ROWS = 4_096
PILOT_PROTOCOL = "semantic_k50_s0"
BCE_WEIGHT = 0.5
PAIR_WEIGHT = 0.5
L2_WEIGHT = 0.0015
MAX_ITERATIONS = 100
HISTORY_SIZE = 20
TOLERANCE_GRADIENT = 1e-7
TOLERANCE_CHANGE = 1e-9
REGULARIZATION_C = 0.1
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
MAX_PROJECTED_HOURS = 2.0
MAX_RSS_BYTES = 8 * 1024**3


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_path(project_root: str | Path) -> Path:
    return (
        discover_project_paths(project_root).cache_dir
        / "objective_pairwise_e570_benchmark.json"
    )


def build_objective_pairs(
    frame: pd.DataFrame,
    training_mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if (
        training_mask.shape != (len(frame),)
        or labels.shape != (len(frame),)
        or set(np.unique(labels[training_mask]).tolist()) != {0, 1}
    ):
        raise ValueError("E570 pair inputs violate the frozen contract.")
    training_indices = np.flatnonzero(training_mask)
    ordered = (
        frame.iloc[training_indices]
        .assign(_row=training_indices)
        .sort_values(["learning_objective_id", "response_id"], kind="mergesort")
    )
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    mixed_objectives = 0
    for _, group in ordered.groupby("learning_objective_id", sort=True):
        rows = group["_row"].to_numpy(dtype=np.int64)
        positives = rows[labels[rows] == 1]
        negatives = rows[labels[rows] == 0]
        if not len(positives) or not len(negatives):
            continue
        mixed_objectives += 1
        pair_count = max(len(positives), len(negatives))
        positive_rows.append(np.resize(positives, pair_count))
        negative_rows.append(np.resize(negatives, pair_count))
    if not positive_rows:
        raise ValueError("E570 found no mixed-label training objective.")
    positives = np.concatenate(positive_rows)
    negatives = np.concatenate(negative_rows)
    if (
        positives.shape != negatives.shape
        or np.any(labels[positives] != 1)
        or np.any(labels[negatives] != 0)
        or np.any(
            frame.iloc[positives]["learning_objective_id"].to_numpy()
            != frame.iloc[negatives]["learning_objective_id"].to_numpy()
        )
    ):
        raise RuntimeError("E570 objective pair audit failed.")
    audit = {
        "training_rows": int(training_mask.sum()),
        "training_objectives": int(
            frame.loc[training_mask, "learning_objective_id"].nunique()
        ),
        "mixed_label_objectives": mixed_objectives,
        "pairs": len(positives),
        "unique_positive_rows": len(np.unique(positives)),
        "unique_negative_rows": len(np.unique(negatives)),
    }
    return positives, negatives, audit


class LinearPairwiseRanker(nn.Module):
    def __init__(self, input_width: int, prior: float) -> None:
        super().__init__()
        if input_width <= 0 or not 0.0 < prior < 1.0:
            raise ValueError("E570 model initialization is invalid.")
        self.weight = nn.Parameter(torch.zeros(input_width, dtype=torch.float32))
        self.bias = nn.Parameter(
            torch.tensor(np.log(prior / (1.0 - prior)), dtype=torch.float32)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight + self.bias


def fit_pairwise_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    pair_positive: np.ndarray,
    pair_negative: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if (
        x_train.ndim != 2
        or x_validation.ndim != 2
        or x_train.shape[1] != x_validation.shape[1]
        or y_train.shape != (len(x_train),)
        or set(np.unique(y_train).tolist()) != {0, 1}
        or pair_positive.shape != pair_negative.shape
        or pair_positive.ndim != 1
        or not len(pair_positive)
        or pair_positive.min() < 0
        or pair_negative.min() < 0
        or pair_positive.max() >= len(x_train)
        or pair_negative.max() >= len(x_train)
    ):
        raise ValueError("E570 fit arrays violate the frozen contract.")
    torch.use_deterministic_algorithms(True)
    features = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    target = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    positive = torch.from_numpy(pair_positive.astype(np.int64))
    negative = torch.from_numpy(pair_negative.astype(np.int64))
    model = LinearPairwiseRanker(x_train.shape[1], float(np.mean(y_train)))
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=MAX_ITERATIONS,
        history_size=HISTORY_SIZE,
        tolerance_grad=TOLERANCE_GRADIENT,
        tolerance_change=TOLERANCE_CHANGE,
        line_search_fn="strong_wolfe",
    )
    loss_trace: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        bce = functional.binary_cross_entropy_with_logits(logits, target)
        pair_loss = functional.softplus(
            -(logits[positive] - logits[negative])
        ).mean()
        penalty = L2_WEIGHT * torch.sum(model.weight.square())
        loss = BCE_WEIGHT * bce + PAIR_WEIGHT * pair_loss + penalty
        if not torch.isfinite(loss):
            raise RuntimeError("E570 encountered non-finite objective.")
        loss.backward()
        loss_trace.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    model.eval()
    with torch.inference_mode():
        train_logits = model(features)
        prediction = torch.sigmoid(
            model(torch.from_numpy(np.asarray(x_validation, dtype=np.float32)))
        ).numpy()
        final_bce = float(
            functional.binary_cross_entropy_with_logits(train_logits, target)
        )
        final_pair = float(
            functional.softplus(
                -(train_logits[positive] - train_logits[negative])
            ).mean()
        )
    if (
        prediction.shape != (len(x_validation),)
        or not np.isfinite(prediction).all()
        or np.any((prediction <= 0) | (prediction >= 1))
        or not loss_trace
    ):
        raise ValueError("E570 prediction or optimization audit failed.")
    state = optimizer.state.get(model.weight, {})
    summary = {
        "input_width": x_train.shape[1],
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "pairs": len(pair_positive),
        "training_prior": float(np.mean(y_train)),
        "closure_evaluations": len(loss_trace),
        "optimizer_iterations": int(state.get("n_iter", -1)),
        "first_objective": loss_trace[0],
        "final_objective": loss_trace[-1],
        "final_bce": final_bce,
        "final_pairwise_loss": final_pair,
        "weight_norm": float(torch.linalg.vector_norm(model.weight.detach())),
    }
    return prediction.astype(np.float64), summary


def _local_pair_positions(
    training_mask: np.ndarray,
    positive_global: np.ndarray,
    negative_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    training_rows = np.flatnonzero(training_mask)
    lookup = np.full(len(training_mask), -1, dtype=np.int64)
    lookup[training_rows] = np.arange(len(training_rows), dtype=np.int64)
    positive = lookup[positive_global]
    negative = lookup[negative_global]
    if np.any(positive < 0) or np.any(negative < 0):
        raise RuntimeError("E570 pair localization failed.")
    return positive, negative


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    _configure_torch()
    source_hashes = verify_sources(project_root)
    frame, indices, folds, semantic, dense = _load_features(project_root)
    train_mask, validation_mask = _fold_masks(frame, folds, 0)
    synthetic = (indices % 2).astype(np.int8)
    positive, negative, pair_audit = build_objective_pairs(
        frame, train_mask, synthetic
    )
    local_positive, local_negative = _local_pair_positions(
        train_mask, positive, negative
    )
    x_train, x_validation = _fold_features(
        semantic, dense, train_mask, validation_mask
    )
    started = time.perf_counter()
    _, fit_summary = fit_pairwise_ranker(
        x_train,
        synthetic[train_mask],
        x_validation,
        local_positive,
        local_negative,
    )
    elapsed = time.perf_counter() - started
    projected_seconds = elapsed * 5
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "one fold with deterministic index-parity labels",
        "elapsed_seconds": elapsed,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "pair_audit": pair_audit,
        "fit_summary": fit_summary,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "outcome_labels_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    _benchmark_path(project_root).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_benchmark(project_root: str | Path) -> dict[str, object]:
    path = _benchmark_path(project_root)
    if not path.exists():
        raise FileNotFoundError("Run the E570 benchmark before validation.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol_id") != PROTOCOL_ID
        or result.get("source_hashes") != verify_sources(project_root)
        or result.get("proceed") is not True
        or result.get("outcome_labels_accessed") is not False
    ):
        raise ValueError("E570 benchmark binding or gate is invalid.")
    return result


def _fold_metrics(
    frame: pd.DataFrame,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> pd.DataFrame:
    target = frame["target"].to_numpy(dtype=np.int8)
    rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = folds == fold
        base_metrics = binary_metrics(target[mask], baseline[mask])
        candidate_metrics = binary_metrics(target[mask], candidate[mask])
        rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                **{f"bge_{key}": value for key, value in base_metrics.items()},
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                **{
                    f"delta_{key}_vs_bge": candidate_metrics[key]
                    - base_metrics[key]
                    for key in (
                        "log_loss",
                        "roc_auc",
                        "brier_score",
                        "ece_10",
                        "prediction_mean",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    _configure_torch()
    source_hashes = verify_sources(project_root)
    benchmark_result = _load_benchmark(project_root)
    frame, _, folds, semantic, dense = _load_features(project_root)
    target = frame["target"].to_numpy(dtype=np.int8)
    baseline_prediction = semantic_logistic_oof(
        frame,
        folds,
        semantic,
        dense,
        regularization_c=REGULARIZATION_C,
        n_splits=5,
    )
    ranker_prediction = np.full(PILOT_ROWS, np.nan, dtype=np.float64)
    fit_summaries: list[dict[str, object]] = []
    pair_audits: list[dict[str, object]] = []
    peak_rss = _current_rss_bytes()
    for fold in range(5):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        positive, negative, pair_audit = build_objective_pairs(
            frame, train_mask, target
        )
        local_positive, local_negative = _local_pair_positions(
            train_mask, positive, negative
        )
        x_train, x_validation = _fold_features(
            semantic, dense, train_mask, validation_mask
        )
        prediction, fit_summary = fit_pairwise_ranker(
            x_train,
            target[train_mask],
            x_validation,
            local_positive,
            local_negative,
        )
        ranker_prediction[validation_mask] = prediction
        pair_audits.append({"fold": fold, **pair_audit})
        fit_summaries.append({"fold": fold, **fit_summary})
        peak_rss = max(peak_rss, _current_rss_bytes())
    if not np.isfinite(ranker_prediction).all():
        raise RuntimeError("E570 OOF predictions are incomplete.")

    baseline_metrics = binary_metrics(target, baseline_prediction)
    metric_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for weight in BLEND_WEIGHTS:
        prediction = (
            (1.0 - weight) * baseline_prediction + weight * ranker_prediction
        )
        metrics = binary_metrics(target, prediction)
        metric_rows.append(
            {
                "blend_weight": weight,
                **metrics,
                **{
                    f"delta_{key}_vs_bge": metrics[key] - baseline_metrics[key]
                    for key in (
                        "log_loss",
                        "roc_auc",
                        "brier_score",
                        "ece_10",
                        "prediction_mean",
                    )
                },
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "response_id": frame["response_id"],
                    "session_id": frame["session_id"],
                    "learning_objective_id": frame["learning_objective_id"],
                    "fold": folds,
                    "target": target,
                    "blend_weight": weight,
                    "pred_bge_base": baseline_prediction,
                    "pred_pairwise_ranker": ranker_prediction,
                    "prediction": prediction,
                }
            )
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["log_loss", "roc_auc", "blend_weight"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = metrics.iloc[0].to_dict()
    all_predictions = pd.concat(predictions, ignore_index=True)
    selected_predictions = all_predictions.loc[
        all_predictions["blend_weight"].eq(selected["blend_weight"])
    ].reset_index(drop=True)
    bootstrap = _session_bootstrap(
        frame,
        selected_predictions["pred_bge_base"].to_numpy(),
        selected_predictions["prediction"].to_numpy(),
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    fold_metrics = _fold_metrics(
        frame,
        folds,
        selected_predictions["pred_bge_base"].to_numpy(),
        selected_predictions["prediction"].to_numpy(),
    )
    loss_path = bool(
        selected["delta_log_loss_vs_bge"] <= -0.0015
        and selected["delta_roc_auc_vs_bge"] >= 0
        and selected["delta_brier_score_vs_bge"] <= 0
        and selected["delta_ece_10_vs_bge"] <= 0
    )
    auc_path = bool(
        selected["delta_roc_auc_vs_bge"] >= 0.005
        and selected["delta_log_loss_vs_bge"] <= 0
        and selected["delta_brier_score_vs_bge"] <= 0
        and selected["delta_ece_10_vs_bge"] <= 0
    )
    passes = bool(
        (loss_path or auc_path)
        and bootstrap["support_positive_log_loss_gain"] >= 0.90
    )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_objective_pairwise_ranker")
    paths = discover_project_paths(project_root)
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "oof_predictions.parquet"
    fold_path = run_dir / "fold_metrics.csv"
    bootstrap_path = run_dir / "bootstrap.csv"
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    all_predictions.to_parquet(predictions_path, index=False)
    fold_metrics.to_csv(fold_path, index=False, lineterminator="\n")
    pd.DataFrame([bootstrap]).to_csv(
        bootstrap_path, index=False, lineterminator="\n"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E570_objective_conditional_pairwise_ranker",
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "objective": {
            "bce_weight": BCE_WEIGHT,
            "pairwise_weight": PAIR_WEIGHT,
            "l2_weight": L2_WEIGHT,
        },
        "optimizer": {
            "name": "PyTorch LBFGS",
            "max_iterations": MAX_ITERATIONS,
            "history_size": HISTORY_SIZE,
            "line_search": "strong_wolfe",
            "tolerance_gradient": TOLERANCE_GRADIENT,
            "tolerance_change": TOLERANCE_CHANGE,
        },
        "blend_weights": list(BLEND_WEIGHTS),
        "baseline_metrics": baseline_metrics,
        "selected": selected,
        "fold_log_loss_changes": fold_metrics["delta_log_loss_vs_bge"].tolist(),
        "paired_session_bootstrap": bootstrap,
        "screen_clauses": {
            "loss_path": loss_path,
            "auc_path": auc_path,
            "session_bootstrap_support_at_least_0_90": (
                bootstrap["support_positive_log_loss_gain"] >= 0.90
            ),
        },
        "passes_screen": passes,
        "pair_audits": pair_audits,
        "fit_summaries": fit_summaries,
        "benchmark": benchmark_result,
        "peak_rss_bytes": peak_rss,
        "source_hashes": source_hashes,
        "runtime": runtime,
        "validation_runtime_seconds": time.perf_counter() - started,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        "metrics": _sha256(metrics_path),
        "predictions": _sha256(predictions_path),
        "fold_metrics": _sha256(fold_path),
        "bootstrap": _sha256(bootstrap_path),
    }
    report["artifact_sha256"] = artifact_hashes
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report": report,
        "report_sha256": _sha256(report_path),
        "passes_screen": passes,
        "artifact_sha256": artifact_hashes,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E570 objective-pairwise ranker."
    )
    parser.add_argument("stage", choices=("benchmark", "validate", "pipeline"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "validate":
        result = validate(args.project_root)
    else:
        benchmark_result = benchmark(args.project_root)
        if not benchmark_result["proceed"]:
            raise RuntimeError("E570 benchmark failed; validation is prohibited.")
        result = {
            "benchmark": benchmark_result,
            "validation": validate(args.project_root),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
