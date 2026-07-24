from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from trace_ace.bge_base_dual_pooling import _load_pilot, verify_sources
from trace_ace.bge_base_multiview_screen import (
    _configure_torch,
    _current_rss_bytes,
    _session_bootstrap,
    assert_runtime,
)
from trace_ace.encoder_upgrade_validation import prepare_bge_base_features
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


PROTOCOL_ID = "E560_residual_nonlinear_head_v1"
PILOT_ROWS = 4_096
PILOT_PROTOCOL = "semantic_k50_s0"
HIDDEN_WIDTH = 32
DROPOUT = 0.15
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
GRADIENT_NORM_CAP = 1.0
DENSE_SCALE = 0.08
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
        / "residual_head_e560_benchmark.json"
    )


class ResidualOutcomeHead(nn.Module):
    def __init__(self, input_width: int, prior: float, seed: int) -> None:
        super().__init__()
        if input_width <= 0 or not 0.0 < prior < 1.0:
            raise ValueError("E560 head initialization inputs are invalid.")
        torch.manual_seed(seed)
        self.direct = nn.Linear(input_width, 1, bias=False)
        self.hidden = nn.Linear(input_width, HIDDEN_WIDTH)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)
        self.residual_out = nn.Linear(HIDDEN_WIDTH, 1, bias=False)
        self.bias = nn.Parameter(
            torch.tensor(math.log(prior / (1.0 - prior)), dtype=torch.float32)
        )
        nn.init.zeros_(self.direct.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.residual_out.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.residual_out(
            self.dropout(self.activation(self.hidden(features)))
        )
        return (self.direct(features) + residual).squeeze(1) + self.bias


def _fold_features(
    semantic: np.ndarray,
    dense: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    dense_train = scaler.fit_transform(dense[train_mask]).astype(np.float32)
    dense_validation = scaler.transform(dense[validation_mask]).astype(np.float32)
    x_train = np.column_stack(
        [semantic[train_mask], dense_train * np.float32(DENSE_SCALE)]
    ).astype(np.float32)
    x_validation = np.column_stack(
        [semantic[validation_mask], dense_validation * np.float32(DENSE_SCALE)]
    ).astype(np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise ValueError("E560 fold features are non-finite.")
    return x_train, x_validation


def train_residual_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[np.ndarray, dict[str, object]]:
    if (
        x_train.ndim != 2
        or x_validation.ndim != 2
        or x_train.shape[1] != x_validation.shape[1]
        or y_train.shape != (len(x_train),)
        or set(np.unique(y_train).tolist()) != {0, 1}
        or epochs <= 0
    ):
        raise ValueError("E560 training arrays violate the frozen contract.")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ResidualOutcomeHead(
        x_train.shape[1], float(np.mean(y_train)), seed=seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    epoch_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor), generator=generator)
        cumulative_loss = 0.0
        rows_seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[indices])
            loss = loss_function(logits, y_tensor[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("E560 encountered non-finite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM_CAP)
            optimizer.step()
            cumulative_loss += float(loss.detach()) * len(indices)
            rows_seen += len(indices)
        epoch_losses.append(cumulative_loss / rows_seen)
    model.eval()
    with torch.inference_mode():
        prediction = torch.sigmoid(
            model(torch.from_numpy(np.asarray(x_validation, dtype=np.float32)))
        ).numpy()
    if (
        prediction.shape != (len(x_validation),)
        or not np.isfinite(prediction).all()
        or np.any((prediction <= 0) | (prediction >= 1))
    ):
        raise ValueError("E560 prediction audit failed.")
    summary = {
        "seed": seed,
        "epochs": epochs,
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "input_width": x_train.shape[1],
        "training_prior": float(np.mean(y_train)),
        "epoch_1_loss": epoch_losses[0],
        "final_epoch_loss": epoch_losses[-1],
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }
    return prediction.astype(np.float64), summary


def _load_features(
    project_root: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame, _, indices, folds = _load_pilot(project_root)
    paths = discover_project_paths(project_root)
    full = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    semantic_full, similarity_full = prepare_bge_base_features(
        paths.cache_dir, full
    )
    _, dense_full, _ = prepare_semantic_features(paths.cache_dir, full)
    dense = dense_full[indices].copy()
    dense[:, -1:] = similarity_full[indices]
    semantic = semantic_full[indices]
    if (
        semantic.shape != (PILOT_ROWS, 4 * 768)
        or dense.shape[0] != PILOT_ROWS
        or not np.isfinite(semantic).all()
        or not np.isfinite(dense).all()
    ):
        raise ValueError("E560 pilot feature contract changed.")
    return frame, indices, folds, semantic, dense


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    _configure_torch()
    source_hashes = verify_sources(project_root)
    frame, indices, folds, semantic, dense = _load_features(project_root)
    train_mask, validation_mask = _fold_masks(frame, folds, 0)
    x_train, x_validation = _fold_features(
        semantic, dense, train_mask, validation_mask
    )
    synthetic_target = (indices[train_mask] % 2).astype(np.int8)
    if set(synthetic_target.tolist()) != {0, 1}:
        raise RuntimeError("E560 synthetic benchmark labels are invalid.")
    started = time.perf_counter()
    _, summary = train_residual_head(
        x_train,
        synthetic_target,
        x_validation,
        seed=20260725,
        epochs=1,
    )
    elapsed = time.perf_counter() - started
    projected_seconds = elapsed * EPOCHS * 5
    peak_rss = _current_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "one fold/one epoch with deterministic index-parity labels",
        "elapsed_seconds": elapsed,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
        "peak_rss_bytes": peak_rss,
        "proceed": bool(
            projected_seconds <= MAX_PROJECTED_HOURS * 3600
            and peak_rss < MAX_RSS_BYTES
        ),
        "frozen_epochs": EPOCHS,
        "frozen_folds": 5,
        "training_summary": summary,
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
        raise FileNotFoundError("Run the E560 benchmark before validation.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol_id") != PROTOCOL_ID
        or result.get("source_hashes") != verify_sources(project_root)
        or result.get("proceed") is not True
        or result.get("outcome_labels_accessed") is not False
    ):
        raise ValueError("E560 benchmark binding or gate is invalid.")
    return result


def _fold_metric_frame(
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
    residual_prediction = np.full(PILOT_ROWS, np.nan, dtype=np.float64)
    training_summaries: list[dict[str, object]] = []
    peak_rss = _current_rss_bytes()
    for fold in range(5):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        x_train, x_validation = _fold_features(
            semantic, dense, train_mask, validation_mask
        )
        prediction, summary = train_residual_head(
            x_train,
            target[train_mask],
            x_validation,
            seed=20260725 + fold,
        )
        residual_prediction[validation_mask] = prediction
        training_summaries.append({"fold": fold, **summary})
        peak_rss = max(peak_rss, _current_rss_bytes())
    if not np.isfinite(residual_prediction).all():
        raise RuntimeError("E560 OOF predictions are incomplete.")

    baseline_metrics = binary_metrics(target, baseline_prediction)
    metric_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for weight in BLEND_WEIGHTS:
        prediction = (
            (1.0 - weight) * baseline_prediction + weight * residual_prediction
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
                    "pred_residual_head": residual_prediction,
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
    fold_metrics = _fold_metric_frame(
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_residual_head")
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
        "candidate": "E560_residual_nonlinear_head",
        "pilot_protocol": PILOT_PROTOCOL,
        "pilot_rows": PILOT_ROWS,
        "architecture": {
            "hidden_width": HIDDEN_WIDTH,
            "activation": "GELU",
            "dropout": DROPOUT,
            "direct_linear_residual": True,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "loss": "unweighted BCEWithLogitsLoss",
            "dense_scale": DENSE_SCALE,
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
        "training_summaries": training_summaries,
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
        description="Run the frozen E560 residual nonlinear-head screen."
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
            raise RuntimeError("E560 benchmark failed; validation is prohibited.")
        result = {
            "benchmark": benchmark_result,
            "validation": validate(args.project_root),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
