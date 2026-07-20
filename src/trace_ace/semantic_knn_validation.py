from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


@dataclass(frozen=True)
class KnnConfig:
    name: str
    objective_neighbors: int
    response_neighbors: int
    objective_weight: float
    temperature: float
    prior_mass: float


DEFAULT_CONFIGS = (
    KnnConfig("m8_k64_balanced", 8, 64, 0.35, 12.0, 2.0),
    KnnConfig("m20_k128_balanced", 20, 128, 0.35, 12.0, 2.0),
    KnnConfig("m40_k256_balanced", 40, 256, 0.35, 12.0, 2.0),
    KnnConfig("m20_k128_objective", 20, 128, 0.60, 12.0, 2.0),
)


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-8))


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), values.shape[-1])
    if count <= 0:
        raise ValueError("Neighbor count must be positive.")
    if count == values.shape[-1]:
        return np.argsort(values, axis=-1)[..., ::-1]
    partition = np.argpartition(values, -count, axis=-1)[..., -count:]
    selected = np.take_along_axis(values, partition, axis=-1)
    order = np.argsort(selected, axis=-1)[..., ::-1]
    return np.take_along_axis(partition, order, axis=-1)


def semantic_knn_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    context_embeddings: np.ndarray,
    objective_embeddings: np.ndarray,
    config: KnnConfig,
    n_splits: int,
) -> np.ndarray:
    if len(frame) != len(folds):
        raise ValueError("Fold assignments do not align with modeling rows.")
    context = _row_normalize(context_embeddings)
    objective = _row_normalize(objective_embeddings)
    if context.shape != objective.shape or context.shape[0] != len(frame):
        raise ValueError("Semantic embeddings do not align with modeling rows.")
    if not np.isfinite(context).all() or not np.isfinite(objective).all():
        raise ValueError("Semantic embeddings contain non-finite values.")

    target = frame["target"].to_numpy(dtype=np.float64)
    objective_ids = frame["learning_objective_id"].astype(str).to_numpy()
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        train_objective_ids = objective_ids[train_indices]
        validation_objective_ids = objective_ids[validation_indices]
        prior = float(target[train_indices].mean())

        unique_train_ids, first_positions = np.unique(
            train_objective_ids, return_index=True
        )
        train_objective_prototypes = objective[train_indices[first_positions]]
        for validation_objective_id in np.unique(validation_objective_ids):
            group_indices = validation_indices[
                validation_objective_ids == validation_objective_id
            ]
            query_objective = objective[group_indices[0]]
            prototype_similarity = train_objective_prototypes @ query_objective
            selected_objectives = unique_train_ids[
                _top_indices(prototype_similarity, config.objective_neighbors)
            ]
            candidate_indices = train_indices[
                np.isin(train_objective_ids, selected_objectives)
            ]
            if candidate_indices.size == 0:
                prediction[group_indices] = prior
                continue

            context_similarity = context[group_indices] @ context[candidate_indices].T
            objective_similarity = objective[group_indices] @ objective[candidate_indices].T
            score = (
                np.float32(1.0 - config.objective_weight) * context_similarity
                + np.float32(config.objective_weight) * objective_similarity
            )
            neighbor_positions = _top_indices(score, config.response_neighbors)
            neighbor_scores = np.take_along_axis(score, neighbor_positions, axis=1)
            neighbor_targets = target[candidate_indices[neighbor_positions]]
            weights = np.exp(
                np.float64(config.temperature)
                * (neighbor_scores - neighbor_scores[:, :1]).astype(np.float64)
            )
            weighted_positive = np.sum(weights * neighbor_targets, axis=1)
            weight_sum = np.sum(weights, axis=1)
            prediction[group_indices] = (
                weighted_positive + config.prior_mass * prior
            ) / (weight_sum + config.prior_mass)

    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete semantic kNN predictions.")
    return np.clip(prediction, 1e-5, 1.0 - 1e-5)


def _load_baseline(
    experiments_dir: Path,
    baseline_run_id: str,
    protocols: set[str] | None,
) -> pd.DataFrame:
    path = experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    baseline = pd.read_parquet(path)
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline predictions match the requested protocols.")
    return baseline.reset_index(drop=True)


def run_semantic_knn_validation(
    project_root: str | Path,
    configs: Iterable[KnnConfig] = DEFAULT_CONFIGS,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    context = np.load(paths.cache_dir / "bge_small_context_256.npy")
    objective = np.load(paths.cache_dir / "bge_small_objective_256.npy")
    baseline = _load_baseline(paths.experiments_dir, baseline_run_id, protocols)
    target = frame["target"].to_numpy(dtype=np.float64)
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    config_list = list(configs)
    for config_index, config in enumerate(config_list, start=1):
        for protocol_name in protocol_names:
            protocol_base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
            if list(protocol_base["response_id"]) != list(frame["response_id"]):
                raise ValueError(f"Baseline row order mismatch for {protocol_name}.")
            folds = protocol_base["fold"].to_numpy(dtype=np.int8)
            print(
                f"Semantic kNN {config_index}/{len(config_list)}: "
                f"{config.name} on {protocol_name}",
                flush=True,
            )
            knn_prediction = semantic_knn_oof(
                frame,
                folds,
                context,
                objective,
                config,
                n_splits=int(folds.max()) + 1,
            )
            baseline_prediction = protocol_base["pred_v02"].to_numpy(dtype=np.float64)
            baseline_metrics = binary_metrics(target, baseline_prediction)
            candidates = [("standalone", 1.0, knn_prediction)]
            candidates.extend(
                (
                    "blend",
                    weight,
                    (1.0 - weight) * baseline_prediction + weight * knn_prediction,
                )
                for weight in BLEND_WEIGHTS
            )
            for prediction_type, weight, candidate_prediction in candidates:
                metrics = binary_metrics(target, candidate_prediction)
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": config.name,
                        "prediction_type": prediction_type,
                        "new_model_weight": weight,
                        **metrics,
                        "delta_log_loss_vs_v02": (
                            metrics["log_loss"] - baseline_metrics["log_loss"]
                        ),
                        "delta_roc_auc_vs_v02": (
                            metrics["roc_auc"] - baseline_metrics["roc_auc"]
                        ),
                    }
                )
                for fold in range(int(folds.max()) + 1):
                    fold_mask = folds == fold
                    fold_metrics = binary_metrics(
                        target[fold_mask], candidate_prediction[fold_mask]
                    )
                    fold_baseline_metrics = binary_metrics(
                        target[fold_mask], baseline_prediction[fold_mask]
                    )
                    fold_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold,
                            "model": config.name,
                            "prediction_type": prediction_type,
                            "new_model_weight": weight,
                            **fold_metrics,
                            "delta_log_loss_vs_v02": (
                                fold_metrics["log_loss"]
                                - fold_baseline_metrics["log_loss"]
                            ),
                            "delta_roc_auc_vs_v02": (
                                fold_metrics["roc_auc"]
                                - fold_baseline_metrics["roc_auc"]
                            ),
                        }
                    )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "protocol": protocol_name,
                        "response_id": frame["response_id"],
                        "fold": folds,
                        "target": target,
                        "model": config.name,
                        "pred_v02": baseline_prediction,
                        "pred_candidate": knn_prediction,
                    }
                )
            )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_semantic_knn_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "baseline_run_id": baseline_run_id,
        "protocols": protocol_names,
        "configs": [asdict(config) for config in config_list],
        "blend_weights": list(BLEND_WEIGHTS),
        "legality": (
            "Each prediction retrieves only labeled training rows selected from the "
            "sample's own objective and context embeddings. No test aggregate is used."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics_frame}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate objective-conditioned semantic label retrieval."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_semantic_knn_validation(
        args.project_root,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
    )
    print(f"Semantic kNN validation complete: {result['run_id']}")
    print(result["metrics"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
