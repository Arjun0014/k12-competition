from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.config import load_config
from trace_ace.hard_validation import (
    _fold_masks,
    best_prior_shrink,
    build_purged_objective_folds,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.retrieval_hard_validation import RETRIEVAL_DENSE_COLUMNS
from trace_ace.role_hard_validation import (
    _response_rows,
    prepare_behavior_features,
    prepare_role_hashes,
)


def prepare_semantic_features(
    cache_dir: Path,
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    context = np.load(cache_dir / "bge_small_context_256.npy").astype(np.float32)
    objective = np.load(cache_dir / "bge_small_objective_256.npy").astype(np.float32)
    expected_shape = (len(frame), 384)
    if context.shape != expected_shape or objective.shape != expected_shape:
        raise ValueError("BGE semantic cache shape does not match modeling data.")
    if not np.isfinite(context).all() or not np.isfinite(objective).all():
        raise ValueError("BGE semantic cache contains non-finite values.")

    interaction = context * objective * np.float32(6.0)
    difference = np.abs(context - objective) * np.float32(0.7)
    variants = {
        "semantic_context": context,
        "semantic_context_objective": np.column_stack(
            [context * np.float32(0.7), objective * np.float32(0.7)]
        ),
        "semantic_interaction": np.column_stack(
            [
                context * np.float32(0.7),
                objective * np.float32(0.7),
                interaction,
                difference,
            ]
        ),
    }

    session_order, _ = prepare_role_hashes(cache_dir)
    response_rows = _response_rows(frame, session_order)
    behavior, behavior_names = prepare_behavior_features(
        frame, cache_dir, session_order, response_rows
    )
    contexts = pd.read_parquet(cache_dir / "response_objective_context.parquet")
    if list(contexts["response_id"]) != list(frame["response_id"]):
        raise ValueError("Objective-context order does not match semantic modeling data.")
    retrieval = contexts[RETRIEVAL_DENSE_COLUMNS].to_numpy(dtype=np.float64)
    retrieval[:, [0, 1, 2, 3, 4, 8, 9]] = np.log1p(
        np.maximum(retrieval[:, [0, 1, 2, 3, 4, 8, 9]], 0.0)
    )
    similarity = np.sum(context * objective, axis=1, keepdims=True)
    dense = np.column_stack([behavior, retrieval, similarity])
    dense_names = behavior_names + RETRIEVAL_DENSE_COLUMNS + ["bge_objective_context_cosine"]
    return variants, dense, dense_names


def semantic_logistic_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    features: np.ndarray,
    extra_dense: np.ndarray | None,
    regularization_c: float,
    n_splits: int,
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    target = frame["target"].to_numpy()
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        x_train = features[train_mask]
        x_validation = features[validation_mask]
        if extra_dense is not None:
            scaler = StandardScaler()
            dense_train = scaler.fit_transform(extra_dense[train_mask]).astype(np.float32)
            dense_validation = scaler.transform(extra_dense[validation_mask]).astype(np.float32)
            x_train = np.column_stack([x_train, dense_train * np.float32(0.08)])
            x_validation = np.column_stack(
                [x_validation, dense_validation * np.float32(0.08)]
            )
        model = LogisticRegression(
            C=regularization_c,
            solver="lbfgs",
            max_iter=400,
            tol=1e-5,
            random_state=20260716 + fold,
        )
        model.fit(x_train, target[train_mask])
        prediction[validation_mask] = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete semantic hard-validation predictions.")
    return prediction


def run_semantic_hard_validation(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    n_splits = int(config["n_splits"])
    folds, fold_summaries = build_purged_objective_folds(
        frame, n_splits=n_splits, seed=int(config["fold_seed"])
    )
    variants, dense, dense_names = prepare_semantic_features(paths.cache_dir, frame)
    variants["semantic_interaction_dense"] = variants["semantic_interaction"]

    target = frame["target"].to_numpy()
    fold_prior = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        fold_prior[validation_mask] = target[train_mask].mean()
    prediction_frame = pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "target": target,
            "objective_fold": folds,
            "pred_fold_prior": fold_prior,
        }
    )
    metric_rows: list[dict[str, object]] = []
    for name, features in variants.items():
        extra_dense = dense if name.endswith("_dense") else None
        for regularization_c in [0.03, 0.1, 0.3]:
            prediction = semantic_logistic_oof(
                frame,
                folds,
                features,
                extra_dense,
                regularization_c,
                n_splits,
            )
            prediction_frame[f"pred_{name}_c{regularization_c:g}"] = prediction
            metrics = binary_metrics(target, prediction)
            shrink = best_prior_shrink(target, prediction, fold_prior)
            metric_rows.append(
                {
                    "model": name,
                    "regularization_c": regularization_c,
                    **metrics,
                    **{f"shrink_{key}": value for key, value in shrink.items()},
                }
            )
            print(
                f"{name} C={regularization_c:g}: loss={metrics['log_loss']:.6f}, "
                f"auc={metrics['roc_auc']:.4f}, shrink={shrink['weight']:.2f}/"
                f"{shrink['log_loss']:.6f}",
                flush=True,
            )

    metrics_frame = pd.DataFrame(metric_rows).sort_values("shrink_log_loss")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_semantic_hard_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_frame.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "learning_objective-disjoint validation with validation-session purge",
        "encoder": "BAAI/bge-small-en-v1.5",
        "encoder_max_sequence_length": 256,
        "folds": fold_summaries,
        "dense_features": dense_names,
        "best_models": metrics_frame.head(10).to_dict(orient="records"),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics_frame}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run BGE semantic hard validation.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_semantic_hard_validation(args.project_root)
    print(f"Semantic hard validation complete: {result['run_id']}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
