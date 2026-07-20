from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from trace_ace.config import load_config
from trace_ace.hard_validation import (
    _fold_masks,
    _response_transcript_matrix,
    best_prior_shrink,
    build_purged_objective_folds,
    prepare_hash_matrices,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


ROLE_FEATURES = 2**16
CLOSING_FEATURES = 2**15

ROLE_FIELDS = {
    "student": ("student_text", ROLE_FEATURES),
    "tutor": ("tutor_text", ROLE_FEATURES),
    "closing_student": ("closing_student_text", CLOSING_FEATURES),
    "closing_tutor": ("closing_tutor_text", CLOSING_FEATURES),
    "opening": ("opening_text", ROLE_FEATURES),
}

BEHAVIOR_COLUMNS = [
    "role_switches",
    "tutor_words",
    "student_words",
    "tutor_questions",
    "student_questions",
    "tutor_affirmations",
    "tutor_corrections",
    "student_uncertainty",
    "student_reasoning_cues",
    "tutor_reasoning_cues",
    "student_digit_tokens",
    "tutor_digit_tokens",
    "student_mean_words",
    "tutor_mean_words",
    "closing_student_words",
    "closing_tutor_words",
]


def _hasher(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def prepare_role_hashes(cache_dir: Path) -> tuple[pd.Series, dict[str, sparse.csr_matrix]]:
    role_texts = pd.read_parquet(cache_dir / "session_role_texts.parquet")
    if not role_texts["session_id"].is_unique:
        raise ValueError("Role-text cache has duplicate session IDs.")

    matrices: dict[str, sparse.csr_matrix] = {}
    for name, (field, n_features) in ROLE_FIELDS.items():
        matrix_path = cache_dir / f"hash_{name}_word_{n_features}.npz"
        order_path = cache_dir / f"hash_{name}_session_order.parquet"
        if matrix_path.exists() and order_path.exists():
            cached_order = pd.read_parquet(order_path)["session_id"]
            if list(cached_order) != list(role_texts["session_id"]):
                raise ValueError(f"Cached {name} session order is stale.")
            matrices[name] = sparse.load_npz(matrix_path).tocsr()
            continue

        print(f"Hashing {name} text ({n_features} features)...", flush=True)
        matrix = _hasher(n_features).transform(role_texts[field].fillna("")).tocsr()
        sparse.save_npz(matrix_path, matrix, compressed=True)
        role_texts[["session_id"]].to_parquet(order_path, index=False)
        matrices[name] = matrix
    return role_texts["session_id"], matrices


def _response_rows(frame: pd.DataFrame, session_order: pd.Series) -> np.ndarray:
    lookup = pd.Series(np.arange(len(session_order)), index=session_order)
    rows = frame["session_id"].map(lookup)
    if rows.isna().any():
        raise ValueError("At least one modeling session is missing from the role cache.")
    return rows.to_numpy(dtype=np.int64)


def prepare_behavior_features(
    frame: pd.DataFrame,
    cache_dir: Path,
    session_order: pd.Series,
    response_rows: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    behavior = pd.read_parquet(cache_dir / "session_behavior.parquet")
    behavior = behavior.set_index("session_id").loc[list(session_order)].reset_index()
    values = behavior[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float64)[response_rows]
    values = np.log1p(np.maximum(values, 0.0))

    eps = 1.0
    tutor_words = behavior["tutor_words"].to_numpy(dtype=np.float64)[response_rows]
    student_words = behavior["student_words"].to_numpy(dtype=np.float64)[response_rows]
    closing_tutor = behavior["closing_tutor_words"].to_numpy(dtype=np.float64)[response_rows]
    closing_student = behavior["closing_student_words"].to_numpy(dtype=np.float64)[response_rows]
    ratio_values = np.column_stack(
        [
            student_words / (tutor_words + eps),
            closing_student / (student_words + eps),
            closing_tutor / (tutor_words + eps),
            behavior["student_questions"].to_numpy(dtype=np.float64)[response_rows]
            / (student_words + eps),
            behavior["tutor_affirmations"].to_numpy(dtype=np.float64)[response_rows]
            / (tutor_words + eps),
            behavior["tutor_corrections"].to_numpy(dtype=np.float64)[response_rows]
            / (tutor_words + eps),
            behavior["student_uncertainty"].to_numpy(dtype=np.float64)[response_rows]
            / (student_words + eps),
            behavior["student_reasoning_cues"].to_numpy(dtype=np.float64)[response_rows]
            / (student_words + eps),
        ]
    )
    names = BEHAVIOR_COLUMNS + [
        "student_tutor_word_ratio",
        "closing_student_share",
        "closing_tutor_share",
        "student_question_rate",
        "tutor_affirmation_rate",
        "tutor_correction_rate",
        "student_uncertainty_rate",
        "student_reasoning_rate",
    ]
    return np.column_stack([values, ratio_values]), names


def _objective_hash(frame: pd.DataFrame, n_features: int) -> sparse.csr_matrix:
    return _hasher(n_features).transform(frame["learning_objective"].fillna("")).tocsr()


def _row_cosine(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def _unit_weighted_hstack(
    blocks: list[tuple[sparse.csr_matrix, float]],
) -> sparse.csr_matrix:
    norm = math.sqrt(sum(weight * weight for _, weight in blocks))
    if norm <= 0:
        raise ValueError("Feature block weights must have positive norm.")
    return sparse.hstack(
        [matrix * np.float32(weight / norm) for matrix, weight in blocks],
        format="csr",
        dtype=np.float32,
    )


def sparse_dense_sgd_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    matrix: sparse.csr_matrix,
    dense_values: np.ndarray | None,
    alpha: float,
    n_splits: int,
    dense_weight: float = 0.2,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    target = frame["target"].to_numpy()
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        x_train = matrix[train_mask]
        x_validation = matrix[validation_mask]
        if dense_values is not None:
            scaler = StandardScaler()
            dense_train = scaler.fit_transform(dense_values[train_mask]).astype(np.float32)
            dense_validation = scaler.transform(dense_values[validation_mask]).astype(np.float32)
            x_train = sparse.hstack(
                [x_train, sparse.csr_matrix(dense_train * dense_weight)], format="csr"
            )
            x_validation = sparse.hstack(
                [x_validation, sparse.csr_matrix(dense_validation * dense_weight)], format="csr"
            )
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=100,
            tol=1e-4,
            shuffle=True,
            random_state=20260716 + fold,
            average=True,
        )
        model.fit(x_train, target[train_mask])
        predictions[validation_mask] = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete role hard-validation predictions.")
    return predictions


def _best_pair_blend(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    prior: np.ndarray,
) -> dict[str, float]:
    best = {"first_weight": 0.0, "model_weight": 0.0, "log_loss": float("inf")}
    for first_weight in np.linspace(0.0, 1.0, 21):
        model_prediction = first_weight * first + (1.0 - first_weight) * second
        for model_weight in np.linspace(0.4, 1.0, 31):
            prediction = model_weight * model_prediction + (1.0 - model_weight) * prior
            score = float(log_loss(target, prediction, labels=[0, 1]))
            if score < best["log_loss"]:
                best = {
                    "first_weight": float(first_weight),
                    "model_weight": float(model_weight),
                    "log_loss": score,
                }
    return best


def run_role_hard_validation(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    n_splits = int(config["n_splits"])
    folds, fold_summaries = build_purged_objective_folds(
        frame, n_splits=n_splits, seed=int(config["fold_seed"])
    )

    full_session_hash, _ = prepare_hash_matrices(frame, paths.cache_dir)
    full = _response_transcript_matrix(frame, paths.cache_dir, full_session_hash)
    session_order, session_roles = prepare_role_hashes(paths.cache_dir)
    response_rows = _response_rows(frame, session_order)
    roles = {name: matrix[response_rows] for name, matrix in session_roles.items()}
    objective_64k = _objective_hash(frame, ROLE_FEATURES)
    objective_32k = _objective_hash(frame, CLOSING_FEATURES)
    behavior, behavior_names = prepare_behavior_features(
        frame, paths.cache_dir, session_order, response_rows
    )

    alignment = np.column_stack(
        [
            _row_cosine(roles["student"], objective_64k),
            _row_cosine(roles["tutor"], objective_64k),
            _row_cosine(roles["opening"], objective_64k),
            _row_cosine(roles["closing_student"], objective_32k),
            _row_cosine(roles["closing_tutor"], objective_32k),
        ]
    )
    dense = np.column_stack([behavior, alignment])
    dense_names = behavior_names + [
        "objective_student_cosine",
        "objective_tutor_cosine",
        "objective_opening_cosine",
        "objective_closing_student_cosine",
        "objective_closing_tutor_cosine",
    ]

    variants: dict[str, tuple[sparse.csr_matrix, np.ndarray | None]] = {
        "student_objective": (
            _unit_weighted_hstack([(roles["student"], 1.0), (objective_64k, 1.0)]),
            None,
        ),
        "tutor_objective": (
            _unit_weighted_hstack([(roles["tutor"], 1.0), (objective_64k, 1.0)]),
            None,
        ),
        "closing_student_objective": (
            _unit_weighted_hstack(
                [(roles["closing_student"], 1.0), (objective_32k, 1.0)]
            ),
            None,
        ),
        "opening_objective": (
            _unit_weighted_hstack([(roles["opening"], 1.0), (objective_64k, 1.0)]),
            None,
        ),
        "roles_objective": (
            _unit_weighted_hstack(
                [
                    (roles["student"], 0.75),
                    (roles["tutor"], 0.55),
                    (objective_64k, 1.0),
                ]
            ),
            None,
        ),
        "closing_roles_objective": (
            _unit_weighted_hstack(
                [
                    (roles["closing_student"], 0.75),
                    (roles["closing_tutor"], 0.55),
                    (objective_32k, 1.0),
                ]
            ),
            None,
        ),
        "full_student_closing_objective": (
            _unit_weighted_hstack(
                [
                    (full, 0.70),
                    (roles["student"], 0.65),
                    (roles["closing_student"], 0.45),
                    (objective_64k, 1.0),
                ]
            ),
            None,
        ),
        "roles_objective_dense": (
            _unit_weighted_hstack(
                [
                    (roles["student"], 0.75),
                    (roles["tutor"], 0.55),
                    (objective_64k, 1.0),
                ]
            ),
            dense,
        ),
    }

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
    alphas = [3e-5, 1e-4]
    for variant_name, (matrix, dense_values) in variants.items():
        for alpha in alphas:
            prediction = sparse_dense_sgd_oof(
                frame,
                folds,
                matrix,
                dense_values,
                alpha,
                n_splits,
            )
            column = f"pred_{variant_name}_a{alpha:g}"
            prediction_frame[column] = prediction
            metrics = binary_metrics(target, prediction)
            shrink = best_prior_shrink(target, prediction, fold_prior)
            metric_rows.append(
                {
                    "model": variant_name,
                    "alpha": alpha,
                    **metrics,
                    **{f"shrink_{key}": value for key, value in shrink.items()},
                }
            )
            print(
                f"{variant_name} alpha={alpha:g}: loss={metrics['log_loss']:.6f}, "
                f"auc={metrics['roc_auc']:.4f}, shrink={shrink['weight']:.2f}/"
                f"{shrink['log_loss']:.6f}",
                flush=True,
            )

    metric_frame = pd.DataFrame(metric_rows).sort_values("shrink_log_loss")
    candidate_columns = [
        f"pred_{row.model}_a{row.alpha:g}"
        for row in metric_frame.head(8).itertuples(index=False)
    ]
    blend_rows: list[dict[str, object]] = []
    for first_index, first_column in enumerate(candidate_columns):
        for second_column in candidate_columns[first_index + 1 :]:
            blend = _best_pair_blend(
                target,
                prediction_frame[first_column].to_numpy(),
                prediction_frame[second_column].to_numpy(),
                fold_prior,
            )
            model_prediction = (
                blend["first_weight"] * prediction_frame[first_column].to_numpy()
                + (1.0 - blend["first_weight"])
                * prediction_frame[second_column].to_numpy()
            )
            prediction = (
                blend["model_weight"] * model_prediction
                + (1.0 - blend["model_weight"]) * fold_prior
            )
            metrics = binary_metrics(target, prediction)
            blend_rows.append(
                {
                    "first": first_column,
                    "second": second_column,
                    **blend,
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                }
            )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_role_hard_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_frame.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    metric_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    blend_frame = pd.DataFrame(blend_rows).sort_values("log_loss")
    blend_frame.to_csv(run_dir / "blends.csv", index=False, lineterminator="\n")
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "learning_objective-disjoint validation with validation-session purge",
        "folds": fold_summaries,
        "role_hash_features": {
            key: n_features for key, (_, n_features) in ROLE_FIELDS.items()
        },
        "dense_features": dense_names,
        "best_models": metric_frame.head(10).to_dict(orient="records"),
        "best_blends": blend_frame.head(10).to_dict(orient="records"),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metric_frame,
        "blends": blend_frame,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run role-aware purged-objective validation.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_role_hard_validation(args.project_root)
    print(f"Role hard validation complete: {result['run_id']}")
    print(result["metrics"].head(12).to_string(index=False))
    print("Best blends:")
    print(result["blends"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
