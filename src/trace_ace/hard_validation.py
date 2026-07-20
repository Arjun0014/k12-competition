from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from trace_ace.config import load_config
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics


HASH_TRANSCRIPT_FEATURES = 2**17
HASH_OBJECTIVE_FEATURES = 2**14

META_COLUMNS = [
    "n_utterances",
    "duration_seconds",
    "content_chars",
    "tutor_utterances",
    "student_utterances",
    "background_utterances",
    "other_role_utterances",
    "unclear_utterances",
    "empty_utterances",
    "objective_chars",
    "objective_words",
    "student_fraction",
    "tutor_fraction",
    "background_fraction",
    "unclear_fraction",
]


def build_purged_objective_folds(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(frame, frame["target"], groups=frame["learning_objective_id"])
    ):
        folds[validation_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("Incomplete objective-fold assignment.")

    summaries: list[dict[str, object]] = []
    for fold in range(n_splits):
        validation_mask = folds == fold
        validation_sessions = set(frame.loc[validation_mask, "session_id"])
        training_mask = (~validation_mask) & (~frame["session_id"].isin(validation_sessions).to_numpy())
        training_objectives = set(frame.loc[training_mask, "learning_objective_id"])
        validation_objectives = set(frame.loc[validation_mask, "learning_objective_id"])
        if training_objectives.intersection(validation_objectives):
            raise RuntimeError(f"Objective leakage in hard fold {fold}.")
        if set(frame.loc[training_mask, "session_id"]).intersection(validation_sessions):
            raise RuntimeError(f"Session leakage in hard fold {fold}.")
        summaries.append(
            {
                "fold": fold,
                "train_rows_after_session_purge": int(training_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "validation_sessions": len(validation_sessions),
                "validation_objectives": len(validation_objectives),
                "validation_positive_rate": float(frame.loc[validation_mask, "target"].mean()),
            }
        )
    return folds, summaries


def _hash_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "transcript": cache_dir / f"hash_full_word_{HASH_TRANSCRIPT_FEATURES}.npz",
        "objective": cache_dir / f"hash_objective_word_{HASH_OBJECTIVE_FEATURES}.npz",
        "session_order": cache_dir / "hash_session_order.parquet",
        "response_order": cache_dir / "hash_response_order.parquet",
    }


def prepare_hash_matrices(
    frame: pd.DataFrame,
    cache_dir: Path,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    paths = _hash_cache_paths(cache_dir)
    if all(path.exists() for path in paths.values()):
        session_order = pd.read_parquet(paths["session_order"])["session_id"]
        response_order = pd.read_parquet(paths["response_order"])["response_id"]
        if list(response_order) != list(frame["response_id"]):
            raise ValueError("Cached hash response order does not match modeling data.")
        session_texts = pd.read_parquet(cache_dir / "session_texts.parquet", columns=["session_id"])
        if list(session_order) != list(session_texts["session_id"]):
            raise ValueError("Cached hash session order does not match text cache.")
        return sparse.load_npz(paths["transcript"]), sparse.load_npz(paths["objective"])

    session_texts = pd.read_parquet(cache_dir / "session_texts.parquet")
    transcript_hasher = HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=HASH_TRANSCRIPT_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )
    transcript_matrix = transcript_hasher.transform(session_texts["full_text"]).tocsr()
    objective_hasher = HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=HASH_OBJECTIVE_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )
    objective_matrix = objective_hasher.transform(frame["learning_objective"]).tocsr()
    sparse.save_npz(paths["transcript"], transcript_matrix, compressed=True)
    sparse.save_npz(paths["objective"], objective_matrix, compressed=True)
    session_texts[["session_id"]].to_parquet(paths["session_order"], index=False)
    frame[["response_id"]].to_parquet(paths["response_order"], index=False)
    return transcript_matrix, objective_matrix


def _response_transcript_matrix(
    frame: pd.DataFrame,
    cache_dir: Path,
    session_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    session_order = pd.read_parquet(cache_dir / "hash_session_order.parquet")["session_id"]
    lookup = pd.Series(np.arange(len(session_order)), index=session_order)
    rows = frame["session_id"].map(lookup)
    if rows.isna().any():
        raise ValueError("Missing session in hashed transcript cache.")
    return session_matrix[rows.to_numpy(dtype=np.int64)]


def _meta_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[META_COLUMNS].to_numpy(dtype=np.float64)
    count_columns = list(range(11))
    values[:, count_columns] = np.log1p(np.maximum(values[:, count_columns], 0.0))
    return values


def _fold_masks(frame: pd.DataFrame, folds: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    validation_mask = folds == fold
    validation_sessions = set(frame.loc[validation_mask, "session_id"])
    training_mask = (~validation_mask) & (~frame["session_id"].isin(validation_sessions).to_numpy())
    return training_mask, validation_mask


def metadata_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    priors = np.full(len(frame), np.nan, dtype=np.float64)
    values = _meta_matrix(frame)
    target = frame["target"].to_numpy()
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(values[train_mask])
        x_validation = scaler.transform(values[validation_mask])
        model = LogisticRegression(C=0.3, solver="liblinear", max_iter=500, random_state=0)
        model.fit(x_train, target[train_mask])
        predictions[validation_mask] = model.predict_proba(x_validation)[:, 1]
        priors[validation_mask] = target[train_mask].mean()
    return predictions, priors


def sparse_sgd_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    matrix: sparse.csr_matrix,
    alpha: float,
    n_splits: int,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    target = frame["target"].to_numpy()
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=80,
            tol=1e-4,
            shuffle=True,
            random_state=20260716 + fold,
            average=True,
        )
        model.fit(matrix[train_mask], target[train_mask])
        predictions[validation_mask] = model.predict_proba(matrix[validation_mask])[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete hard-validation predictions.")
    return predictions


def best_prior_shrink(
    target: np.ndarray,
    prediction: np.ndarray,
    prior: np.ndarray,
) -> dict[str, float]:
    best = {"weight": 0.0, "log_loss": float("inf")}
    for weight in np.linspace(0.0, 1.0, 101):
        blended = (1.0 - weight) * prior + weight * prediction
        score = float(log_loss(target, blended, labels=[0, 1]))
        if score < best["log_loss"]:
            best = {"weight": float(weight), "log_loss": score}
    return best


def run_hard_validation(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    n_splits = int(config["n_splits"])
    folds, fold_summaries = build_purged_objective_folds(
        frame, n_splits=n_splits, seed=int(config["fold_seed"])
    )
    pd.DataFrame(
        {"response_id": frame["response_id"], "objective_fold": folds}
    ).to_parquet(paths.cache_dir / "objective_folds.parquet", index=False)

    session_hash, objective_hash = prepare_hash_matrices(frame, paths.cache_dir)
    response_hash = _response_transcript_matrix(frame, paths.cache_dir, session_hash)
    meta_prediction, fold_prior = metadata_oof(frame, folds, n_splits)

    variants = {
        "transcript": response_hash,
        "transcript_objective": sparse.hstack(
            [response_hash, objective_hash], format="csr", dtype=np.float32
        ),
    }
    alphas = [3e-5, 1e-4, 3e-4, 1e-3]
    prediction_frame = pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "target": frame["target"],
            "objective_fold": folds,
            "pred_fold_prior": fold_prior,
            "pred_metadata": meta_prediction,
        }
    )
    metric_rows: list[dict[str, object]] = []
    target = frame["target"].to_numpy()

    for name in ["fold_prior", "metadata"]:
        prediction = prediction_frame[f"pred_{name}"].to_numpy()
        metrics = binary_metrics(target, prediction)
        shrink = best_prior_shrink(target, prediction, fold_prior)
        metric_rows.append({"model": name, "alpha": np.nan, **metrics, **{f"shrink_{k}": v for k, v in shrink.items()}})

    for variant_name, matrix in variants.items():
        for alpha in alphas:
            column = f"pred_{variant_name}_a{alpha:g}"
            prediction = sparse_sgd_oof(
                frame=frame,
                folds=folds,
                matrix=matrix,
                alpha=alpha,
                n_splits=n_splits,
            )
            prediction_frame[column] = prediction
            metrics = binary_metrics(target, prediction)
            shrink = best_prior_shrink(target, prediction, fold_prior)
            metric_rows.append(
                {
                    "model": variant_name,
                    "alpha": alpha,
                    **metrics,
                    **{f"shrink_{k}": value for k, value in shrink.items()},
                }
            )
            print(
                f"{variant_name} alpha={alpha:g}: loss={metrics['log_loss']:.6f}, "
                f"auc={metrics['roc_auc']:.4f}, shrink={shrink['weight']:.2f}/"
                f"{shrink['log_loss']:.6f}",
                flush=True,
            )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_hard_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_frame.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    metrics_frame = pd.DataFrame(metric_rows).sort_values("shrink_log_loss")
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "learning_objective-disjoint validation with validation-session purge",
        "folds": fold_summaries,
        "hash_features": {
            "transcript": HASH_TRANSCRIPT_FEATURES,
            "objective": HASH_OBJECTIVE_FEATURES,
        },
        "best_models": metrics_frame.head(10).to_dict(orient="records"),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics_frame}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run purged-objective hard validation.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_hard_validation(args.project_root)
    print(f"Hard validation complete: {result['run_id']}")
    print(result["metrics"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
