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
from trace_ace.role_hard_validation import (
    ROLE_FEATURES,
    _objective_hash,
    _response_rows,
    _row_cosine,
    _unit_weighted_hstack,
    prepare_behavior_features,
    prepare_role_hashes,
    sparse_dense_sgd_oof,
)


RETRIEVAL_WORD_FEATURES = 2**16
RETRIEVAL_CHAR_FEATURES = 2**17

RETRIEVAL_DENSE_COLUMNS = [
    "retrieval_total_lines",
    "retrieval_positive_lines",
    "retrieval_selected_lines",
    "retrieval_objective_terms",
    "retrieval_matched_terms",
    "retrieval_term_coverage",
    "retrieval_first_match_position",
    "retrieval_last_match_position",
    "retrieval_mean_positive_score",
    "retrieval_max_score",
]


def _word_hasher() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=RETRIEVAL_WORD_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def _char_hasher() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=RETRIEVAL_CHAR_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def prepare_retrieval_hashes(
    cache_dir: Path,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, sparse.csr_matrix, sparse.csr_matrix]:
    contexts = pd.read_parquet(cache_dir / "response_objective_context.parquet")
    if list(contexts["response_id"]) != list(frame["response_id"]):
        raise ValueError("Objective-context cache order does not match modeling data.")
    text = contexts["objective_context"].fillna("").str.partition("\n")[2]

    word_path = cache_dir / f"hash_objective_context_word_{RETRIEVAL_WORD_FEATURES}.npz"
    if word_path.exists():
        word_matrix = sparse.load_npz(word_path).tocsr()
    else:
        print("Hashing objective-conditioned word contexts...", flush=True)
        word_matrix = _word_hasher().transform(text).tocsr()
        sparse.save_npz(word_path, word_matrix, compressed=True)

    char_path = cache_dir / f"hash_objective_context_char_{RETRIEVAL_CHAR_FEATURES}.npz"
    if char_path.exists():
        char_matrix = sparse.load_npz(char_path).tocsr()
    else:
        print("Hashing objective-conditioned character contexts...", flush=True)
        char_matrix = _char_hasher().transform(text).tocsr()
        sparse.save_npz(char_path, char_matrix, compressed=True)
    if word_matrix.shape[0] != len(frame) or char_matrix.shape[0] != len(frame):
        raise ValueError("Objective-context hash row count is invalid.")
    return contexts, word_matrix, char_matrix


def _retrieval_dense(contexts: pd.DataFrame) -> np.ndarray:
    values = contexts[RETRIEVAL_DENSE_COLUMNS].to_numpy(dtype=np.float64)
    count_columns = [0, 1, 2, 3, 4, 8, 9]
    values[:, count_columns] = np.log1p(np.maximum(values[:, count_columns], 0.0))
    return values


def run_retrieval_hard_validation(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    n_splits = int(config["n_splits"])
    folds, fold_summaries = build_purged_objective_folds(
        frame, n_splits=n_splits, seed=int(config["fold_seed"])
    )

    full_session, _ = prepare_hash_matrices(frame, paths.cache_dir)
    full = _response_transcript_matrix(frame, paths.cache_dir, full_session)
    contexts, retrieval_word, retrieval_char = prepare_retrieval_hashes(paths.cache_dir, frame)
    objective_word = _objective_hash(frame, RETRIEVAL_WORD_FEATURES)
    objective_char = _char_hasher().transform(frame["learning_objective"].fillna("")).tocsr()
    word_overlap = retrieval_word.multiply(objective_word).tocsr()
    char_overlap = retrieval_char.multiply(objective_char).tocsr()

    session_order, session_roles = prepare_role_hashes(paths.cache_dir)
    response_rows = _response_rows(frame, session_order)
    student = session_roles["student"][response_rows]
    tutor = session_roles["tutor"][response_rows]
    behavior, behavior_names = prepare_behavior_features(
        frame, paths.cache_dir, session_order, response_rows
    )
    alignment = np.column_stack(
        [
            _row_cosine(student, objective_word),
            _row_cosine(tutor, objective_word),
            _row_cosine(retrieval_word, objective_word),
            _row_cosine(retrieval_char, objective_char),
        ]
    )
    dense = np.column_stack([behavior, _retrieval_dense(contexts), alignment])
    dense_names = behavior_names + RETRIEVAL_DENSE_COLUMNS + [
        "objective_student_cosine",
        "objective_tutor_cosine",
        "objective_context_word_cosine",
        "objective_context_char_cosine",
    ]

    variants: dict[str, tuple[sparse.csr_matrix, np.ndarray | None]] = {
        "retrieval_word_objective": (
            _unit_weighted_hstack([(retrieval_word, 1.0), (objective_word, 1.0)]),
            None,
        ),
        "retrieval_word_overlap_objective": (
            _unit_weighted_hstack(
                [(retrieval_word, 0.85), (word_overlap, 0.55), (objective_word, 1.0)]
            ),
            None,
        ),
        "retrieval_char_objective": (
            _unit_weighted_hstack([(retrieval_char, 1.0), (objective_char, 1.0)]),
            None,
        ),
        "retrieval_word_char_objective": (
            _unit_weighted_hstack(
                [
                    (retrieval_word, 0.75),
                    (retrieval_char, 0.65),
                    (objective_word, 0.75),
                    (objective_char, 0.65),
                ]
            ),
            None,
        ),
        "retrieval_word_char_overlap_objective": (
            _unit_weighted_hstack(
                [
                    (retrieval_word, 0.70),
                    (retrieval_char, 0.55),
                    (word_overlap, 0.45),
                    (char_overlap, 0.35),
                    (objective_word, 0.75),
                    (objective_char, 0.55),
                ]
            ),
            None,
        ),
        "full_retrieval_objective": (
            _unit_weighted_hstack(
                [(full, 0.80), (retrieval_word, 0.75), (objective_word, 1.0)]
            ),
            None,
        ),
        "roles_retrieval_objective_dense": (
            _unit_weighted_hstack(
                [
                    (student, 0.60),
                    (tutor, 0.45),
                    (retrieval_word, 0.70),
                    (objective_word, 1.0),
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
    predictions = pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "target": target,
            "objective_fold": folds,
            "pred_fold_prior": fold_prior,
        }
    )
    metric_rows: list[dict[str, object]] = []
    for name, (matrix, dense_values) in variants.items():
        for alpha in [3e-5, 1e-4]:
            prediction = sparse_dense_sgd_oof(
                frame,
                folds,
                matrix,
                dense_values,
                alpha,
                n_splits,
            )
            predictions[f"pred_{name}_a{alpha:g}"] = prediction
            metrics = binary_metrics(target, prediction)
            shrink = best_prior_shrink(target, prediction, fold_prior)
            metric_rows.append(
                {
                    "model": name,
                    "alpha": alpha,
                    **metrics,
                    **{f"shrink_{key}": value for key, value in shrink.items()},
                }
            )
            print(
                f"{name} alpha={alpha:g}: loss={metrics['log_loss']:.6f}, "
                f"auc={metrics['roc_auc']:.4f}, shrink={shrink['weight']:.2f}/"
                f"{shrink['log_loss']:.6f}",
                flush=True,
            )

    metrics_frame = pd.DataFrame(metric_rows).sort_values("shrink_log_loss")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_retrieval_hard_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "learning_objective-disjoint validation with validation-session purge",
        "folds": fold_summaries,
        "hash_features": {
            "retrieval_word": RETRIEVAL_WORD_FEATURES,
            "retrieval_char": RETRIEVAL_CHAR_FEATURES,
        },
        "dense_features": dense_names,
        "best_models": metrics_frame.head(10).to_dict(orient="records"),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics_frame}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run objective-retrieval hard validation.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_retrieval_hard_validation(args.project_root)
    print(f"Retrieval hard validation complete: {result['run_id']}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
