from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from trace_ace.config import load_config
from trace_ace.hard_validation import (
    _fold_masks,
    _response_transcript_matrix,
    prepare_hash_matrices,
    sparse_sgd_oof,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.retrieval_hard_validation import RETRIEVAL_DENSE_COLUMNS
from trace_ace.role_hard_validation import (
    CLOSING_FEATURES,
    ROLE_FEATURES,
    _objective_hash,
    _response_rows,
    _row_cosine,
    _unit_weighted_hstack,
    prepare_behavior_features,
    prepare_role_hashes,
    sparse_dense_sgd_oof,
)
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


FINAL_WEIGHTS = np.array([0.25, 0.25, 0.50], dtype=np.float64)


@dataclass(frozen=True)
class SemanticProtocol:
    name: str
    n_clusters: int
    cluster_seed: int
    fold_seed: int


DEFAULT_PROTOCOLS = (
    SemanticProtocol("semantic_k25_s0", 25, 20260716, 20260716),
    SemanticProtocol("semantic_k50_s0", 50, 20260716, 20260717),
    SemanticProtocol("semantic_k50_s1", 50, 20260717, 20260718),
    SemanticProtocol("semantic_k80_s0", 80, 20260716, 20260719),
)


def _unique_objective_embeddings(
    frame: pd.DataFrame,
    response_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    if response_embeddings.ndim != 2 or len(response_embeddings) != len(frame):
        raise ValueError("Objective embeddings do not align with the modeling rows.")
    if not np.isfinite(response_embeddings).all():
        raise ValueError("Objective embeddings contain non-finite values.")

    objective_rows = (
        frame.reset_index(names="response_row")
        .drop_duplicates("learning_objective_id")
        .loc[:, ["learning_objective_id", "learning_objective", "response_row"]]
        .reset_index(drop=True)
    )
    embeddings = response_embeddings[
        objective_rows["response_row"].to_numpy(dtype=np.int64)
    ].astype(np.float64, copy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)
    return objective_rows, embeddings


def build_semantic_family_folds(
    frame: pd.DataFrame,
    response_embeddings: np.ndarray,
    n_clusters: int,
    n_splits: int,
    cluster_seed: int,
    fold_seed: int,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, object]]]:
    required = {"session_id", "learning_objective_id", "learning_objective", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for semantic-family folds: {missing}")
    if frame["learning_objective_id"].isna().any():
        raise ValueError("Objective IDs cannot be missing.")

    objectives, unique_embeddings = _unique_objective_embeddings(frame, response_embeddings)
    if n_clusters < n_splits:
        raise ValueError("n_clusters must be at least n_splits.")
    if n_clusters > len(objectives):
        raise ValueError("n_clusters cannot exceed the number of objectives.")

    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=cluster_seed,
        n_init=20,
        max_iter=500,
    )
    objectives["semantic_family"] = clusterer.fit_predict(unique_embeddings).astype(np.int32)
    family_lookup = objectives.set_index("learning_objective_id")["semantic_family"]
    row_families = frame["learning_objective_id"].map(family_lookup)
    if row_families.isna().any():
        raise RuntimeError("At least one response is missing a semantic-family assignment.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=fold_seed,
    )
    folds = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(frame, frame["target"], groups=row_families)
    ):
        folds[validation_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("Incomplete semantic-family fold assignment.")

    family_to_fold = (
        pd.DataFrame({"semantic_family": row_families, "fold": folds})
        .groupby("semantic_family")["fold"]
        .first()
    )
    objectives["fold"] = objectives["semantic_family"].map(family_to_fold).astype(np.int8)

    summaries: list[dict[str, object]] = []
    row_family_array = row_families.to_numpy(dtype=np.int32)
    for fold in range(n_splits):
        train_mask, validation_mask = _fold_masks(frame, folds, fold)
        train_objectives = set(frame.loc[train_mask, "learning_objective_id"])
        validation_objectives = set(frame.loc[validation_mask, "learning_objective_id"])
        train_families = set(row_family_array[train_mask])
        validation_families = set(row_family_array[validation_mask])
        train_sessions = set(frame.loc[train_mask, "session_id"])
        validation_sessions = set(frame.loc[validation_mask, "session_id"])
        if train_objectives.intersection(validation_objectives):
            raise RuntimeError(f"Objective leakage in semantic fold {fold}.")
        if train_families.intersection(validation_families):
            raise RuntimeError(f"Semantic-family leakage in fold {fold}.")
        if train_sessions.intersection(validation_sessions):
            raise RuntimeError(f"Session leakage in semantic fold {fold}.")
        summaries.append(
            {
                "fold": fold,
                "train_rows_after_session_purge": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "validation_sessions": len(validation_sessions),
                "validation_objectives": len(validation_objectives),
                "validation_semantic_families": len(validation_families),
                "validation_positive_rate": float(frame.loc[validation_mask, "target"].mean()),
            }
        )
    return folds, objectives, summaries


def _quantile_regime(values: pd.Series, bins: int = 5) -> pd.Series:
    ranked = values.rank(method="average", pct=True)
    bucket = np.minimum((ranked * bins).astype(int), bins - 1)
    return bucket.map(lambda value: f"q{int(value) + 1}")


def build_regime_frame(frame: pd.DataFrame, retrieval: pd.DataFrame) -> pd.DataFrame:
    if list(retrieval["response_id"]) != list(frame["response_id"]):
        raise ValueError("Retrieval rows do not align with the modeling frame.")
    objective_frequency = frame["learning_objective_id"].map(
        frame["learning_objective_id"].value_counts()
    )
    session_responses = frame["session_id"].map(frame["session_id"].value_counts())
    return pd.DataFrame(
        {
            "response_id": frame["response_id"],
            "transcript_length": _quantile_regime(frame["n_utterances"]),
            "asr_uncertainty": _quantile_regime(frame["unclear_fraction"]),
            "background_rate": _quantile_regime(frame["background_fraction"]),
            "objective_frequency": _quantile_regime(objective_frequency),
            "retrieval_coverage": _quantile_regime(
                retrieval["retrieval_term_coverage"].fillna(0.0)
            ),
            "session_objectives": session_responses.map(
                lambda value: "one" if value == 1 else ("two" if value == 2 else "three_plus")
            ),
        }
    )


def _safe_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "rows": int(len(target)),
        "positive_rate": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "log_loss": float(log_loss(target, prediction, labels=[0, 1])),
        "roc_auc": float("nan"),
    }
    if len(np.unique(target)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(target, prediction))
    return metrics


def regime_scorecard(
    protocol_name: str,
    target: np.ndarray,
    prediction: np.ndarray,
    folds: np.ndarray,
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    if not (len(target) == len(prediction) == len(folds) == len(regimes)):
        raise ValueError("Scorecard inputs have different row counts.")
    rows: list[dict[str, object]] = []

    def add(regime: str, value: str, mask: np.ndarray) -> None:
        rows.append(
            {
                "protocol": protocol_name,
                "regime": regime,
                "value": value,
                **_safe_metrics(target[mask], prediction[mask]),
            }
        )

    add("overall", "all", np.ones(len(target), dtype=bool))
    for fold in sorted(np.unique(folds)):
        add("fold", str(int(fold)), folds == fold)
    for column in regimes.columns:
        if column == "response_id":
            continue
        values = regimes[column].astype(str)
        for value in sorted(values.unique()):
            add(column, value, values.eq(value).to_numpy())
    return pd.DataFrame(rows)


def _prepare_v02_blocks(
    frame: pd.DataFrame,
    cache_dir: Path,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    full_sessions, _ = prepare_hash_matrices(frame, cache_dir)
    full = _response_transcript_matrix(frame, cache_dir, full_sessions)

    session_order, session_roles = prepare_role_hashes(cache_dir)
    response_rows = _response_rows(frame, session_order)
    student = session_roles["student"][response_rows]
    tutor = session_roles["tutor"][response_rows]
    opening = session_roles["opening"][response_rows]
    closing_student = session_roles["closing_student"][response_rows]
    closing_tutor = session_roles["closing_tutor"][response_rows]
    objective_64k = _objective_hash(frame, ROLE_FEATURES)
    objective_32k = _objective_hash(frame, CLOSING_FEATURES)
    role_sparse = _unit_weighted_hstack(
        [(student, 0.75), (tutor, 0.55), (objective_64k, 1.0)]
    )
    behavior, _ = prepare_behavior_features(frame, cache_dir, session_order, response_rows)
    alignment = np.column_stack(
        [
            _row_cosine(student, objective_64k),
            _row_cosine(tutor, objective_64k),
            _row_cosine(opening, objective_64k),
            _row_cosine(closing_student, objective_32k),
            _row_cosine(closing_tutor, objective_32k),
        ]
    )
    role_dense = np.column_stack([behavior, alignment])

    semantic_variants, semantic_dense, _ = prepare_semantic_features(cache_dir, frame)
    semantic = semantic_variants["semantic_interaction"]
    return full, role_sparse, role_dense, semantic, semantic_dense


def run_robust_validation(
    project_root: str | Path,
    protocols: Iterable[SemanticProtocol] = DEFAULT_PROTOCOLS,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    objective_embeddings = np.load(paths.cache_dir / "bge_small_objective_256.npy")
    retrieval = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", *RETRIEVAL_DENSE_COLUMNS],
    )
    regimes = build_regime_frame(frame, retrieval)
    full, role_sparse, role_dense, semantic, semantic_dense = _prepare_v02_blocks(
        frame, paths.cache_dir
    )

    n_splits = int(config["n_splits"])
    target = frame["target"].to_numpy()
    metric_rows: list[dict[str, object]] = []
    scorecards: list[pd.DataFrame] = []
    prediction_outputs: list[pd.DataFrame] = []
    protocol_reports: list[dict[str, object]] = []
    objective_assignments: list[pd.DataFrame] = []

    protocol_list = list(protocols)
    for protocol_index, protocol in enumerate(protocol_list, start=1):
        print(
            f"Protocol {protocol_index}/{len(protocol_list)}: {protocol.name} "
            f"({protocol.n_clusters} semantic families)",
            flush=True,
        )
        folds, objectives, fold_summaries = build_semantic_family_folds(
            frame,
            objective_embeddings,
            n_clusters=protocol.n_clusters,
            n_splits=n_splits,
            cluster_seed=protocol.cluster_seed,
            fold_seed=protocol.fold_seed,
        )
        objectives.insert(0, "protocol", protocol.name)
        objective_assignments.append(objectives)

        full_prediction = sparse_sgd_oof(
            frame, folds, full, alpha=3e-5, n_splits=n_splits
        )
        role_prediction = sparse_dense_sgd_oof(
            frame,
            folds,
            role_sparse,
            role_dense,
            alpha=3e-5,
            n_splits=n_splits,
            dense_weight=0.2,
        )
        semantic_prediction = semantic_logistic_oof(
            frame,
            folds,
            semantic,
            semantic_dense,
            regularization_c=0.1,
            n_splits=n_splits,
        )
        components = np.column_stack(
            [full_prediction, role_prediction, semantic_prediction]
        )
        ensemble_prediction = components @ FINAL_WEIGHTS

        predictions = pd.DataFrame(
            {
                "protocol": protocol.name,
                "response_id": frame["response_id"],
                "target": target,
                "fold": folds,
                "pred_full": full_prediction,
                "pred_role": role_prediction,
                "pred_semantic": semantic_prediction,
                "pred_v02": ensemble_prediction,
            }
        )
        prediction_outputs.append(predictions)

        for model, prediction in (
            ("full_transcript", full_prediction),
            ("role_objective_dense", role_prediction),
            ("semantic_interaction_dense", semantic_prediction),
            ("v02_fixed_25_25_50", ensemble_prediction),
        ):
            metric_rows.append(
                {"protocol": protocol.name, "model": model, **binary_metrics(target, prediction)}
            )
        scorecard = regime_scorecard(
            protocol.name, target, ensemble_prediction, folds, regimes
        )
        scorecards.append(scorecard)
        overall = scorecard.loc[
            scorecard["regime"].eq("overall") & scorecard["value"].eq("all")
        ].iloc[0]
        print(
            f"{protocol.name}: v0.2 loss={overall['log_loss']:.6f}, "
            f"auc={overall['roc_auc']:.6f}",
            flush=True,
        )
        protocol_reports.append(
            {
                "name": protocol.name,
                "n_clusters": protocol.n_clusters,
                "cluster_seed": protocol.cluster_seed,
                "fold_seed": protocol.fold_seed,
                "folds": fold_summaries,
                "v02_log_loss": float(overall["log_loss"]),
                "v02_roc_auc": float(overall["roc_auc"]),
            }
        )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_robust_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(["protocol", "log_loss"])
    scorecard_frame = pd.concat(scorecards, ignore_index=True)
    predictions_frame = pd.concat(prediction_outputs, ignore_index=True)
    assignments_frame = pd.concat(objective_assignments, ignore_index=True)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    scorecard_frame.to_csv(run_dir / "regime_scorecard.csv", index=False, lineterminator="\n")
    predictions_frame.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    assignments_frame.to_parquet(run_dir / "objective_family_assignments.parquet", index=False)
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "model": "v0.2 fixed 25/25/50 ensemble",
        "protocol_type": "semantic-family-disjoint with validation-session purge",
        "protocols": protocol_reports,
        "regime_columns": [column for column in regimes.columns if column != "response_id"],
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ledger_rows = metrics_frame.loc[
        metrics_frame["model"].eq("v02_fixed_25_25_50")
    ].copy()
    ledger_rows.insert(0, "run_id", run_id)
    ledger_rows.insert(1, "timestamp_utc", timestamp.isoformat())
    ledger_rows["model"] = "v02_fixed_25_25_50__" + ledger_rows["protocol"]
    ledger_rows["fold_scheme"] = "semantic-family-disjoint + session purge"
    ledger_rows["status"] = "completed"
    ledger_columns = [
        "run_id",
        "timestamp_utc",
        "model",
        "fold_scheme",
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "status",
    ]
    ledger_rows[ledger_columns].to_csv(
        paths.experiments_dir / "experiment_ledger.csv",
        mode="a",
        header=False,
        index=False,
        lineterminator="\n",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics_frame,
        "scorecard": scorecard_frame,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated semantic-family stress validation for the v0.2 ensemble."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one 50-family protocol as an implementation smoke test.",
    )
    parser.add_argument(
        "--protocol-spec",
        action="append",
        help=(
            "Custom frozen protocol as name:n_clusters:cluster_seed:fold_seed; "
            "may be repeated."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.protocol_spec:
        custom: list[SemanticProtocol] = []
        for specification in args.protocol_spec:
            parts = specification.split(":")
            if len(parts) != 4:
                raise ValueError(
                    "Protocol specs require name:n_clusters:cluster_seed:fold_seed."
                )
            name, n_clusters, cluster_seed, fold_seed = parts
            custom.append(
                SemanticProtocol(
                    name,
                    int(n_clusters),
                    int(cluster_seed),
                    int(fold_seed),
                )
            )
        protocols = tuple(custom)
    else:
        protocols = (DEFAULT_PROTOCOLS[1],) if args.quick else DEFAULT_PROTOCOLS
    result = run_robust_validation(args.project_root, protocols=protocols)
    print(f"Robust validation complete: {result['run_id']}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
