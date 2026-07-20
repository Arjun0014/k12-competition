from __future__ import annotations

import argparse
import gc
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.multiview_validation import BASELINE_RUN_ID, BLEND_WEIGHTS, PILOT_PROTOCOL
from trace_ace.objective_retrieval import _objective_terms, _tokens
from trace_ace.ordered_features import (
    AFFIRM_RE,
    CORRECTION_RE,
    ORDERED_FEATURE_NAMES,
    REASONING_RE,
    UNCERTAINTY_RE,
)
from trace_ace.robust_validation import _safe_metrics
from trace_ace.role_hard_validation import (
    _unit_weighted_hstack,
    sparse_dense_sgd_oof,
)


WORD_FEATURES = 2**17
CHAR_FEATURES = 2**17
EVENT_FEATURES = 2**15
HASH_SCHEMA_TAG = "v2_budgeted"
DEFAULT_ALPHAS = (3e-5, 1e-4, 3e-4)
DEFAULT_LOGISTIC_C = (0.1, 0.3, 1.0)
DEFAULT_VARIANTS = (
    "feedback_word",
    "feedback_char",
    "feedback_word_char",
    "feedback_word_ordered",
    "feedback_word_char_ordered",
    "feedback_event",
    "feedback_event_ordered",
    "feedback_word_event_ordered",
    "trajectory_word_ordered",
    "trajectory_char_ordered",
    "trajectory_word_char_ordered",
)

VIEW_LINE_RE = re.compile(r"^\[([A-Z_]+)\]\s*(.*)$")


def _word_hasher() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=WORD_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def _char_hasher() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=CHAR_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def _event_hasher() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 3),
        n_features=EVENT_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )


def feedback_event_text(view_text: str) -> str:
    parsed: list[tuple[str, str]] = []
    objective = ""
    for raw_line in str(view_text).splitlines():
        match = VIEW_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        label, content = match.group(1).lower(), match.group(2).strip()
        if label == "objective":
            objective = content
        else:
            parsed.append((label, content))
    objective_terms, _ = _objective_terms(objective)
    objective_set = set(objective_terms)
    pairs: list[tuple[str, str]] = []
    for left, right in zip(parsed, parsed[1:]):
        if left[0] == "answer" and right[0] == "feedback":
            pairs.append((left[1], right[1]))
    rendered: list[str] = []
    for index, (answer, feedback) in enumerate(pairs):
        answer_tokens = _tokens(answer)
        overlap = len(objective_set.intersection(answer_tokens)) / max(
            1, len(objective_set)
        )
        position = "early" if index < len(pairs) / 3 else (
            "late" if index >= 2 * len(pairs) / 3 else "middle"
        )
        objective_level = "none" if overlap == 0 else ("high" if overlap >= 0.5 else "some")
        answer_flags = [
            "reasoning" if REASONING_RE.search(answer) else "no_reasoning",
            "uncertain" if UNCERTAINTY_RE.search(answer) else "certain",
            "answer_question" if "?" in answer else "answer_statement",
            "answer_numeric" if any(token.isdigit() for token in answer_tokens) else "answer_text",
            "answer_short" if len(answer.split()) <= 5 else "answer_long",
        ]
        feedback_flags = [
            "affirm" if AFFIRM_RE.search(feedback) else "no_affirm",
            "corrective" if CORRECTION_RE.search(feedback) else "no_corrective",
            "feedback_question" if "?" in feedback else "feedback_statement",
            "feedback_reasoning" if REASONING_RE.search(feedback) else "no_feedback_reasoning",
        ]
        combined = [
            f"pair_obj_{objective_level}_{feedback_flags[0]}",
            f"pair_{answer_flags[0]}_{feedback_flags[0]}",
            f"pair_{answer_flags[1]}_{feedback_flags[1]}",
            f"pair_{position}_{feedback_flags[0]}",
            f"pair_{position}_{feedback_flags[1]}",
        ]
        rendered.append(
            " ".join(
                [
                    f"pair_position_{position}",
                    f"objective_{objective_level}",
                    *answer_flags,
                    *feedback_flags,
                    *combined,
                ]
            )
        )
    return "\n".join(rendered) if rendered else "no_answer_feedback_pair"


def prepare_feedback_hashes(
    cache_dir: Path, response_ids: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    texts = pd.read_parquet(
        cache_dir / "response_multiview_texts.parquet",
        columns=["response_id", "feedback_evidence"],
    )
    if list(texts["response_id"]) != list(response_ids):
        raise ValueError("Feedback text rows do not align with modeling data.")
    word_path = cache_dir / f"hash_feedback_word_{WORD_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    char_path = cache_dir / f"hash_feedback_char_{CHAR_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    event_path = cache_dir / f"hash_feedback_event_{EVENT_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    if word_path.exists():
        word = sparse.load_npz(word_path).tocsr()
    else:
        print("Hashing ordered feedback word n-grams...", flush=True)
        word = _word_hasher().transform(texts["feedback_evidence"].fillna("")).tocsr()
        sparse.save_npz(word_path, word, compressed=True)
    if char_path.exists():
        char = sparse.load_npz(char_path).tocsr()
    else:
        print("Hashing ordered feedback character n-grams...", flush=True)
        char = _char_hasher().transform(texts["feedback_evidence"].fillna("")).tocsr()
        sparse.save_npz(char_path, char, compressed=True)
    if event_path.exists():
        event = sparse.load_npz(event_path).tocsr()
    else:
        print("Hashing symbolic answer-feedback event sequences...", flush=True)
        event_texts = texts["feedback_evidence"].fillna("").map(feedback_event_text)
        event = _event_hasher().transform(event_texts).tocsr()
        sparse.save_npz(event_path, event, compressed=True)
    expected_shape = (len(response_ids), WORD_FEATURES)
    if (
        word.shape != expected_shape
        or char.shape != expected_shape
        or event.shape != (len(response_ids), EVENT_FEATURES)
    ):
        raise ValueError("Feedback hash cache shape does not match modeling data.")
    return word, char, event


def prepare_trajectory_hashes(
    cache_dir: Path, response_ids: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    texts = pd.read_parquet(
        cache_dir / "response_multiview_texts.parquet",
        columns=["response_id", "session_trajectory"],
    )
    if list(texts["response_id"]) != list(response_ids):
        raise ValueError("Trajectory text rows do not align with modeling data.")
    word_path = cache_dir / f"hash_trajectory_word_{WORD_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    char_path = cache_dir / f"hash_trajectory_char_{CHAR_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    if word_path.exists():
        word = sparse.load_npz(word_path).tocsr()
    else:
        print("Hashing compact trajectory word n-grams...", flush=True)
        word = _word_hasher().transform(texts["session_trajectory"].fillna("")).tocsr()
        sparse.save_npz(word_path, word, compressed=True)
    if char_path.exists():
        char = sparse.load_npz(char_path).tocsr()
    else:
        print("Hashing compact trajectory character n-grams...", flush=True)
        char = _char_hasher().transform(texts["session_trajectory"].fillna("")).tocsr()
        sparse.save_npz(char_path, char, compressed=True)
    expected = (len(response_ids), WORD_FEATURES)
    if word.shape != expected or char.shape != expected:
        raise ValueError("Trajectory hash cache shape does not match modeling data.")
    return word, char


def _load_baseline(
    experiments_dir: Path, protocols: set[str] | None, baseline_run_id: str
) -> pd.DataFrame:
    baseline = pd.read_parquet(
        experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    )
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline rows match the requested feedback protocols.")
    return baseline.reset_index(drop=True)


def sparse_dense_logistic_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    matrix: sparse.csr_matrix,
    dense_values: np.ndarray | None,
    regularization_c: float,
    dense_weight: float = 0.12,
) -> np.ndarray:
    target = frame["target"].to_numpy()
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train_mask, validation_mask = _fold_masks(frame, folds, int(fold))
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
                [x_validation, sparse.csr_matrix(dense_validation * dense_weight)],
                format="csr",
            )
        model = LogisticRegression(
            C=regularization_c,
            solver="liblinear",
            max_iter=300,
            tol=1e-5,
            random_state=20260717 + int(fold),
        )
        model.fit(x_train, target[train_mask])
        prediction[validation_mask] = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete feedback logistic predictions.")
    return prediction


def run_feedback_hash_validation(
    project_root: str | Path,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
    logistic_c_values: Iterable[float] = (),
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy()
    variant_list = list(variants)
    word, char, event = prepare_feedback_hashes(paths.cache_dir, frame["response_id"])
    needs_trajectory = any(str(name).startswith("trajectory_") for name in variant_list)
    trajectory_word: sparse.csr_matrix | None = None
    trajectory_char: sparse.csr_matrix | None = None
    if needs_trajectory:
        trajectory_word, trajectory_char = prepare_trajectory_hashes(
            paths.cache_dir, frame["response_id"]
        )
    ordered = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    if list(ordered["response_id"]) != list(frame["response_id"]):
        raise ValueError("Ordered feature rows do not align with modeling data.")
    ordered_values = ordered[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    baseline = _load_baseline(paths.experiments_dir, protocols, baseline_run_id)
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))

    matrices: dict[str, tuple[sparse.csr_matrix, np.ndarray | None]] = {
        "feedback_word": (word, None),
        "feedback_char": (char, None),
        "feedback_word_char": (
            _unit_weighted_hstack([(word, 1.0), (char, 1.0)]),
            None,
        ),
        "feedback_word_ordered": (word, ordered_values),
        "feedback_word_char_ordered": (
            _unit_weighted_hstack([(word, 1.0), (char, 1.0)]),
            ordered_values,
        ),
        "feedback_event": (event, None),
        "feedback_event_ordered": (event, ordered_values),
        "feedback_word_event_ordered": (
            _unit_weighted_hstack([(word, 1.0), (event, 0.7)]),
            ordered_values,
        ),
    }
    if trajectory_word is not None and trajectory_char is not None:
        matrices.update(
            {
                "trajectory_word_ordered": (trajectory_word, ordered_values),
                "trajectory_char_ordered": (trajectory_char, ordered_values),
                "trajectory_word_char_ordered": (
                    _unit_weighted_hstack(
                        [(trajectory_word, 1.0), (trajectory_char, 1.0)]
                    ),
                    ordered_values,
                ),
            }
        )
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    alpha_list = list(alphas)
    logistic_c_list = list(logistic_c_values)
    for variant_name in variant_list:
        matrix, dense = matrices[variant_name]
        model_specs = [
            (f"{variant_name}_a{alpha:g}", "sgd", alpha) for alpha in alpha_list
        ] + [
            (f"{variant_name}_logistic_c{c_value:g}", "logistic", c_value)
            for c_value in logistic_c_list
        ]
        for model_name, classifier, parameter in model_specs:
            for protocol_name in protocol_names:
                protocol_base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
                if list(protocol_base["response_id"]) != list(frame["response_id"]):
                    raise ValueError(f"Baseline order mismatch for {protocol_name}.")
                folds = protocol_base["fold"].to_numpy(dtype=np.int8)
                if classifier == "sgd":
                    prediction = sparse_dense_sgd_oof(
                        frame,
                        folds,
                        matrix,
                        dense,
                        alpha=parameter,
                        n_splits=int(np.max(folds)) + 1,
                        dense_weight=0.12,
                    )
                else:
                    prediction = sparse_dense_logistic_oof(
                        frame,
                        folds,
                        matrix,
                        dense,
                        regularization_c=parameter,
                        dense_weight=0.12,
                    )
                baseline_prediction = protocol_base["pred_v02"].to_numpy(dtype=np.float64)
                baseline_metrics = binary_metrics(target, baseline_prediction)
                standalone_metrics = binary_metrics(target, prediction)
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": model_name,
                        "prediction_type": "standalone",
                        "new_model_weight": 1.0,
                        **standalone_metrics,
                        "delta_log_loss_vs_v02": standalone_metrics["log_loss"]
                        - baseline_metrics["log_loss"],
                        "delta_roc_auc_vs_v02": standalone_metrics["roc_auc"]
                        - baseline_metrics["roc_auc"],
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "protocol": protocol_name,
                            "response_id": frame["response_id"],
                            "target": target,
                            "fold": folds,
                            "model": model_name,
                            "pred_v02": baseline_prediction,
                            "pred_feedback_hash": prediction,
                        }
                    )
                )
                for weight in BLEND_WEIGHTS:
                    blended = (1.0 - weight) * baseline_prediction + weight * prediction
                    metrics = binary_metrics(target, blended)
                    blend_name = f"{model_name}__blend_w{weight:.2f}"
                    metric_rows.append(
                        {
                            "protocol": protocol_name,
                            "model": blend_name,
                            "prediction_type": "blend",
                            "new_model_weight": weight,
                            **metrics,
                            "delta_log_loss_vs_v02": metrics["log_loss"]
                            - baseline_metrics["log_loss"],
                            "delta_roc_auc_vs_v02": metrics["roc_auc"]
                            - baseline_metrics["roc_auc"],
                        }
                    )
                    for fold in sorted(np.unique(folds)):
                        mask = folds == fold
                        fold_metrics = _safe_metrics(target[mask], blended[mask])
                        fold_baseline = _safe_metrics(
                            target[mask], baseline_prediction[mask]
                        )
                        fold_rows.append(
                            {
                                "protocol": protocol_name,
                                "model": blend_name,
                                "fold": int(fold),
                                **fold_metrics,
                                "delta_log_loss_vs_v02": float(fold_metrics["log_loss"])
                                - float(fold_baseline["log_loss"]),
                                "delta_roc_auc_vs_v02": float(fold_metrics["roc_auc"])
                                - float(fold_baseline["roc_auc"]),
                            }
                        )
                print(
                    f"{protocol_name} {model_name}: loss={standalone_metrics['log_loss']:.6f}, "
                    f"auc={standalone_metrics['roc_auc']:.6f}",
                    flush=True,
                )
        gc.collect()

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_feedback_hash_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "baseline_run_id": baseline_run_id,
        "protocols": protocol_names,
        "variants": variant_list,
        "alphas": alpha_list,
        "logistic_c_values": logistic_c_list,
        "blend_weights": list(BLEND_WEIGHTS),
        "word_features": WORD_FEATURES,
        "char_features": CHAR_FEATURES,
        "selection_policy": "Pilot first; lock before untouched protocol confirmation.",
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = metrics_frame.copy()
    ledger.insert(0, "run_id", run_id)
    ledger.insert(1, "timestamp_utc", timestamp.isoformat())
    ledger["model"] = ledger["model"] + "__" + ledger["protocol"]
    ledger["fold_scheme"] = "semantic-family-disjoint + session purge"
    ledger["status"] = "completed"
    ledger[
        [
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
    ].to_csv(
        paths.experiments_dir / "experiment_ledger.csv",
        mode="a",
        header=False,
        index=False,
        lineterminator="\n",
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics_frame}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate ordered answer-feedback hashing on frozen hard folds."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS)
    parser.add_argument("--alpha", action="append", type=float)
    parser.add_argument("--logistic-c", action="append", type=float)
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    selected_alphas = args.alpha or (() if args.logistic_c else DEFAULT_ALPHAS)
    result = run_feedback_hash_validation(
        args.project_root,
        variants=args.variant or DEFAULT_VARIANTS,
        alphas=selected_alphas,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
        logistic_c_values=args.logistic_c or (),
    )
    print(f"Feedback hash validation complete: {result['run_id']}")
    print(result["metrics"].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
