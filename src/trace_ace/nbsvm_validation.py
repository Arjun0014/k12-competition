from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.feedback_hash_validation import (
    HASH_SCHEMA_TAG,
    VIEW_LINE_RE,
    WORD_FEATURES,
    _word_hasher,
    prepare_feedback_hashes,
)
from trace_ace.hard_validation import (
    _fold_masks,
    _response_transcript_matrix,
    prepare_hash_matrices,
)
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.multiview_validation import BASELINE_RUN_ID, PILOT_PROTOCOL
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
from trace_ace.role_hard_validation import _unit_weighted_hstack


DEFAULT_C_VALUES = (0.1, 0.3, 1.0)
DEFAULT_VARIANTS = (
    "full_word",
    "feedback_word_ordered",
    "feedback_word_char_ordered",
    "feedback_rolepref_word_ordered",
)
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
PREFIX_TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def role_prefixed_feedback_text(view_text: str) -> str:
    rendered: list[str] = []
    found_student = False
    found_feedback = False
    prefix_by_label = {
        "objective": "learning_objective",
        "answer": "student_speech",
        "feedback": "tutor_feedback",
    }
    for raw_line in str(view_text).splitlines():
        match = VIEW_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        label = match.group(1).lower()
        prefix = prefix_by_label.get(label)
        if prefix is None:
            continue
        tokens = PREFIX_TOKEN_RE.findall(match.group(2).lower())
        rendered.extend(f"{prefix}_{token}" for token in tokens)
        if label == "answer" and tokens:
            found_student = True
        elif label == "feedback" and tokens:
            found_feedback = True
    if not found_student:
        rendered.append("student_speech_missing")
    if not found_feedback:
        rendered.append("tutor_feedback_missing")
    return " ".join(rendered)


def prepare_role_prefixed_feedback_hash(
    cache_dir: Path, response_ids: pd.Series
) -> sparse.csr_matrix:
    texts = pd.read_parquet(
        cache_dir / "response_multiview_texts.parquet",
        columns=["response_id", "feedback_evidence"],
    )
    if list(texts["response_id"]) != list(response_ids):
        raise ValueError("Role-prefixed feedback rows do not align with modeling data.")
    path = cache_dir / (
        f"hash_feedback_rolepref_word_{WORD_FEATURES}_{HASH_SCHEMA_TAG}.npz"
    )
    if path.exists():
        matrix = sparse.load_npz(path).tocsr()
    else:
        print("Hashing token-level role-prefixed feedback word n-grams...", flush=True)
        rendered = texts["feedback_evidence"].fillna("").map(
            role_prefixed_feedback_text
        )
        matrix = _word_hasher().transform(rendered).tocsr()
        sparse.save_npz(path, matrix, compressed=True)
    expected = (len(response_ids), WORD_FEATURES)
    if matrix.shape != expected or not np.isfinite(matrix.data).all():
        raise ValueError("Role-prefixed feedback hash violates its feature contract.")
    return matrix


def nb_log_count_ratio(matrix: sparse.csr_matrix, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.int8)
    if matrix.shape[0] != len(target) or set(np.unique(target)) != {0, 1}:
        raise ValueError("NB-SVM ratio requires aligned binary targets.")
    positive = np.asarray(matrix[target == 1].sum(axis=0)).ravel()
    negative = np.asarray(matrix[target == 0].sum(axis=0)).ravel()
    positive = (positive + 1.0) / (float(np.sum(target == 1)) + 1.0)
    negative = (negative + 1.0) / (float(np.sum(target == 0)) + 1.0)
    ratio = np.log(positive / negative).astype(np.float32)
    if ratio.shape != (matrix.shape[1],) or not np.isfinite(ratio).all():
        raise RuntimeError("NB-SVM log-count ratio failed its feature contract.")
    return ratio


def nbsvm_oof(
    frame: pd.DataFrame,
    folds: np.ndarray,
    matrix: sparse.csr_matrix,
    dense_values: np.ndarray | None,
    regularization_c: float,
    dense_weight: float = 0.12,
) -> np.ndarray:
    target = frame["target"].to_numpy(dtype=np.int8)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train_mask, validation_mask = _fold_masks(frame, folds, int(fold))
        ratio = nb_log_count_ratio(matrix[train_mask], target[train_mask])
        x_train = matrix[train_mask].multiply(ratio).tocsr()
        x_validation = matrix[validation_mask].multiply(ratio).tocsr()
        if dense_values is not None:
            scaler = StandardScaler()
            dense_train = scaler.fit_transform(dense_values[train_mask]).astype(np.float32)
            dense_validation = scaler.transform(dense_values[validation_mask]).astype(np.float32)
            x_train = sparse.hstack(
                [x_train, sparse.csr_matrix(dense_train * np.float32(dense_weight))],
                format="csr",
            )
            x_validation = sparse.hstack(
                [
                    x_validation,
                    sparse.csr_matrix(dense_validation * np.float32(dense_weight)),
                ],
                format="csr",
            )
        model = LogisticRegression(
            C=float(regularization_c),
            penalty="l2",
            solver="liblinear",
            dual=True,
            max_iter=400,
            tol=1e-5,
            random_state=20260717 + int(fold),
        )
        model.fit(x_train, target[train_mask])
        prediction[validation_mask] = model.predict_proba(x_validation)[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("NB-SVM produced incomplete OOF predictions.")
    return prediction


def run_nbsvm_validation(
    project_root: str | Path,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy(dtype=np.float64)
    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    )
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline rows match the requested NB-SVM protocols.")

    variant_list = list(variants)
    full_session, _ = prepare_hash_matrices(frame, paths.cache_dir)
    full_word = _response_transcript_matrix(frame, paths.cache_dir, full_session)
    feedback_word, feedback_char, _ = prepare_feedback_hashes(
        paths.cache_dir, frame["response_id"]
    )
    needs_role_prefixed = "feedback_rolepref_word_ordered" in variant_list
    feedback_role_prefixed = (
        prepare_role_prefixed_feedback_hash(paths.cache_dir, frame["response_id"])
        if needs_role_prefixed
        else None
    )
    ordered = pd.read_parquet(paths.cache_dir / "response_ordered_features.parquet")
    if list(ordered["response_id"]) != list(frame["response_id"]):
        raise ValueError("Ordered feature rows do not align for NB-SVM.")
    ordered_values = ordered[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    matrices = {
        "full_word": (full_word, None),
        "feedback_word_ordered": (feedback_word, ordered_values),
        "feedback_word_char_ordered": (
            _unit_weighted_hstack(
                [(feedback_word, 1.0), (feedback_char, 1.0)]
            ),
            ordered_values,
        ),
    }
    if feedback_role_prefixed is not None:
        matrices["feedback_rolepref_word_ordered"] = (
            feedback_role_prefixed,
            ordered_values,
        )
    unknown = sorted(set(variant_list).difference(matrices))
    if unknown:
        raise ValueError(f"Unknown NB-SVM variants: {unknown}")

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))
    c_list = list(c_values)
    for variant in variant_list:
        matrix, dense = matrices[variant]
        for c_value in c_list:
            model_name = f"nbsvm_{variant}_c{c_value:g}"
            for protocol in protocol_names:
                base = baseline.loc[baseline["protocol"].eq(protocol)].reset_index(drop=True)
                if list(base["response_id"]) != list(frame["response_id"]):
                    raise ValueError(f"NB-SVM baseline order mismatch for {protocol}.")
                folds = base["fold"].to_numpy(dtype=np.int8)
                print(f"{model_name} on {protocol}", flush=True)
                component = nbsvm_oof(
                    frame, folds, matrix, dense, regularization_c=float(c_value)
                )
                pred_v02 = base["pred_v02"].to_numpy(dtype=np.float64)
                baseline_metrics = binary_metrics(target, pred_v02)
                candidates = [("standalone", 1.0, component)]
                candidates.extend(
                    (
                        "blend",
                        weight,
                        (1.0 - weight) * pred_v02 + weight * component,
                    )
                    for weight in BLEND_WEIGHTS
                )
                for prediction_type, weight, prediction in candidates:
                    metrics = binary_metrics(target, prediction)
                    metric_rows.append(
                        {
                            "protocol": protocol,
                            "model": model_name,
                            "variant": variant,
                            "prediction_type": prediction_type,
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
                        fm = binary_metrics(target[mask], prediction[mask])
                        fb = binary_metrics(target[mask], pred_v02[mask])
                        fold_rows.append(
                            {
                                "protocol": protocol,
                                "fold": int(fold),
                                "model": model_name,
                                "prediction_type": prediction_type,
                                "new_model_weight": weight,
                                **fm,
                                "delta_log_loss_vs_v02": fm["log_loss"]
                                - fb["log_loss"],
                                "delta_roc_auc_vs_v02": fm["roc_auc"]
                                - fb["roc_auc"],
                            }
                        )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "protocol": protocol,
                            "response_id": frame["response_id"],
                            "target": target,
                            "fold": folds,
                            "model": model_name,
                            "pred_v02": pred_v02,
                            "pred_nbsvm": component,
                        }
                    )
                )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_nbsvm_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at_utc": timestamp.isoformat(),
                "baseline_run_id": baseline_run_id,
                "protocols": protocol_names,
                "variants": variant_list,
                "regularization_c": c_list,
                "blend_weights": list(BLEND_WEIGHTS),
                "selection_policy": (
                    "Select one variant, C, and blend weight only on the frozen pilot; "
                    "then lock all three before untouched confirmation protocols."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate fold-local NB-SVM text models.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS)
    parser.add_argument("--c", action="append", type=float)
    parser.add_argument("--protocol", action="append")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_nbsvm_validation(
        args.project_root,
        variants=args.variant or DEFAULT_VARIANTS,
        c_values=args.c or DEFAULT_C_VALUES,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
    )
    print(f"NB-SVM validation complete: {result['run_id']}")
    print(result["metrics"].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
