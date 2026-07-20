from __future__ import annotations

import argparse
import gc
import json
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.ordered_features import ORDERED_FEATURE_NAMES
from trace_ace.robust_validation import _safe_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
ALL_VARIANTS = (
    "semantic_reference",
    "feedback_geometry",
    "context_geometry",
    "student_interaction",
    "feedback_interaction",
    "trajectory_geometry",
    "trajectory_interaction",
    "context_trajectory_interaction",
    "context_feedback_interaction",
    "evidence_mean_interaction",
    "multiview_interaction",
)
DEFAULT_VARIANTS = tuple(name for name in ALL_VARIANTS if name != "semantic_reference")
DEFAULT_C_VALUES = (0.01, 0.03, 0.1)
BLEND_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50)
VARIANT_VIEWS = {
    "semantic_reference": (),
    "feedback_geometry": ("feedback",),
    "context_geometry": ("student", "tutor", "feedback", "trajectory"),
    "student_interaction": ("student",),
    "feedback_interaction": ("feedback",),
    "trajectory_geometry": ("trajectory",),
    "trajectory_interaction": ("trajectory",),
    "context_trajectory_interaction": ("trajectory",),
    "context_feedback_interaction": ("feedback",),
    "evidence_mean_interaction": ("student", "tutor", "feedback", "trajectory"),
    "multiview_interaction": ("student", "tutor", "feedback", "trajectory"),
}


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-8))


def _normalized_mean(*values: np.ndarray) -> np.ndarray:
    if not values:
        raise ValueError("At least one embedding block is required.")
    return _row_normalize(np.mean(np.stack(values, axis=0), axis=0))


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1, keepdims=True).astype(np.float32)


def load_embedding_blocks(
    cache_dir: Path, rows: int, view_names: Iterable[str]
) -> dict[str, np.ndarray]:
    all_filenames = {
        "context": "bge_small_context_256.npy",
        "objective": "bge_small_objective_256.npy",
        "student": "bge_small_student_evidence_256.npy",
        "tutor": "bge_small_tutor_evidence_256.npy",
        "feedback": "bge_small_feedback_evidence_256.npy",
        "trajectory": "bge_small_session_trajectory_256.npy",
    }
    requested = ["context", "objective", *list(view_names)]
    filenames = {name: all_filenames[name] for name in dict.fromkeys(requested)}
    blocks: dict[str, np.ndarray] = {}
    expected_shape = (rows, 384)
    for name, filename in filenames.items():
        values = np.load(cache_dir / filename).astype(np.float32)
        if values.shape != expected_shape:
            raise ValueError(f"{name} cache has shape {values.shape}; expected {expected_shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} cache contains non-finite values.")
        blocks[name] = _row_normalize(values)
    return blocks


def build_multiview_dense(
    base_dense: np.ndarray,
    blocks: dict[str, np.ndarray],
    view_names: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    context = blocks["context"]
    objective = blocks["objective"]
    selected = list(view_names)
    unknown = sorted(set(selected).difference({"student", "tutor", "feedback", "trajectory"}))
    if unknown:
        raise ValueError(f"Unknown multiview dense blocks: {unknown}")
    similarity_blocks: list[np.ndarray] = []
    names: list[str] = []
    for name in selected:
        similarity_blocks.extend(
            [_row_cosine(objective, blocks[name]), _row_cosine(context, blocks[name])]
        )
        names.extend([f"cos_objective_{name}", f"cos_context_{name}"])
    for left, right in combinations(selected, 2):
        similarity_blocks.append(_row_cosine(blocks[left], blocks[right]))
        names.append(f"cos_{left}_{right}")
    if not similarity_blocks:
        return base_dense, names
    similarities = np.column_stack(similarity_blocks).astype(np.float64)
    return np.column_stack([base_dense, similarities]), names


def build_variant(name: str, blocks: dict[str, np.ndarray]) -> np.ndarray:
    objective = blocks["objective"]
    if name in {
        "semantic_reference",
        "feedback_geometry",
        "trajectory_geometry",
        "context_geometry",
    }:
        context = blocks["context"]
        return np.column_stack(
            [
                context * np.float32(0.7),
                objective * np.float32(0.7),
                context * objective * np.float32(6.0),
                np.abs(context - objective) * np.float32(0.7),
            ]
        )
    if name == "student_interaction":
        evidence = blocks["student"]
        return np.column_stack(
            [
                evidence * np.float32(0.7),
                objective * np.float32(0.7),
                evidence * objective * np.float32(6.0),
                np.abs(evidence - objective) * np.float32(0.7),
            ]
        )
    if name == "feedback_interaction":
        evidence = blocks["feedback"]
        return np.column_stack(
            [
                evidence * np.float32(0.7),
                objective * np.float32(0.7),
                evidence * objective * np.float32(6.0),
                np.abs(evidence - objective) * np.float32(0.7),
            ]
        )
    if name == "trajectory_interaction":
        evidence = blocks["trajectory"]
        return np.column_stack(
            [
                evidence * np.float32(0.7),
                objective * np.float32(0.7),
                evidence * objective * np.float32(6.0),
                np.abs(evidence - objective) * np.float32(0.7),
            ]
        )
    if name == "context_trajectory_interaction":
        context = blocks["context"]
        trajectory = blocks["trajectory"]
        return np.column_stack(
            [
                context * np.float32(0.7),
                objective * np.float32(0.7),
                context * objective * np.float32(6.0),
                np.abs(context - objective) * np.float32(0.7),
                trajectory * np.float32(0.55),
                trajectory * objective * np.float32(4.0),
                np.abs(trajectory - objective) * np.float32(0.5),
            ]
        )
    if name == "context_feedback_interaction":
        context = blocks["context"]
        feedback = blocks["feedback"]
        return np.column_stack(
            [
                context * np.float32(0.7),
                objective * np.float32(0.7),
                context * objective * np.float32(6.0),
                np.abs(context - objective) * np.float32(0.7),
                feedback * np.float32(0.55),
                feedback * objective * np.float32(4.0),
                np.abs(feedback - objective) * np.float32(0.5),
            ]
        )
    if name == "evidence_mean_interaction":
        evidence = _normalized_mean(
            blocks["student"], blocks["tutor"], blocks["feedback"]
        )
        return np.column_stack(
            [
                evidence * np.float32(0.7),
                objective * np.float32(0.7),
                evidence * objective * np.float32(6.0),
                np.abs(evidence - objective) * np.float32(0.7),
                blocks["trajectory"] * np.float32(0.35),
            ]
        )
    if name == "multiview_interaction":
        context = blocks["context"]
        student = blocks["student"]
        tutor = blocks["tutor"]
        feedback = blocks["feedback"]
        trajectory = blocks["trajectory"]
        return np.column_stack(
            [
                objective * np.float32(0.7),
                context * np.float32(0.7),
                student * np.float32(0.45),
                tutor * np.float32(0.45),
                feedback * np.float32(0.65),
                trajectory * np.float32(0.30),
                context * objective * np.float32(6.0),
                np.abs(context - objective) * np.float32(0.7),
                student * objective * np.float32(3.0),
                tutor * objective * np.float32(3.0),
                feedback * objective * np.float32(4.0),
                trajectory * objective * np.float32(2.0),
            ]
        )
    raise ValueError(f"Unknown multiview variant: {name}")


def _load_baseline_predictions(
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


def load_ordered_features(cache_dir: Path, frame: pd.DataFrame) -> np.ndarray:
    ordered = pd.read_parquet(cache_dir / "response_ordered_features.parquet")
    if list(ordered["response_id"]) != list(frame["response_id"]):
        raise ValueError("Ordered feature rows do not align with modeling data.")
    values = ordered[ORDERED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Ordered feature cache contains non-finite values.")
    return values


def run_multiview_validation(
    project_root: str | Path,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
    include_ordered: bool = False,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(drop=True)
    target = frame["target"].to_numpy()
    baseline = _load_baseline_predictions(
        paths.experiments_dir, baseline_run_id, protocols
    )
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))
    variant_list = list(variants)
    required_views = list(
        dict.fromkeys(
            view
            for variant_name in variant_list
            for view in VARIANT_VIEWS[variant_name]
        )
    )
    blocks = load_embedding_blocks(paths.cache_dir, len(frame), required_views)
    semantic_variants, base_dense, base_dense_names = prepare_semantic_features(
        paths.cache_dir, frame
    )
    del semantic_variants
    gc.collect()
    ordered_values = (
        load_ordered_features(paths.cache_dir, frame) if include_ordered else None
    )

    metric_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    dense_names_by_variant: dict[str, list[str]] = {}
    c_list = list(c_values)
    for variant_index, variant_name in enumerate(variant_list, start=1):
        print(
            f"Building variant {variant_index}/{len(variant_list)}: {variant_name}",
            flush=True,
        )
        features = build_variant(variant_name, blocks)
        dense, geometry_names = build_multiview_dense(
            base_dense, blocks, VARIANT_VIEWS[variant_name]
        )
        dense_names_by_variant[variant_name] = [
            *base_dense_names,
            *geometry_names,
            *(ORDERED_FEATURE_NAMES if include_ordered else []),
        ]
        if ordered_values is not None:
            dense = np.column_stack([dense, ordered_values])
        print(f"{variant_name} feature shape: {features.shape}", flush=True)
        for c_value in c_list:
            model_name = f"{variant_name}_c{c_value:g}"
            if include_ordered:
                model_name += "__ordered"
            for protocol_name in protocol_names:
                protocol_base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
                if list(protocol_base["response_id"]) != list(frame["response_id"]):
                    raise ValueError(f"Baseline order mismatch for {protocol_name}.")
                folds = protocol_base["fold"].to_numpy(dtype=np.int8)
                prediction = semantic_logistic_oof(
                    frame,
                    folds,
                    features,
                    dense,
                    regularization_c=c_value,
                    n_splits=int(np.max(folds)) + 1,
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
                        "delta_log_loss_vs_v02": (
                            standalone_metrics["log_loss"] - baseline_metrics["log_loss"]
                        ),
                        "delta_roc_auc_vs_v02": (
                            standalone_metrics["roc_auc"] - baseline_metrics["roc_auc"]
                        ),
                    }
                )
                output = pd.DataFrame(
                    {
                        "protocol": protocol_name,
                        "response_id": frame["response_id"],
                        "target": target,
                        "fold": folds,
                        "model": model_name,
                        "pred_v02": baseline_prediction,
                        "pred_multiview": prediction,
                    }
                )
                prediction_frames.append(output)
                for weight in BLEND_WEIGHTS:
                    blended = baseline_prediction * (1.0 - weight) + prediction * weight
                    blend_name = f"{model_name}__blend_w{weight:.2f}"
                    blend_metrics = binary_metrics(target, blended)
                    metric_rows.append(
                        {
                            "protocol": protocol_name,
                            "model": blend_name,
                            "prediction_type": "blend",
                            "new_model_weight": weight,
                            **blend_metrics,
                            "delta_log_loss_vs_v02": (
                                blend_metrics["log_loss"] - baseline_metrics["log_loss"]
                            ),
                            "delta_roc_auc_vs_v02": (
                                blend_metrics["roc_auc"] - baseline_metrics["roc_auc"]
                            ),
                        }
                    )
                    for fold in sorted(np.unique(folds)):
                        fold_mask = folds == fold
                        fold_metrics = _safe_metrics(
                            target[fold_mask], blended[fold_mask]
                        )
                        fold_baseline_metrics = _safe_metrics(
                            target[fold_mask], baseline_prediction[fold_mask]
                        )
                        fold_metric_rows.append(
                            {
                                "protocol": protocol_name,
                                "model": blend_name,
                                "fold": int(fold),
                                **fold_metrics,
                                "delta_log_loss_vs_v02": (
                                    float(fold_metrics["log_loss"])
                                    - float(fold_baseline_metrics["log_loss"])
                                ),
                                "delta_roc_auc_vs_v02": (
                                    float(fold_metrics["roc_auc"])
                                    - float(fold_baseline_metrics["roc_auc"])
                                ),
                            }
                        )
                metrics = binary_metrics(target, prediction)
                print(
                    f"{protocol_name} {model_name}: loss={metrics['log_loss']:.6f}, "
                    f"auc={metrics['roc_auc']:.6f}",
                    flush=True,
                )
        del features
        del dense
        gc.collect()

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_multiview_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    fold_metrics = pd.DataFrame(fold_metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    predictions.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    report = {
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "baseline_run_id": baseline_run_id,
        "protocols": protocol_names,
        "variants": variant_list,
        "regularization_c": c_list,
        "blend_weights": list(BLEND_WEIGHTS),
        "dense_features_by_variant": dense_names_by_variant,
        "include_ordered_features": include_ordered,
        "selection_policy": (
            "Use the pilot protocol only for screening; confirm a locked candidate "
            "on the remaining semantic-family protocols before promotion."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger_rows = metrics_frame.copy()
    ledger_rows.insert(0, "run_id", run_id)
    ledger_rows.insert(1, "timestamp_utc", timestamp.isoformat())
    ledger_rows["model"] = ledger_rows["model"] + "__" + ledger_rows["protocol"]
    ledger_rows["fold_scheme"] = "semantic-family-disjoint + session purge"
    ledger_rows["status"] = "completed"
    ledger_rows[
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
        description="Validate role-aware BGE evidence on frozen v0.2 hard folds."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--variant", action="append", choices=ALL_VARIANTS)
    parser.add_argument("--c", action="append", type=float)
    parser.add_argument(
        "--include-ordered",
        action="store_true",
        help="Append deterministic answer-feedback-repair trajectory features.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Screen candidates only on the frozen pilot protocol {PILOT_PROTOCOL}.",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        help="Evaluate only the named frozen protocol; may be repeated.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_multiview_validation(
        args.project_root,
        variants=args.variant or DEFAULT_VARIANTS,
        c_values=args.c or DEFAULT_C_VALUES,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
        include_ordered=args.include_ordered,
    )
    print(f"Multiview validation complete: {result['run_id']}")
    print(result["metrics"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
