from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax

from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.semantic_hard_validation import (
    prepare_semantic_features,
    semantic_logistic_oof,
)


BASELINE_RUN_ID = "20260716T183434Z_robust_validation"
PILOT_PROTOCOL = "semantic_k50_s0"
DEFAULT_C_VALUES = (0.01, 0.03, 0.1)
BLEND_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50)


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), np.float32(1e-8)
    )


def run_nli_cross_encoder_validation(
    project_root: str | Path,
    c_values: Iterable[float] = DEFAULT_C_VALUES,
    protocols: set[str] | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    frame = pd.read_parquet(paths.cache_dir / "modeling_base.parquet").reset_index(
        drop=True
    )
    target = frame["target"].to_numpy(dtype=np.float64)
    baseline = pd.read_parquet(
        paths.experiments_dir / "runs" / baseline_run_id / "oof_predictions.parquet"
    )
    if protocols is not None:
        baseline = baseline.loc[baseline["protocol"].isin(protocols)].copy()
    if baseline.empty:
        raise ValueError("No baseline predictions match requested protocols.")
    pooled = np.load(paths.cache_dir / "nli_v3_small_mastery_pooled_256.npy")
    logits = np.load(paths.cache_dir / "nli_v3_small_mastery_logits_256.npy")
    if pooled.shape != (len(frame), 768) or logits.shape != (len(frame), 3):
        raise ValueError("NLI cache shape mismatch.")
    if not np.isfinite(pooled).all() or not np.isfinite(logits).all():
        raise ValueError("NLI cache contains non-finite values.")
    features = _row_normalize(pooled)
    probabilities = softmax(logits.astype(np.float64), axis=1)
    _, dense, dense_names = prepare_semantic_features(paths.cache_dir, frame)
    nli_dense = np.column_stack(
        [
            logits,
            probabilities,
            logits[:, 1] - logits[:, 0],
            probabilities[:, 1] - probabilities[:, 0],
        ]
    )
    dense = np.column_stack([dense, nli_dense])
    nli_dense_names = [
        "nli_logit_contradiction",
        "nli_logit_entailment",
        "nli_logit_neutral",
        "nli_probability_contradiction",
        "nli_probability_entailment",
        "nli_probability_neutral",
        "nli_logit_entailment_minus_contradiction",
        "nli_probability_entailment_minus_contradiction",
    ]

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    protocol_names = list(dict.fromkeys(baseline["protocol"].astype(str)))
    c_list = list(c_values)
    for c_value in c_list:
        for protocol_name in protocol_names:
            base = baseline.loc[baseline["protocol"].eq(protocol_name)].copy()
            if list(base["response_id"]) != list(frame["response_id"]):
                raise ValueError(f"Baseline row order mismatch for {protocol_name}.")
            folds = base["fold"].to_numpy(dtype=np.int8)
            print(f"NLI pooled probe C={c_value:g} on {protocol_name}", flush=True)
            nli_prediction = semantic_logistic_oof(
                frame,
                folds,
                features,
                dense,
                regularization_c=c_value,
                n_splits=int(folds.max()) + 1,
            )
            pred_v02 = base["pred_v02"].to_numpy(dtype=np.float64)
            baseline_metrics = binary_metrics(target, pred_v02)
            candidates = [("standalone", 1.0, nli_prediction)]
            candidates.extend(
                (
                    "blend",
                    weight,
                    (1.0 - weight) * pred_v02 + weight * nli_prediction,
                )
                for weight in BLEND_WEIGHTS
            )
            for prediction_type, weight, candidate in candidates:
                metrics = binary_metrics(target, candidate)
                model_name = f"nli_v3_small_c{c_value:g}"
                metric_rows.append(
                    {
                        "protocol": protocol_name,
                        "model": model_name,
                        "prediction_type": prediction_type,
                        "new_model_weight": weight,
                        **metrics,
                        "delta_log_loss_vs_v02": metrics["log_loss"] - baseline_metrics["log_loss"],
                        "delta_roc_auc_vs_v02": metrics["roc_auc"] - baseline_metrics["roc_auc"],
                    }
                )
                for fold in range(int(folds.max()) + 1):
                    mask = folds == fold
                    fm = binary_metrics(target[mask], candidate[mask])
                    fb = binary_metrics(target[mask], pred_v02[mask])
                    fold_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold,
                            "model": model_name,
                            "prediction_type": prediction_type,
                            "new_model_weight": weight,
                            **fm,
                            "delta_log_loss_vs_v02": fm["log_loss"] - fb["log_loss"],
                            "delta_roc_auc_vs_v02": fm["roc_auc"] - fb["roc_auc"],
                        }
                    )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "protocol": protocol_name,
                        "response_id": frame["response_id"],
                        "fold": folds,
                        "target": target,
                        "regularization_c": c_value,
                        "pred_v02": pred_v02,
                        "pred_nli": nli_prediction,
                    }
                )
            )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_nli_cross_encoder_validation")
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_dir / "oof_predictions.parquet", index=False
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at_utc": timestamp.isoformat(),
                "baseline_run_id": baseline_run_id,
                "protocols": protocol_names,
                "model": "cross-encoder/nli-deberta-v3-small",
                "license": "Apache-2.0",
                "max_sequence_length": 256,
                "regularization_c": c_list,
                "blend_weights": list(BLEND_WEIGHTS),
                "dense_features": [*dense_names, *nli_dense_names],
                "hypothesis_template": (
                    "The student demonstrates mastery of this learning objective: "
                    "<objective>."
                ),
                "selection_policy": "Use only the frozen pilot for selection, then lock one candidate.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen DeBERTa NLI cross-encoder features."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--c", action="append", type=float)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args(list(argv) if argv is not None else None)
    protocols = set(args.protocol) if args.protocol else None
    if args.quick:
        protocols = {PILOT_PROTOCOL}
    result = run_nli_cross_encoder_validation(
        args.project_root,
        c_values=args.c or DEFAULT_C_VALUES,
        protocols=protocols,
        baseline_run_id=args.baseline_run_id,
    )
    print(f"NLI validation complete: {result['run_id']}")
    print(result["metrics"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
