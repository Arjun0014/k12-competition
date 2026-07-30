from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.alter_math_knowledge_state import (
    LABELS,
    PROTOCOL_ID,
    _cluster_bootstrap,
    _json_write,
    _label_metrics,
    _sha256,
    assert_runtime,
)


EXPECTED_OOF_SHA256 = (
    "a956a7a9d9e37f0c17502d25c3bb1ae64b3e401ebf017f599b4d238c73b36fe2"
)
ORIGINAL_SCORING_WALL_SECONDS = 10.7


def recover_report(oof_path: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    path = Path(oof_path).resolve()
    if _sha256(path) != EXPECTED_OOF_SHA256:
        raise ValueError("E910 recovery OOF hash changed.")
    frame = pd.read_parquet(path)
    if len(frame) != 8_573:
        raise ValueError("E910 recovery OOF row count changed.")
    candidate_columns = [f"pred_{index}" for index in range(len(LABELS))]
    comparator_columns = [f"prior_{index}" for index in range(len(LABELS))]
    required = {
        "session_id",
        "turn_id",
        "fold",
        *LABELS,
        *candidate_columns,
        *comparator_columns,
    }
    if not required.issubset(frame.columns):
        raise ValueError("E910 recovery OOF schema changed.")
    if frame[[*candidate_columns, *comparator_columns]].isna().any().any():
        raise ValueError("E910 recovery OOF predictions are incomplete.")

    fold_metrics: list[dict[str, object]] = []
    for fold in range(5):
        validation = frame[frame["fold"].eq(fold)]
        for label_index, label in enumerate(LABELS):
            metrics = _label_metrics(
                validation[label].to_numpy(),
                validation[candidate_columns[label_index]].to_numpy(),
                validation[comparator_columns[label_index]].to_numpy(),
            )
            fold_metrics.append({"fold": fold, "label": label, **metrics})

    label_metrics: dict[str, dict[str, float]] = {}
    for label_index, label in enumerate(LABELS):
        label_metrics[label] = _label_metrics(
            frame[label].to_numpy(),
            frame[candidate_columns[label_index]].to_numpy(),
            frame[comparator_columns[label_index]].to_numpy(),
        )
    macro = {
        key: float(np.mean([metrics[key] for metrics in label_metrics.values()]))
        for key in (
            "candidate_log_loss",
            "comparator_log_loss",
            "log_loss_gain",
            "candidate_brier",
            "comparator_brier",
            "brier_gain",
            "auroc",
            "average_precision",
            "ece_10",
        )
    }
    bootstrap = _cluster_bootstrap(
        frame,
        candidate_columns,
        comparator_columns,
    )
    fold_gains = [float(item["log_loss_gain"]) for item in fold_metrics]
    clauses = {
        "each_label_auroc_at_least_0_70": bool(
            all(metrics["auroc"] >= 0.70 for metrics in label_metrics.values())
        ),
        "each_label_ap_at_least_3x_prevalence": bool(
            all(
                metrics["average_precision"] >= 3 * metrics["prevalence"]
                for metrics in label_metrics.values()
            )
        ),
        "macro_ap_gain_at_least_0_08": bool(
            macro["average_precision"]
            - np.mean([metrics["prevalence"] for metrics in label_metrics.values()])
            >= 0.08
        ),
        "each_label_log_loss_gain_at_least_0_005": bool(
            all(
                metrics["log_loss_gain"] >= 0.005
                for metrics in label_metrics.values()
            )
        ),
        "macro_log_loss_gain_at_least_0_015": bool(
            macro["log_loss_gain"] >= 0.015
        ),
        "each_label_brier_gain_positive": bool(
            all(metrics["brier_gain"] > 0 for metrics in label_metrics.values())
        ),
        "macro_brier_gain_at_least_0_003": bool(macro["brier_gain"] >= 0.003),
        "each_label_ece_at_most_0_03": bool(
            all(metrics["ece_10"] <= 0.03 for metrics in label_metrics.values())
        ),
        "at_least_13_of_15_fold_gains_positive": bool(
            sum(gain > 0 for gain in fold_gains) >= 13
        ),
        "worst_fold_gain_at_least_minus_0_003": bool(
            min(fold_gains) >= -0.003
        ),
        "bootstrap_support_at_least_0_99": bool(
            bootstrap["positive_support"] >= 0.99
        ),
        "bootstrap_lower_bound_at_least_0_0075": bool(
            bootstrap["interval"][0] >= 0.0075
        ),
    }
    report = {
        "protocol": PROTOCOL_ID,
        "run_id": path.parent.name,
        "runtime": assert_runtime(),
        "original_scoring_wall_seconds": ORIGINAL_SCORING_WALL_SECONDS,
        "report_recovery_elapsed_seconds": time.perf_counter() - started,
        "rows": len(frame),
        "sessions": frame["session_id"].nunique(),
        "labels": label_metrics,
        "macro": macro,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "clauses": clauses,
        "passed": bool(all(clauses.values())),
        "success_column_read": False,
        "competition_text_accessed": False,
        "competition_outcomes_accessed": False,
        "recovered_from_completed_oof": True,
        "recovery_reason": "numpy.bool_ JSON serialization failure",
        "model_refit_or_prediction_rerun": False,
        "oof_sha256": EXPECTED_OOF_SHA256,
    }
    report_path = path.parent / "e910_external_report.json"
    _json_write(report_path, report)
    report["report_sha256"] = _sha256(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover E910's report from its one completed frozen OOF."
    )
    parser.add_argument("--oof", required=True)
    args = parser.parse_args()
    report = recover_report(args.oof)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
