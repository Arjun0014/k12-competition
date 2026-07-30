from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTECTED_ZIP = "submission_builds/final_ensemble_v05_bge_backup.zip"
PROTECTED_ZIP_SHA256 = (
    "65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186"
)
OUTPUT_CSV = "reports/generated/writeup_evidence_index.csv"
OUTPUT_MANIFEST = "reports/generated/writeup_evidence_manifest.json"


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    claim_text: str
    status: str
    experiment_family: str
    run_id: str
    metric_path: tuple[str, ...]
    metric: str
    comparison: str
    decision: str
    notes: str = ""


CLAIMS = (
    ClaimSpec(
        "bge_base_hardened_loss_gain",
        "Replacing BGE-small with raw BGE-base improved hardened log loss.",
        "deployed",
        "BGE-base semantic capacity",
        "20260720T075633Z_environment_component_validation",
        ("gate_report", "mean_log_loss_gain"),
        "mean_log_loss_gain",
        "raw BGE-base replacement versus v0.2",
        "deploy_as_v0.5",
        "The frozen audit improved all four environments but did not clear the "
        "stricter 0.0025 top-five gate.",
    ),
    ClaimSpec(
        "bge_base_hardened_auroc_delta",
        "The raw BGE-base replacement improved mean AUROC in the hardened audit.",
        "deployed",
        "BGE-base semantic capacity",
        "20260720T075633Z_environment_component_validation",
        ("gate_report", "mean_delta_roc_auc_vs_v02"),
        "mean_delta_roc_auc",
        "raw BGE-base replacement minus v0.2",
        "deploy_as_v0.5",
    ),
    ClaimSpec(
        "ordered_feedback_mean_loss_gain",
        "BGE-base plus ordered-feedback NB-SVM improved average audit log loss.",
        "validated_positive_not_deployed",
        "ordered answer-to-feedback NB-SVM",
        "20260717T023447Z_candidate_audit",
        ("summary", "mean_log_loss_improvement"),
        "mean_log_loss_improvement",
        "locked candidate versus v0.2 across four protocols",
        "reject_for_deployment",
        "Average gain was positive, but frozen loss-size and regime gates failed.",
    ),
    ClaimSpec(
        "ordered_feedback_mean_auroc_gain",
        "BGE-base plus ordered-feedback NB-SVM improved average audit AUROC.",
        "validated_positive_not_deployed",
        "ordered answer-to-feedback NB-SVM",
        "20260717T023447Z_candidate_audit",
        ("summary", "mean_roc_auc_improvement"),
        "mean_roc_auc_improvement",
        "locked candidate versus v0.2 across four protocols",
        "reject_for_deployment",
    ),
    ClaimSpec(
        "ordered_feedback_worst_fold_regression",
        "The ordered-feedback candidate regressed its worst fold.",
        "rejected_outcome",
        "ordered answer-to-feedback NB-SVM",
        "20260717T023447Z_candidate_audit",
        ("summary", "worst_fold_log_loss_regression"),
        "worst_fold_log_loss_regression",
        "locked candidate versus v0.2",
        "reject_for_deployment",
    ),
    ClaimSpec(
        "ordered_feedback_worst_regime_regression",
        "The ordered-feedback candidate regressed its worst legal regime.",
        "rejected_outcome",
        "ordered answer-to-feedback NB-SVM",
        "20260717T023447Z_candidate_audit",
        ("summary", "worst_legal_regime_log_loss_regression"),
        "worst_legal_regime_log_loss_regression",
        "locked candidate versus v0.2",
        "reject_for_deployment",
    ),
    ClaimSpec(
        "e770_move_state_loss_gain",
        "Tutor-move by student-state interactions regressed mean log loss.",
        "rejected_outcome",
        "E770 tutor-move/state interactions",
        "20260728T123253Z_tutor_move_state",
        ("preregistered_gate", "selected_row", "mean_log_loss_gain_vs_v05_raw"),
        "mean_log_loss_gain",
        "selected 10% candidate versus raw v0.5",
        "reject",
    ),
    ClaimSpec(
        "e810_numeric_loss_gain",
        "Productive numeric elaboration regressed mean log loss.",
        "rejected_outcome",
        "E810 productive numeric elaboration",
        "20260729T102615Z_productive_numeric_elaboration",
        ("selection_gate", "selected_row", "mean_log_loss_gain_vs_v05_raw"),
        "mean_log_loss_gain",
        "selected 10% candidate versus raw v0.5",
        "reject",
    ),
    ClaimSpec(
        "e820_contrastive_loss_gain",
        "MathDial contrastive objective alignment regressed mean log loss.",
        "rejected_outcome",
        "E820 contrastive objective alignment",
        "20260729T192619Z_mathdial_contrastive_competition",
        ("selection_gate", "selected_row", "mean_log_loss_gain_vs_v05_raw"),
        "mean_log_loss_gain",
        "selected 10% candidate versus raw v0.5",
        "reject",
    ),
    ClaimSpec(
        "e830_curriculum_loss_gain",
        "Curriculum progression regressed mean log loss.",
        "rejected_outcome",
        "E830 curriculum progression",
        "20260729T194357Z_curriculum_progression",
        ("selection_gate", "selected_row", "mean_log_loss_gain_vs_v05_raw"),
        "mean_log_loss_gain",
        "selected 10% candidate versus raw v0.5",
        "reject",
    ),
    ClaimSpec(
        "e840_studychat_rmse_gain",
        "StudyChat dialogue behaviors worsened unseen-learner exam RMSE.",
        "rejected_external",
        "E840 StudyChat longitudinal behavior",
        "20260729T195758Z_studychat_longitudinal",
        ("external_validation", "pooled_rmse_gain"),
        "pooled_rmse_gain",
        "dialogue candidate versus prior-grade/usage baseline",
        "reject_before_competition_outcomes",
    ),
    ClaimSpec(
        "e850_speaker_macro_f1",
        "Speaker-role recovery missed its frozen macro-F1 gate.",
        "rejected_target_free",
        "E850 speaker-role denoising",
        "20260729T201140Z_speaker_role_denoising",
        ("sample_oof_metrics", "macro_f1"),
        "target_free_macro_f1",
        "grouped role OOF",
        "reject_before_competition_outcomes",
    ),
    ClaimSpec(
        "e850_speaker_high_confidence_coverage",
        "Speaker-role recovery missed its high-confidence coverage gate.",
        "rejected_target_free",
        "E850 speaker-role denoising",
        "20260729T201140Z_speaker_role_denoising",
        ("sample_oof_metrics", "high_confidence_coverage"),
        "target_free_high_confidence_coverage",
        "grouped role OOF",
        "reject_before_competition_outcomes",
    ),
    ClaimSpec(
        "e860_competition_alignment_gain",
        "ASR math normalization reduced transcript/objective cosine alignment.",
        "rejected_target_free",
        "E860 ASR math normalization",
        "20260729T204454Z_asr_math_normalization",
        ("metrics", "competition", "objective_context_cosine_gain_mean"),
        "target_free_objective_context_cosine_gain",
        "normalized minus raw competition transcript/objective cosine",
        "reject_before_competition_outcomes",
    ),
    ClaimSpec(
        "e870_misconception_top5",
        "Misconception retrieval missed its all-label top-5 gate.",
        "rejected_external",
        "E870 misconception atlas",
        "20260729T205857Z_misconception_atlas",
        ("external_metrics", "top5_all_55"),
        "external_top5_accuracy",
        "leave-one-example retrieval across 55 labels",
        "reject_before_competition_outcomes",
    ),
    ClaimSpec(
        "e870_misconception_bootstrap_support",
        "Misconception retrieval had weak bootstrap support for top-5 >= 0.55.",
        "rejected_external",
        "E870 misconception atlas",
        "20260729T205857Z_misconception_atlas",
        ("misconception_bootstrap", "support_top5_at_least_0_55"),
        "external_bootstrap_support",
        "5,000 misconception-group replicates",
        "reject_before_competition_outcomes",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing report path: {'.'.join(path)}")
        value = value[key]
    return value


def _access_flag(report: dict[str, Any], key: str, run_id: str) -> bool:
    if key in report:
        return bool(report[key])
    if run_id == "20260720T075633Z_environment_component_validation":
        if key == "V_joint_accessed":
            return report.get("stage") == "joint"
        if key == "V_final_accessed":
            return bool(report.get("V_final_accessed", False))
    return False


def build_writeup_evidence(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protected_zip = root / PROTECTED_ZIP
    if not protected_zip.is_file():
        raise FileNotFoundError(f"Missing protected ZIP: {protected_zip}")
    protected_hash = sha256_file(protected_zip)
    if protected_hash != PROTECTED_ZIP_SHA256:
        raise RuntimeError(
            f"Protected ZIP hash changed: {protected_hash} != {PROTECTED_ZIP_SHA256}"
        )

    reports: dict[str, dict[str, Any]] = {}
    report_manifest: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in CLAIMS:
        report_path = (
            root / "experiments" / "runs" / spec.run_id / "report.json"
        )
        if spec.run_id not in reports:
            if not report_path.is_file():
                raise FileNotFoundError(f"Missing evidence report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if _access_flag(report, "V_final_accessed", spec.run_id):
                raise RuntimeError(f"Evidence report accessed V_final: {spec.run_id}")
            reports[spec.run_id] = report
            report_manifest[spec.run_id] = {
                "path": report_path.relative_to(root).as_posix(),
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
                "V_joint_accessed": _access_flag(
                    report, "V_joint_accessed", spec.run_id
                ),
                "V_final_accessed": False,
            }

        report = reports[spec.run_id]
        value = nested_value(report, spec.metric_path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Claim metric is not numeric: {spec.claim_id} -> {value!r}"
            )
        manifest_item = report_manifest[spec.run_id]
        rows.append(
            {
                "claim_id": spec.claim_id,
                "claim_text": spec.claim_text,
                "status": spec.status,
                "experiment_family": spec.experiment_family,
                "run_id": spec.run_id,
                "source_artifact": manifest_item["path"],
                "source_sha256": manifest_item["sha256"],
                "environment": "multi_environment"
                if spec.run_id
                in {
                    "20260720T075633Z_environment_component_validation",
                    "20260717T023447Z_candidate_audit",
                    "20260728T123253Z_tutor_move_state",
                    "20260729T102615Z_productive_numeric_elaboration",
                    "20260729T192619Z_mathdial_contrastive_competition",
                    "20260729T194357Z_curriculum_progression",
                }
                else "external_or_target_free",
                "metric": spec.metric,
                "value": repr(float(value)),
                "comparison": spec.comparison,
                "decision": spec.decision,
                "V_joint_accessed": str(
                    bool(manifest_item["V_joint_accessed"])
                ).lower(),
                "V_final_accessed": "false",
                "notes": spec.notes,
            }
        )

    output_csv = root / OUTPUT_CSV
    output_manifest = root / OUTPUT_MANIFEST
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "writeup_evidence_v1",
        "claim_count": len(rows),
        "report_count": len(report_manifest),
        "protected_zip": {
            "path": PROTECTED_ZIP,
            "bytes": protected_zip.stat().st_size,
            "sha256": protected_hash,
        },
        "reports": dict(sorted(report_manifest.items())),
        "evidence_index": {
            "path": output_csv.relative_to(root).as_posix(),
            "bytes": output_csv.stat().st_size,
            "sha256": sha256_file(output_csv),
        },
        "V_final_accessed": False,
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
