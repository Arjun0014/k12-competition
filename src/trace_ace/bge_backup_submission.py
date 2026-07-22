from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.submission import _stage_and_test


ARTIFACT_NAME = "final_ensemble_v05_bge_backup.joblib"
METADATA_NAME = "final_ensemble_v05_bge_backup.metadata.json"
ZIP_NAME = "final_ensemble_v05_bge_backup.zip"
SOURCE_ARTIFACT = "final_ensemble_v04_rank.joblib"
PHASE_C_RUN_ID = "20260720T075633Z_environment_component_validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bge_base_backup(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    source_path = paths.root / "models" / SOURCE_ARTIFACT
    report_path = paths.experiments_dir / "runs" / PHASE_C_RUN_ID / "report.json"
    if not source_path.exists() or not report_path.exists():
        raise FileNotFoundError("BGE backup source artifact or validation report is missing.")
    phase_report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = phase_report.get("gate_report", {})
    gain = float(gate.get("mean_log_loss_gain", 0.0))
    if (
        phase_report.get("candidate") != "bge_replace"
        or phase_report.get("calibration") != "raw"
        or gain < 0.001
        or int(gate.get("improved_environments", 0)) < 3
    ):
        raise RuntimeError("The frozen BGE replacement does not clear the backup gate.")

    artifact = joblib.load(source_path)
    expected_weights = {
        "full_transcript": 0.25,
        "role_objective_dense": 0.25,
        "semantic_interaction_dense": 0.50,
    }
    if artifact.get("ensemble_weights") != expected_weights:
        raise ValueError("BGE source artifact does not contain the validated raw blend.")
    if artifact.get("semantic_encoder") != "BAAI/bge-base-en-v1.5":
        raise ValueError("BGE source artifact uses the wrong semantic encoder.")
    artifact["artifact_version"] = "5.0.0-bge-base-backup"
    artifact["model_type"] = "validated_raw_bge_base_replacement_backup"
    artifact["validation_reference"] = {
        "run_id": PHASE_C_RUN_ID,
        "candidate": "bge_replace",
        "calibration": "raw",
        "mean_log_loss_gain": gain,
        "improved_environments": int(gate["improved_environments"]),
        "V_final_accessed": False,
    }
    asset_dir = paths.root / "submission_src" / "assets"
    artifact_path = asset_dir / ARTIFACT_NAME
    metadata_path = asset_dir / METADATA_NAME
    joblib.dump(artifact, artifact_path, compress=3)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": _sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "source_artifact": SOURCE_ARTIFACT,
        "source_artifact_sha256": _sha256(source_path),
        "validation_run_id": PHASE_C_RUN_ID,
        "validation_candidate": "bge_replace/raw",
        "mean_hardened_log_loss_gain": gain,
        "public_projection_from_v02": 0.6081 - gain,
        "status": "verified_backup_not_top5_candidate",
        "manual_submission_only": True,
        "V_final_accessed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    smoke_format_path = min(
        paths.submission_formats,
        key=lambda path: len(pd.read_csv(path, usecols=["response_id"])),
    )
    smoke_format = pd.read_csv(smoke_format_path, dtype={"response_id": "string"})
    train_features = pd.read_csv(
        paths.train_features,
        dtype={
            "response_id": "string",
            "session_id": "string",
            "learning_objective_id": "string",
            "learning_objective": "string",
        },
    )
    smoke_features = smoke_format[["response_id"]].merge(
        train_features, on="response_id", how="left", validate="one_to_one"
    )
    if smoke_features.isna().any().any():
        raise ValueError("Backup smoke rows do not map completely to training features.")
    package = _stage_and_test(
        paths=paths,
        staging=paths.root / "tmp" / "final_ensemble_v05_bge_backup_submission",
        smoke_format=smoke_format,
        smoke_features=smoke_features,
        asset_names=[ARTIFACT_NAME, METADATA_NAME, "bge-base-en-v1.5"],
        zip_name=ZIP_NAME,
    )
    package["metadata"] = metadata
    verification_path = paths.root / "submission_builds" / (
        "final_ensemble_v05_bge_backup.verification.json"
    )
    verification = {
        **package,
        "zip_sha256": _sha256(Path(str(package["zip_path"]))),
        "local_smoke_rows": int(len(smoke_format)),
        "local_smoke_passed": True,
        "manual_submission_only": True,
    }
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verification


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and locally verify the frozen BGE-base backup ZIP."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_bge_base_backup(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
