from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from trace_ace.io import discover_project_paths


def _validate_submission(path: Path, expected_ids: pd.Series) -> None:
    frame = pd.read_csv(path, dtype={"response_id": "string"})
    if list(frame.columns) != ["response_id", "probability"]:
        raise ValueError(f"Unexpected submission columns: {list(frame.columns)}")
    if not frame["response_id"].equals(expected_ids.reset_index(drop=True)):
        raise ValueError("Submission response IDs or order do not match the format file.")
    if frame["probability"].isna().any() or not frame["probability"].between(0, 1).all():
        raise ValueError("Submission probabilities must be finite and within [0, 1].")


def _stage_and_test(
    paths,
    staging: Path,
    smoke_format: pd.DataFrame,
    smoke_features: pd.DataFrame,
    asset_names: list[str],
    zip_name: str,
) -> dict[str, object]:
    source_root = paths.root / "submission_src"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "assets").mkdir(parents=True)
    shutil.copy2(source_root / "main.py", staging / "main.py")
    for asset_name in asset_names:
        source = source_root / "assets" / asset_name
        target = staging / "assets" / asset_name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    data_dir = staging / "data"
    transcript_dir = data_dir / "test_transcripts"
    transcript_dir.mkdir(parents=True)
    smoke_format.to_csv(data_dir / "submission_format.csv", index=False, lineterminator="\n")
    smoke_features.to_csv(data_dir / "test_features.csv", index=False, lineterminator="\n")
    for session_id in smoke_features["session_id"].drop_duplicates():
        shutil.copy2(paths.transcript_dir / f"{session_id}.csv", transcript_dir)

    completed = subprocess.run(
        [sys.executable, "main.py"],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout or completed.stderr:
        raise RuntimeError("Submission runner emitted output; test-data logging must remain empty.")
    _validate_submission(staging / "submission.csv", smoke_format["response_id"])

    build_dir = paths.root / "submission_builds"
    build_dir.mkdir(parents=True, exist_ok=True)
    zip_path = build_dir / zip_name
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(staging / "main.py", arcname="main.py")
        for asset_name in asset_names:
            source = staging / "assets" / asset_name
            if source.is_dir():
                for directory in [source, *sorted(path for path in source.rglob("*") if path.is_dir())]:
                    relative = directory.relative_to(staging).as_posix().rstrip("/") + "/"
                    archive.writestr(relative, b"")
                for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
                    archive.write(file_path, arcname=file_path.relative_to(staging).as_posix())
            else:
                archive.write(source, arcname=f"assets/{asset_name}")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    valid_assets = all(
        any(name.startswith(f"assets/{asset_name}/") for name in names)
        if (staging / "assets" / asset_name).is_dir()
        else f"assets/{asset_name}" in names
        for asset_name in asset_names
    )
    if "main.py" not in names or not valid_assets:
        raise RuntimeError("Built ZIP has an invalid root structure.")
    return {
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "assets": asset_names,
    }


def build_submission(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    source_root = paths.root / "submission_src"
    required = [source_root / "main.py", source_root / "assets" / "objective_prior.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Submission source is incomplete: {missing}")

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
        raise ValueError("The supplied smoke format does not map completely to training features.")

    packages = [
        _stage_and_test(
            paths=paths,
            staging=paths.root / "tmp" / "objective_prior_submission",
            smoke_format=smoke_format,
            smoke_features=smoke_features,
            asset_names=["objective_prior.json"],
            zip_name="objective_prior_baseline.zip",
        )
    ]
    final_asset = source_root / "assets" / "final_ensemble_v02.joblib"
    if final_asset.exists():
        packages.append(
            _stage_and_test(
                paths=paths,
                staging=paths.root / "tmp" / "final_ensemble_v02_submission",
                smoke_format=smoke_format,
                smoke_features=smoke_features,
                asset_names=[
                    "final_ensemble_v02.joblib",
                    "final_ensemble_v02.metadata.json",
                    "bge-small-en-v1.5",
                ],
                zip_name="final_ensemble_v02.zip",
            )
        )
    rank_asset = source_root / "assets" / "final_ensemble_v04_rank.joblib"
    if rank_asset.exists():
        packages.append(
            _stage_and_test(
                paths=paths,
                staging=paths.root / "tmp" / "final_ensemble_v04_rank_submission",
                smoke_format=smoke_format,
                smoke_features=smoke_features,
                asset_names=[
                    "final_ensemble_v04_rank.joblib",
                    "final_ensemble_v04_rank.metadata.json",
                    "supervised_delta.safetensors",
                    "bge-base-en-v1.5",
                    "bge-small-en-v1.5",
                ],
                zip_name="final_ensemble_v04_rank.zip",
            )
        )
    feedback_asset = source_root / "assets" / "final_ensemble_v03_feedback.joblib"
    if feedback_asset.exists():
        packages.append(
            _stage_and_test(
                paths=paths,
                staging=paths.root / "tmp" / "final_ensemble_v03_feedback_submission",
                smoke_format=smoke_format,
                smoke_features=smoke_features,
                asset_names=[
                    "final_ensemble_v03_feedback.joblib",
                    "final_ensemble_v03_feedback.metadata.json",
                    "bge-small-en-v1.5",
                ],
                zip_name="final_ensemble_v03_feedback.zip",
            )
        )
    return {
        "packages": packages,
        "smoke_rows": len(smoke_format),
        "smoke_sessions": int(smoke_features["session_id"].nunique()),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and locally smoke-test a submission ZIP.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_submission(args.project_root)
    for package in result["packages"]:
        print(f"Built {package['zip_path']} ({package['zip_bytes']} bytes).")
    print(f"Local smoke tests passed for {result['smoke_rows']} rows.")


if __name__ == "__main__":
    main()
