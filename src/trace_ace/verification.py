from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from trace_ace.io import discover_project_paths


def verify_project(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    manifest = pd.read_csv(paths.cache_dir / "raw_manifest.csv")
    check("manifest_unique_paths", manifest["relative_path"].is_unique, f"rows={len(manifest)}")
    check(
        "manifest_sha256_shape",
        manifest["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all(),
        "all hashes must be lowercase SHA-256",
    )

    foundation_report_path = paths.cache_dir / "foundation_report.json"
    with foundation_report_path.open("r", encoding="utf-8") as handle:
        foundation = json.load(handle)
    expected_sessions = int(foundation["transcript_cache"]["sessions"])
    expected_utterances = int(foundation["transcript_cache"]["utterances"])
    expected_responses = int(foundation["modeling"]["responses"])

    cache_expectations = {
        "utterances.parquet": expected_utterances,
        "session_features.parquet": expected_sessions,
        "session_texts.parquet": expected_sessions,
        "modeling_base.parquet": expected_responses,
        "response_folds.parquet": expected_responses,
    }
    for filename, expected_rows in cache_expectations.items():
        actual_rows = pq.ParquetFile(paths.cache_dir / filename).metadata.num_rows
        check(
            f"cache_rows_{filename}",
            actual_rows == expected_rows,
            f"expected={expected_rows}, actual={actual_rows}",
        )

    modeling = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet",
        columns=["response_id", "session_id", "fold", "target"],
    )
    check("response_ids_unique", modeling["response_id"].is_unique, f"rows={len(modeling)}")
    check("targets_binary", modeling["target"].isin([0, 1]).all(), "target must be 0/1")
    check(
        "folds_complete",
        sorted(modeling["fold"].unique().tolist()) == [0, 1, 2, 3, 4],
        f"folds={sorted(modeling['fold'].unique().tolist())}",
    )
    session_fold_counts = modeling.groupby("session_id", observed=True)["fold"].nunique()
    check(
        "sessions_in_one_fold",
        session_fold_counts.eq(1).all(),
        f"violations={int(session_fold_counts.ne(1).sum())}",
    )

    resources = pd.read_csv(paths.root / "configs" / "external_resources.csv")
    allowlisted = resources[resources["project_status"].eq("allowlisted_candidate")]
    allowlist_safe = (
        allowlisted["commercial_use"].eq("yes")
        & ~allowlisted["license"].str.contains("NC", case=False, na=False)
    ).all()
    check(
        "external_allowlist_commercial",
        bool(allowlist_safe),
        f"allowlisted={allowlisted['name'].tolist()}",
    )

    ledger_path = paths.experiments_dir / "experiment_ledger.csv"
    ledger = pd.read_csv(ledger_path)
    check("experiment_ledger_nonempty", len(ledger) > 0, f"rows={len(ledger)}")
    check(
        "experiment_metrics_finite",
        ledger[["log_loss", "roc_auc", "brier_score", "ece_10"]].notna().all().all(),
        "baseline metric columns must be populated",
    )

    package_expectations = {
        "objective_prior_baseline.zip": {"main.py", "assets/objective_prior.json"},
        "word_full_tfidf_baseline.zip": {
            "main.py",
            "assets/objective_prior.json",
            "assets/word_full_tfidf.joblib",
            "assets/word_full_tfidf.metadata.json",
        },
    }
    for package_name, expected_members in package_expectations.items():
        zip_path = paths.root / "submission_builds" / package_name
        check(f"package_exists_{package_name}", zip_path.exists(), str(zip_path))
        with zipfile.ZipFile(zip_path) as archive:
            zip_names = set(archive.namelist())
            bad_names = [name for name in zip_names if name.startswith("data/")]
        check(
            f"package_members_{package_name}",
            expected_members.issubset(zip_names),
            f"members={sorted(zip_names)}",
        )
        check(
            f"package_excludes_data_{package_name}",
            not bad_names,
            f"unexpected={bad_names}",
        )
        check(
            f"package_under_size_limit_{package_name}",
            zip_path.stat().st_size < 60 * 1024**3,
            f"bytes={zip_path.stat().st_size}",
        )

    sparse_model_path = paths.models_dir / "word_full_tfidf.joblib"
    with (paths.models_dir / "word_full_tfidf.metadata.json").open(
        "r", encoding="utf-8"
    ) as handle:
        sparse_metadata = json.load(handle)
    digest = hashlib.sha256()
    with sparse_model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    check(
        "sparse_artifact_sha256",
        digest.hexdigest() == sparse_metadata["artifact_sha256"],
        f"expected={sparse_metadata['artifact_sha256']}, actual={digest.hexdigest()}",
    )
    check(
        "sparse_artifact_python_version",
        str(sparse_metadata.get("build_python_version", "")).startswith("3.12."),
        f"build_python_version={sparse_metadata.get('build_python_version')}",
    )
    check(
        "sparse_artifact_sklearn_version",
        sparse_metadata.get("build_sklearn_version") == "1.8.0",
        f"build_sklearn_version={sparse_metadata.get('build_sklearn_version')}",
    )

    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "checks_passed": len(checks),
        "checks": checks,
    }
    output_dir = paths.root / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "milestone1_verification.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**report, "output_path": str(output_path)}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify first-milestone project artifacts.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = verify_project(args.project_root)
    print(f"Verification passed: {report['checks_passed']} checks; {report['output_path']}")


if __name__ == "__main__":
    main()
