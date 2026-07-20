from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

from trace_ace.config import load_config
from trace_ace.io import (
    ProjectPaths,
    discover_project_paths,
    load_training_tables,
    validate_submission_formats,
)


TRANSCRIPT_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("utterance_id", pa.int64()),
        ("role", pa.string()),
        ("content", pa.string()),
        ("timestamp", pa.string()),
    ]
)

UTTERANCE_CACHE_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("utterance_id", pa.int64()),
        ("timestamp", pa.string()),
        ("role", pa.string()),
        ("content", pa.string()),
    ]
)

SESSION_FEATURE_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("n_utterances", pa.int32()),
        ("duration_seconds", pa.int32()),
        ("content_chars", pa.int64()),
        ("tutor_utterances", pa.int32()),
        ("student_utterances", pa.int32()),
        ("background_utterances", pa.int32()),
        ("other_role_utterances", pa.int32()),
        ("unclear_utterances", pa.int32()),
        ("empty_utterances", pa.int32()),
    ]
)

SESSION_TEXT_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("full_text", pa.large_string()),
        ("closing_text", pa.large_string()),
    ]
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _competition_source_files(paths: ProjectPaths) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = [
        ("train_features", paths.train_features),
        ("train_labels", paths.train_labels),
    ]
    files.extend(("submission_format", path) for path in paths.submission_formats)
    files.extend(
        ("train_transcript", path)
        for path in sorted(paths.transcript_dir.glob("*.csv"), key=lambda item: item.name)
    )
    return files


def build_raw_manifest(paths: ProjectPaths) -> tuple[pd.DataFrame, str]:
    output_path = paths.cache_dir / "raw_manifest.csv"
    previous: dict[str, tuple[int, int, str]] = {}
    if output_path.exists():
        old = pd.read_csv(output_path)
        required = {"relative_path", "bytes", "mtime_ns", "sha256"}
        if required.issubset(old.columns):
            previous = {
                str(row.relative_path): (int(row.bytes), int(row.mtime_ns), str(row.sha256))
                for row in old.itertuples(index=False)
            }

    records: list[dict[str, object]] = []
    source_files = _competition_source_files(paths)
    for index, (category, path) in enumerate(source_files, start=1):
        stat = path.stat()
        relative = path.relative_to(paths.root).as_posix()
        cached = previous.get(relative)
        if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            file_hash = cached[2]
        else:
            file_hash = sha256_file(path)
        records.append(
            {
                "relative_path": relative,
                "category": category,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_hash,
            }
        )
        if index % 5000 == 0:
            print(f"Manifested {index}/{len(source_files)} competition files.")

    manifest = pd.DataFrame.from_records(records).sort_values("relative_path").reset_index(drop=True)
    manifest.to_csv(output_path, index=False, lineterminator="\n")

    digest = hashlib.sha256()
    for row in manifest.itertuples(index=False):
        digest.update(f"{row.relative_path}|{row.bytes}|{row.sha256}\n".encode("utf-8"))
    return manifest, digest.hexdigest()


def _timestamp_seconds(value: str) -> int:
    pieces = str(value).split(":")
    if len(pieces) != 3:
        raise ValueError(f"Invalid transcript timestamp: {value!r}")
    hours, minutes, seconds = (int(piece) for piece in pieces)
    if minutes not in range(60) or seconds not in range(60) or hours < 0:
        raise ValueError(f"Invalid transcript timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _flush_tables(writer: pq.ParquetWriter, tables: list[pa.Table]) -> None:
    if not tables:
        return
    combined = pa.concat_tables(tables, promote_options="default")
    writer.write_table(combined)
    tables.clear()


def _flush_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, object]],
    schema: pa.Schema,
) -> None:
    if not rows:
        return
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()


def _cache_is_current(paths: ProjectPaths, source_digest: str, cache_version: str) -> bool:
    metadata_path = paths.cache_dir / "cache_metadata.json"
    required = [
        paths.cache_dir / "utterances.parquet",
        paths.cache_dir / "session_features.parquet",
        paths.cache_dir / "session_texts.parquet",
    ]
    if not metadata_path.exists() or not all(path.exists() for path in required):
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return (
        metadata.get("source_manifest_sha256") == source_digest
        and metadata.get("cache_version") == cache_version
    )


def build_transcript_cache(
    paths: ProjectPaths,
    source_digest: str,
    cache_version: str,
    closing_fraction: float,
) -> dict[str, object]:
    if _cache_is_current(paths, source_digest, cache_version):
        with (paths.cache_dir / "cache_metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        print("Transcript cache is current; skipping rebuild.")
        return metadata

    transcript_paths = sorted(paths.transcript_dir.glob("*.csv"), key=lambda item: item.name)
    if not transcript_paths:
        raise FileNotFoundError(f"No transcript CSVs found in {paths.transcript_dir}")

    targets = {
        "utterances": paths.cache_dir / "utterances.parquet",
        "features": paths.cache_dir / "session_features.parquet",
        "texts": paths.cache_dir / "session_texts.parquet",
    }
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in targets.items()}
    for path in temporary.values():
        path.unlink(missing_ok=True)

    utterance_writer = pq.ParquetWriter(
        temporary["utterances"], UTTERANCE_CACHE_SCHEMA, compression="zstd"
    )
    feature_writer = pq.ParquetWriter(
        temporary["features"], SESSION_FEATURE_SCHEMA, compression="zstd"
    )
    text_writer = pq.ParquetWriter(
        temporary["texts"], SESSION_TEXT_SCHEMA, compression="zstd"
    )

    utterance_tables: list[pa.Table] = []
    buffered_utterances = 0
    feature_rows: list[dict[str, object]] = []
    text_rows: list[dict[str, object]] = []
    total_utterances = 0
    total_chars = 0
    observed_roles: Counter[str] = Counter()

    try:
        for index, transcript_path in enumerate(transcript_paths, start=1):
            table = pacsv.read_csv(
                transcript_path,
                read_options=pacsv.ReadOptions(use_threads=False),
                convert_options=pacsv.ConvertOptions(column_types=TRANSCRIPT_SCHEMA),
            )
            missing = set(TRANSCRIPT_SCHEMA.names).difference(table.column_names)
            if missing:
                raise ValueError(f"{transcript_path.name} is missing columns: {sorted(missing)}")
            table = table.select(TRANSCRIPT_SCHEMA.names).cast(TRANSCRIPT_SCHEMA)
            n_rows = table.num_rows
            if n_rows == 0:
                raise ValueError(f"Empty transcript: {transcript_path.name}")

            expected_session = transcript_path.stem
            session_values = table.column("session_id")
            if not bool(pc.all(pc.equal(session_values, expected_session)).as_py()):
                raise ValueError(f"Internal session_id mismatch in {transcript_path.name}")

            utterance_ids = np.asarray(table.column("utterance_id"))
            if not np.array_equal(utterance_ids, np.arange(n_rows, dtype=np.int64)):
                raise ValueError(f"Non-contiguous utterance_id values in {transcript_path.name}")

            roles = [str(value) for value in table.column("role").to_pylist()]
            contents = ["" if value is None else str(value) for value in table.column("content").to_pylist()]
            timestamps = [str(value) for value in table.column("timestamp").to_pylist()]
            seconds = [_timestamp_seconds(value) for value in timestamps]
            if any(later < earlier for earlier, later in zip(seconds, seconds[1:])):
                raise ValueError(f"Non-monotonic timestamps in {transcript_path.name}")

            role_counts = Counter(roles)
            observed_roles.update(role_counts)
            char_count = sum(len(value) for value in contents)
            unclear_count = sum("[unclear]" in value.lower() for value in contents)
            empty_count = sum(not value.strip() for value in contents)
            marked_lines = [f"[{role.upper()}] {content}" for role, content in zip(roles, contents)]
            closing_start = max(0, math.floor(n_rows * (1.0 - closing_fraction)))

            feature_rows.append(
                {
                    "session_id": expected_session,
                    "n_utterances": n_rows,
                    "duration_seconds": max(seconds) - min(seconds),
                    "content_chars": char_count,
                    "tutor_utterances": role_counts.get("tutor", 0),
                    "student_utterances": role_counts.get("student", 0),
                    "background_utterances": role_counts.get("background", 0),
                    "other_role_utterances": n_rows
                    - role_counts.get("tutor", 0)
                    - role_counts.get("student", 0)
                    - role_counts.get("background", 0),
                    "unclear_utterances": unclear_count,
                    "empty_utterances": empty_count,
                }
            )
            text_rows.append(
                {
                    "session_id": expected_session,
                    "full_text": "\n".join(marked_lines),
                    "closing_text": "\n".join(marked_lines[closing_start:]),
                }
            )

            cache_table = table.select(
                ["session_id", "utterance_id", "timestamp", "role", "content"]
            ).cast(UTTERANCE_CACHE_SCHEMA)
            utterance_tables.append(cache_table)
            buffered_utterances += n_rows
            total_utterances += n_rows
            total_chars += char_count

            if buffered_utterances >= 50_000:
                _flush_tables(utterance_writer, utterance_tables)
                buffered_utterances = 0
            if len(feature_rows) >= 512:
                _flush_rows(feature_writer, feature_rows, SESSION_FEATURE_SCHEMA)
                _flush_rows(text_writer, text_rows, SESSION_TEXT_SCHEMA)
            if index % 2000 == 0:
                print(f"Cached {index}/{len(transcript_paths)} transcripts.")

        _flush_tables(utterance_writer, utterance_tables)
        _flush_rows(feature_writer, feature_rows, SESSION_FEATURE_SCHEMA)
        _flush_rows(text_writer, text_rows, SESSION_TEXT_SCHEMA)
    finally:
        utterance_writer.close()
        feature_writer.close()
        text_writer.close()

    for name, destination in targets.items():
        os.replace(temporary[name], destination)

    metadata: dict[str, object] = {
        "cache_version": cache_version,
        "source_manifest_sha256": source_digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": len(transcript_paths),
        "utterances": total_utterances,
        "content_characters": total_chars,
        "roles": dict(sorted(observed_roles.items())),
        "closing_fraction": closing_fraction,
    }
    with (paths.cache_dir / "cache_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def build_folds_and_modeling_table(
    paths: ProjectPaths,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    merged = features.merge(labels, on="response_id", how="inner", validate="one_to_one")
    session_features = pd.read_parquet(paths.cache_dir / "session_features.parquet")
    if session_features["session_id"].duplicated().any():
        raise ValueError("Session cache contains duplicate session_id values.")
    merged = merged.merge(session_features, on="session_id", how="left", validate="many_to_one")
    if merged["n_utterances"].isna().any():
        missing = int(merged["n_utterances"].isna().sum())
        raise ValueError(f"{missing} response rows are missing transcript cache data.")

    merged["objective_chars"] = merged["learning_objective"].str.len().astype("int32")
    merged["objective_words"] = (
        merged["learning_objective"].str.split().str.len().astype("int32")
    )
    merged["student_fraction"] = merged["student_utterances"] / merged["n_utterances"]
    merged["tutor_fraction"] = merged["tutor_utterances"] / merged["n_utterances"]
    merged["background_fraction"] = merged["background_utterances"] / merged["n_utterances"]
    merged["unclear_fraction"] = merged["unclear_utterances"] / merged["n_utterances"]

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_values = np.full(len(merged), -1, dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(merged, merged["target"], groups=merged["session_id"])
    ):
        fold_values[validation_indices] = fold
    if (fold_values < 0).any():
        raise RuntimeError("Some response rows did not receive a fold assignment.")
    merged["fold"] = fold_values

    fold_rows: list[dict[str, object]] = []
    all_sessions = set(merged["session_id"])
    for fold in range(n_splits):
        validation_mask = merged["fold"].eq(fold)
        validation_sessions = set(merged.loc[validation_mask, "session_id"])
        training_sessions = all_sessions.difference(validation_sessions)
        if validation_sessions.intersection(training_sessions):
            raise RuntimeError(f"Session leakage detected in fold {fold}.")
        fold_rows.append(
            {
                "fold": fold,
                "responses": int(validation_mask.sum()),
                "sessions": len(validation_sessions),
                "positive_rate": float(merged.loc[validation_mask, "target"].mean()),
            }
        )

    fold_frame = merged[["response_id", "session_id", "fold"]].copy()
    fold_frame.to_parquet(paths.cache_dir / "response_folds.parquet", index=False)
    merged.to_parquet(paths.cache_dir / "modeling_base.parquet", index=False)

    return {
        "responses": int(len(merged)),
        "sessions": int(merged["session_id"].nunique()),
        "objectives": int(merged["learning_objective_id"].nunique()),
        "positive_rate": float(merged["target"].mean()),
        "folds": fold_rows,
    }


def run_foundation(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(paths.root)
    features, labels = load_training_tables(paths)
    submission_summaries = validate_submission_formats(paths)
    manifest, source_digest = build_raw_manifest(paths)
    cache_metadata = build_transcript_cache(
        paths=paths,
        source_digest=source_digest,
        cache_version=str(config["cache_version"]),
        closing_fraction=float(config["closing_fraction"]),
    )
    modeling_summary = build_folds_and_modeling_table(
        paths=paths,
        features=features,
        labels=labels,
        n_splits=int(config["n_splits"]),
        seed=int(config["fold_seed"]),
    )

    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": source_digest,
        "manifest_files": int(len(manifest)),
        "manifest_bytes": int(manifest["bytes"].sum()),
        "submission_formats": submission_summaries,
        "transcript_cache": cache_metadata,
        "modeling": modeling_summary,
    }
    with (paths.cache_dir / "foundation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Trace the Ace manifests, caches, and folds.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_foundation(args.project_root)
    print(
        "Foundation complete: "
        f"{report['modeling']['responses']} responses, "
        f"{report['transcript_cache']['sessions']} sessions, "
        f"{report['transcript_cache']['utterances']} utterances."
    )


if __name__ == "__main__":
    main()
