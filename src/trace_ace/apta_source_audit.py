"""Target-free provenance and linkage audit for authorized APTA downloads.

This module deliberately does not model outcomes.  It inventories immutable
source files, inspects tabular schemas, and reports whether the source contains
the identity/group/role/transaction/learning-outcome linkage required for a
future preregistration.  Raw identifiers and cell values are never emitted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, TextIO


PROTOCOL_ID = "APTA_target_free_source_audit_v1"
SCHEMA_VERSION = "2026-07-29-apta-source-v1"
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 20 * 1024**3
MAX_ARCHIVE_MEMBER_BYTES = 5 * 1024**3
MAX_ARCHIVE_COMPRESSION_RATIO = 500.0
MIN_LINKED_LEARNERS = 50

DELIMITED_SUFFIXES = {".csv", ".tsv", ".tab", ".txt"}
RISKY_SUFFIXES = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".exe",
    ".jar",
    ".joblib",
    ".js",
    ".msi",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".ps1",
    ".pt",
    ".pth",
    ".pyc",
    ".scr",
    ".vbs",
}
DATA_LIKE_UNSUPPORTED_SUFFIXES = {
    ".arff",
    ".json",
    ".jsonl",
    ".parquet",
    ".sav",
    ".sas7bdat",
    ".xlsx",
    ".xls",
    ".xml",
}

CONCEPT_PATTERNS: dict[str, tuple[str, ...]] = {
    "student_id": (
        r"^anon student id$",
        r"^student id$",
        r"^studentid$",
        r"^learner id$",
        r"^participant id$",
        r"^user id$",
    ),
    "session_id": (
        r"^session id$",
        r"^sessionid$",
        r"^chat id$",
        r"^conversation id$",
    ),
    "class_id": (
        r"^class id$",
        r"^classroom id$",
        r"^school id$",
        r"^teacher id$",
        r"^section id$",
    ),
    "transaction_id": (
        r"^transaction id$",
        r"^transactionid$",
        r"^event id$",
        r"^row$",
    ),
    "time": (
        r"^time$",
        r"^timestamp$",
        r"^start time$",
        r"^event time$",
        r"^duration sec$",
    ),
    "role": (
        r"^role$",
        r"^speaker$",
        r"^speaker role$",
        r"^student role$",
        r"^collaboration role$",
        r"^solver tutor role$",
    ),
    "condition": (
        r"^condition$",
        r"^treatment$",
        r"^study condition$",
        r"^experimental condition$",
        r"^group assignment$",
    ),
    "problem": (
        r"^problem name$",
        r"^problem id$",
        r"^item id$",
        r"^task id$",
    ),
    "step": (
        r"^step name$",
        r"^step id$",
        r"^attempt at step$",
        r"^selection$",
    ),
    "knowledge_component": (
        r"^kc(?: .*)?$",
        r"^knowledge component(?: .*)?$",
        r"^skill(?: .*)?$",
    ),
    "student_action": (
        r"^student response type$",
        r"^input$",
        r"^student input$",
        r"^action$",
    ),
    "tutor_action": (
        r"^tutor response type$",
        r"^feedback text$",
        r"^tutor advice$",
        r"^tutor action$",
    ),
    "transaction_outcome": (
        r"^outcome$",
        r"^action evaluation$",
        r"^correctness$",
        r"^is correct$",
        r"^feedback classification$",
    ),
    "pre_outcome": (
        r"^pre(?:test| test| assessment)?(?: score)?$",
        r"^pre score$",
        r"^pretest .*(?:score|percent|total)$",
    ),
    "post_outcome": (
        r"^post(?:test| test| assessment)?(?: score)?$",
        r"^post score$",
        r"^posttest .*(?:score|percent|total)$",
    ),
    "gain_outcome": (
        r"^(?:normalized )?(?:learning )?gain(?: score)?$",
        r"^gain score$",
        r"^knowledge gain$",
    ),
    "assessment_outcome": (
        r"^(?:final )?assessment score$",
        r"^test score$",
        r"^exam score$",
        r"^learning outcome$",
    ),
    "chat_text": (
        r"^text$",
        r"^message$",
        r"^chat text$",
        r"^utterance$",
        r"^content$",
    ),
}

IDENTITY_CONCEPTS = {"student_id", "session_id", "class_id", "transaction_id"}
LEARNING_OUTCOME_CONCEPTS = {
    "pre_outcome",
    "post_outcome",
    "gain_outcome",
    "assessment_outcome",
}
TRANSACTION_CONCEPTS = {
    "transaction_id",
    "time",
    "problem",
    "step",
    "knowledge_component",
    "student_action",
    "tutor_action",
    "transaction_outcome",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_hash(value: str) -> str:
    return hashlib.sha256(f"APTA-AUDIT|{value}".encode("utf-8")).hexdigest()


def _normalize_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\(\)\[\]\{\}_/\\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _column_concepts(column: str) -> list[str]:
    normalized = _normalize_header(column)
    return [
        concept
        for concept, patterns in CONCEPT_PATTERNS.items()
        if any(re.fullmatch(pattern, normalized) for pattern in patterns)
    ]


def _delimiter_for(name: str, first_line: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    if suffix == ".csv":
        return ","
    try:
        return csv.Sniffer().sniff(first_line, delimiters=",\t;|").delimiter
    except csv.Error:
        return "\t" if "\t" in first_line else ","


def _open_text(binary: BinaryIO) -> TextIO:
    return io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")


def _audit_delimited_stream(
    stream: TextIO,
    *,
    source_name: str,
    size_bytes: int,
) -> tuple[dict[str, object], dict[str, set[str]]]:
    first_line = stream.readline()
    if not first_line:
        return (
            {
                "source": source_name,
                "bytes": int(size_bytes),
                "rows": 0,
                "columns": [],
                "column_concepts": {},
                "concept_nonmissing_rows": {},
                "concept_unique_counts": {},
                "malformed_rows": 0,
                "duplicate_transaction_keys": 0,
            },
            defaultdict(set),
        )
    delimiter = _delimiter_for(source_name, first_line)
    reader = csv.reader(chain((first_line,), stream), delimiter=delimiter)
    try:
        raw_header = next(reader)
    except StopIteration:
        raw_header = []
    columns = [str(value).strip() for value in raw_header]
    concepts_by_index = {
        index: _column_concepts(column) for index, column in enumerate(columns)
    }
    column_concepts = {
        columns[index]: concepts
        for index, concepts in concepts_by_index.items()
        if concepts
    }

    concept_nonmissing: Counter[str] = Counter()
    concept_values: dict[str, set[str]] = defaultdict(set)
    transaction_keys: set[str] = set()
    duplicate_transaction_keys = 0
    malformed_rows = 0
    rows = 0

    for row in reader:
        if not row or not any(str(value).strip() for value in row):
            continue
        rows += 1
        if len(row) != len(columns):
            malformed_rows += 1
            if len(row) < len(columns):
                row = row + [""] * (len(columns) - len(row))
            else:
                row = row[: len(columns)]

        transaction_parts: list[str] = []
        for index, value in enumerate(row):
            stripped = str(value).strip()
            if not stripped:
                continue
            for concept in concepts_by_index.get(index, []):
                concept_nonmissing[concept] += 1
                if concept in IDENTITY_CONCEPTS or concept in {
                    "role",
                    "condition",
                }:
                    concept_values[concept].add(_private_hash(stripped))
                if concept in IDENTITY_CONCEPTS | TRANSACTION_CONCEPTS:
                    transaction_parts.append(f"{concept}={stripped}")
        if transaction_parts:
            key = _private_hash("\x1f".join(transaction_parts))
            if key in transaction_keys:
                duplicate_transaction_keys += 1
            else:
                transaction_keys.add(key)

    table = {
        "source": source_name,
        "bytes": int(size_bytes),
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "rows": rows,
        "columns": columns,
        "column_concepts": column_concepts,
        "concept_nonmissing_rows": dict(sorted(concept_nonmissing.items())),
        "concept_unique_counts": {
            concept: len(values)
            for concept, values in sorted(concept_values.items())
        },
        "malformed_rows": malformed_rows,
        "duplicate_transaction_keys": duplicate_transaction_keys,
    }
    return table, concept_values


def _audit_delimited_path(
    path: Path,
    *,
    source_name: str,
) -> tuple[dict[str, object], dict[str, set[str]]]:
    with path.open("rb") as binary:
        return _audit_delimited_stream(
            _open_text(binary),
            source_name=source_name,
            size_bytes=path.stat().st_size,
        )


def _unsafe_archive_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    return candidate.is_absolute() or ".." in candidate.parts


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _audit_zip(
    path: Path,
    *,
    relative_name: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, set[str]]]]:
    members: list[dict[str, object]] = []
    tables: list[dict[str, object]] = []
    table_sets: list[dict[str, set[str]]] = []
    violations: list[str] = []
    total_uncompressed = 0
    total_compressed = 0

    with zipfile.ZipFile(path) as archive:
        infos = sorted(archive.infolist(), key=lambda item: item.filename)
        for info in infos:
            suffix = Path(info.filename).suffix.lower()
            unsafe_path = _unsafe_archive_path(info.filename)
            symlink = _zip_member_is_symlink(info)
            encrypted = bool(info.flag_bits & 0x1)
            risky = suffix in RISKY_SUFFIXES
            total_uncompressed += int(info.file_size)
            total_compressed += int(info.compress_size)
            member = {
                "name": info.filename,
                "bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "unsafe_path": unsafe_path,
                "symlink": symlink,
                "encrypted": encrypted,
                "risky_extension": risky,
            }
            members.append(member)
            if unsafe_path:
                violations.append(f"unsafe path: {info.filename}")
            if symlink:
                violations.append(f"symlink: {info.filename}")
            if encrypted:
                violations.append(f"encrypted member: {info.filename}")
            if risky:
                violations.append(f"risky member type: {info.filename}")
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                violations.append(f"oversized member: {info.filename}")

        ratio = total_uncompressed / max(total_compressed, 1)
        if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            violations.append("archive uncompressed size exceeds safety limit")
        if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
            violations.append("archive compression ratio exceeds safety limit")

        if not violations:
            for info in infos:
                if info.is_dir() or Path(info.filename).suffix.lower() not in DELIMITED_SUFFIXES:
                    continue
                with archive.open(info, "r") as binary:
                    table, concept_sets = _audit_delimited_stream(
                        _open_text(binary),
                        source_name=f"{relative_name}::{info.filename}",
                        size_bytes=info.file_size,
                    )
                tables.append(table)
                table_sets.append(concept_sets)

    archive_report = {
        "source": relative_name,
        "members": members,
        "member_count": len(members),
        "uncompressed_bytes": total_uncompressed,
        "compressed_bytes": total_compressed,
        "compression_ratio": total_uncompressed / max(total_compressed, 1),
        "safe": not violations,
        "violations": violations,
    }
    return archive_report, tables, table_sets


def _merge_sets(
    table_sets: Iterable[dict[str, set[str]]],
    concepts: set[str],
) -> set[str]:
    merged: set[str] = set()
    for values_by_concept in table_sets:
        for concept in concepts:
            merged.update(values_by_concept.get(concept, set()))
    return merged


def audit_apta_source(source_root: str | Path) -> dict[str, object]:
    """Audit an authorized local APTA source without exposing raw cell values."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"APTA source root does not exist: {root}")

    files: list[dict[str, object]] = []
    archives: list[dict[str, object]] = []
    tables: list[dict[str, object]] = []
    table_sets: list[dict[str, set[str]]] = []
    unsupported_data_like: list[str] = []

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        if suffix == ".zip":
            archive_report, archive_tables, archive_sets = _audit_zip(
                path, relative_name=relative
            )
            archives.append(archive_report)
            tables.extend(archive_tables)
            table_sets.extend(archive_sets)
        elif suffix in DELIMITED_SUFFIXES:
            table, concept_sets = _audit_delimited_path(path, source_name=relative)
            tables.append(table)
            table_sets.append(concept_sets)
        elif suffix in DATA_LIKE_UNSUPPORTED_SUFFIXES:
            unsupported_data_like.append(relative)

    all_concepts = sorted(
        {
            concept
            for table in tables
            for concepts in table["column_concepts"].values()
            for concept in concepts
        }
    )
    student_ids = _merge_sets(table_sets, {"student_id"})
    sessions = _merge_sets(table_sets, {"session_id"})
    classes = _merge_sets(table_sets, {"class_id"})
    roles = _merge_sets(table_sets, {"role"})
    conditions = _merge_sets(table_sets, {"condition"})

    transaction_table_indices = [
        index
        for index, table in enumerate(tables)
        if set(
            concept
            for concepts in table["column_concepts"].values()
            for concept in concepts
        ).intersection(TRANSACTION_CONCEPTS)
    ]
    learning_table_indices = [
        index
        for index, table in enumerate(tables)
        if set(
            concept
            for concepts in table["column_concepts"].values()
            for concept in concepts
        ).intersection(LEARNING_OUTCOME_CONCEPTS)
    ]
    transaction_students = _merge_sets(
        (table_sets[index] for index in transaction_table_indices), {"student_id"}
    )
    learning_students = _merge_sets(
        (table_sets[index] for index in learning_table_indices), {"student_id"}
    )
    linked_students = transaction_students.intersection(learning_students)

    has_pre = "pre_outcome" in all_concepts
    has_post = "post_outcome" in all_concepts
    has_gain = "gain_outcome" in all_concepts
    has_assessment = "assessment_outcome" in all_concepts
    has_learning_outcome = (has_pre and has_post) or has_gain or has_assessment
    has_transaction = bool(
        set(all_concepts).intersection(
            {"transaction_id", "problem", "step", "student_action", "tutor_action"}
        )
    )
    archive_safe = all(bool(archive["safe"]) for archive in archives)
    no_malformed_rows = all(int(table["malformed_rows"]) == 0 for table in tables)

    gates = {
        "source_files_present": bool(files),
        "supported_tabular_source_present": bool(tables),
        "archives_safe": archive_safe,
        "no_unsupported_data_like_files": not unsupported_data_like,
        "no_malformed_rows": no_malformed_rows,
        "student_identity_present": bool(student_ids),
        "stable_group_present": bool(sessions or classes),
        "role_present": bool(roles),
        "transaction_structure_present": has_transaction,
        "transaction_outcome_present": "transaction_outcome" in all_concepts,
        "pre_post_or_learning_outcome_present": has_learning_outcome,
        "condition_present": bool(conditions),
        "linked_learners_at_least_50": len(linked_students) >= MIN_LINKED_LEARNERS,
    }
    mandatory = tuple(gates)

    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "source_root": root.as_posix(),
        "source_file_count": len(files),
        "source_bytes": sum(int(file["bytes"]) for file in files),
        "files": files,
        "archives": archives,
        "tables": tables,
        "unsupported_data_like_files": unsupported_data_like,
        "detected_concepts": all_concepts,
        "privacy_safe_linkage_counts": {
            "students": len(student_ids),
            "sessions": len(sessions),
            "classes_or_teachers": len(classes),
            "roles": len(roles),
            "conditions": len(conditions),
            "transaction_students": len(transaction_students),
            "learning_outcome_students": len(learning_students),
            "linked_transaction_and_learning_outcome_students": len(linked_students),
        },
        "gates": gates,
        "discovery_ready": all(gates[name] for name in mandatory),
        "competition_outcome_accessed": False,
        "competition_predictions_accessed": False,
        "v_joint_accessed": False,
        "v_final_accessed": False,
        "raw_identifiers_emitted": False,
        "outcome_values_emitted": False,
    }


def write_report(report: dict[str, object], output_path: str | Path) -> str:
    """Write a deterministic JSON report and return its SHA-256."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return _sha256(path)
