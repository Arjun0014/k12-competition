from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from trace_ace.io import discover_project_paths


ROLE_TEXT_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("tutor_text", pa.large_string()),
        ("student_text", pa.large_string()),
        ("closing_tutor_text", pa.large_string()),
        ("closing_student_text", pa.large_string()),
        ("opening_text", pa.large_string()),
    ]
)

BEHAVIOR_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("role_switches", pa.int32()),
        ("tutor_words", pa.int32()),
        ("student_words", pa.int32()),
        ("tutor_questions", pa.int32()),
        ("student_questions", pa.int32()),
        ("tutor_affirmations", pa.int32()),
        ("tutor_corrections", pa.int32()),
        ("student_uncertainty", pa.int32()),
        ("student_reasoning_cues", pa.int32()),
        ("tutor_reasoning_cues", pa.int32()),
        ("student_digit_tokens", pa.int32()),
        ("tutor_digit_tokens", pa.int32()),
        ("student_mean_words", pa.float32()),
        ("tutor_mean_words", pa.float32()),
        ("closing_student_words", pa.int32()),
        ("closing_tutor_words", pa.int32()),
    ]
)

CSV_TYPES = {
    "session_id": pa.string(),
    "utterance_id": pa.int64(),
    "role": pa.string(),
    "content": pa.string(),
    "timestamp": pa.string(),
}

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
AFFIRM_RE = re.compile(
    r"\b(?:correct|exactly|excellent|great|good|well done|yes|right|brilliant|perfect|super)\b",
    re.IGNORECASE,
)
CORRECTION_RE = re.compile(
    r"\b(?:incorrect|wrong|not quite|try again|almost|check that|check your)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:i don't know|i do not know|not sure|don't understand|do not understand|confused|help)\b",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"\b(?:because|therefore|so|think|explain|why|how|reason)\b",
    re.IGNORECASE,
)


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _flush(writer: pq.ParquetWriter, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def build_role_cache(project_root: str | Path, closing_fraction: float = 0.25) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    role_path = paths.cache_dir / "session_role_texts.parquet"
    behavior_path = paths.cache_dir / "session_behavior.parquet"
    if role_path.exists() and behavior_path.exists():
        role_rows = pq.ParquetFile(role_path).metadata.num_rows
        behavior_rows = pq.ParquetFile(behavior_path).metadata.num_rows
        if role_rows == behavior_rows == 22_821:
            return {"sessions": role_rows, "reused": True}

    role_tmp = role_path.with_suffix(".parquet.tmp")
    behavior_tmp = behavior_path.with_suffix(".parquet.tmp")
    role_tmp.unlink(missing_ok=True)
    behavior_tmp.unlink(missing_ok=True)
    role_writer = pq.ParquetWriter(role_tmp, ROLE_TEXT_SCHEMA, compression="zstd")
    behavior_writer = pq.ParquetWriter(behavior_tmp, BEHAVIOR_SCHEMA, compression="zstd")
    role_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    transcript_paths = sorted(paths.transcript_dir.glob("*.csv"), key=lambda item: item.name)

    try:
        for index, transcript_path in enumerate(transcript_paths, start=1):
            table = pacsv.read_csv(
                transcript_path,
                read_options=pacsv.ReadOptions(use_threads=False),
                convert_options=pacsv.ConvertOptions(column_types=CSV_TYPES),
            )
            roles = [str(value) for value in table.column("role").to_pylist()]
            contents = ["" if value is None else str(value) for value in table.column("content").to_pylist()]
            n_rows = len(contents)
            closing_start = max(0, math.floor(n_rows * (1.0 - closing_fraction)))
            opening_end = min(n_rows, max(1, math.ceil(n_rows * closing_fraction)))

            tutor_parts = [text for role, text in zip(roles, contents) if role == "tutor"]
            student_parts = [text for role, text in zip(roles, contents) if role == "student"]
            closing_tutor_parts = [
                text
                for role, text in zip(roles[closing_start:], contents[closing_start:])
                if role == "tutor"
            ]
            closing_student_parts = [
                text
                for role, text in zip(roles[closing_start:], contents[closing_start:])
                if role == "student"
            ]
            tutor_text = "\n".join(tutor_parts)
            student_text = "\n".join(student_parts)
            closing_tutor = "\n".join(closing_tutor_parts)
            closing_student = "\n".join(closing_student_parts)
            opening_text = "\n".join(
                f"[{role.upper()}] {text}"
                for role, text in zip(roles[:opening_end], contents[:opening_end])
            )
            tutor_words = _count_words(tutor_text)
            student_words = _count_words(student_text)
            closing_tutor_words = _count_words(closing_tutor)
            closing_student_words = _count_words(closing_student)

            role_rows.append(
                {
                    "session_id": transcript_path.stem,
                    "tutor_text": tutor_text,
                    "student_text": student_text,
                    "closing_tutor_text": closing_tutor,
                    "closing_student_text": closing_student,
                    "opening_text": opening_text,
                }
            )
            behavior_rows.append(
                {
                    "session_id": transcript_path.stem,
                    "role_switches": sum(left != right for left, right in zip(roles, roles[1:])),
                    "tutor_words": tutor_words,
                    "student_words": student_words,
                    "tutor_questions": tutor_text.count("?"),
                    "student_questions": student_text.count("?"),
                    "tutor_affirmations": len(AFFIRM_RE.findall(tutor_text)),
                    "tutor_corrections": len(CORRECTION_RE.findall(tutor_text)),
                    "student_uncertainty": len(UNCERTAINTY_RE.findall(student_text)),
                    "student_reasoning_cues": len(REASONING_RE.findall(student_text)),
                    "tutor_reasoning_cues": len(REASONING_RE.findall(tutor_text)),
                    "student_digit_tokens": len(DIGIT_RE.findall(student_text)),
                    "tutor_digit_tokens": len(DIGIT_RE.findall(tutor_text)),
                    "student_mean_words": student_words / max(1, len(student_parts)),
                    "tutor_mean_words": tutor_words / max(1, len(tutor_parts)),
                    "closing_student_words": closing_student_words,
                    "closing_tutor_words": closing_tutor_words,
                }
            )
            if len(role_rows) >= 512:
                _flush(role_writer, role_rows, ROLE_TEXT_SCHEMA)
                _flush(behavior_writer, behavior_rows, BEHAVIOR_SCHEMA)
            if index % 3000 == 0:
                print(f"Built role cache for {index}/{len(transcript_paths)} sessions.", flush=True)
        _flush(role_writer, role_rows, ROLE_TEXT_SCHEMA)
        _flush(behavior_writer, behavior_rows, BEHAVIOR_SCHEMA)
    finally:
        role_writer.close()
        behavior_writer.close()
    os.replace(role_tmp, role_path)
    os.replace(behavior_tmp, behavior_path)
    return {"sessions": len(transcript_paths), "reused": False}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build role-separated transcript and behavior caches.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_role_cache(args.project_root)
    print(f"Role cache ready: {result['sessions']} sessions; reused={result['reused']}.")


if __name__ == "__main__":
    main()
