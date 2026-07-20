from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.objective_retrieval import _objective_terms, _tokens


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256
VIEW_SCHEMA_VERSION = "2026-07-17-v2-budgeted"
VIEW_COLUMNS = (
    "feedback_evidence",
    "student_evidence",
    "session_trajectory",
    "tutor_evidence",
)

ROLE_LINE_RE = re.compile(r"^\[([A-Z_]+)\]\s*(.*)$")
POSITIVE_FEEDBACK = (
    "correct",
    "well done",
    "good job",
    "exactly",
    "that's right",
    "thats right",
    "excellent",
    "brilliant",
)
CORRECTIVE_FEEDBACK = (
    "not quite",
    "try again",
    "have another",
    "check",
    "mistake",
    "remember",
    "almost",
)
STUDENT_REASONING = ("because", "so ", "therefore", "i think", "means", "equals")
STUDENT_UNCERTAINTY = (
    "don't know",
    "dont know",
    "not sure",
    "confused",
    "i can't",
    "i cant",
)


def _parse_role_lines(full_text: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw_line in str(full_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = ROLE_LINE_RE.match(line)
        if match:
            role = match.group(1).lower()
            content = match.group(2).strip()
        else:
            role = "unknown"
            content = line
        if content:
            parsed.append((role, content))
    return parsed


def _compact(text: str, max_words: int = 22) -> str:
    return " ".join(str(text).split()[:max_words])


def _objective_score(
    content: str,
    term_set: set[str],
    bigram_set: set[tuple[str, str]],
) -> float:
    tokens = _tokens(content)
    return _objective_score_from_features(
        set(tokens), set(zip(tokens, tokens[1:])), term_set, bigram_set
    )


def _objective_score_from_features(
    token_set: set[str],
    line_bigrams: set[tuple[str, str]],
    term_set: set[str],
    objective_bigrams: set[tuple[str, str]],
) -> float:
    overlap = term_set.intersection(token_set)
    score = sum(1.0 + min(len(token), 10) / 20.0 for token in overlap)
    if objective_bigrams and line_bigrams:
        score += 1.5 * len(objective_bigrams.intersection(line_bigrams))
    return float(score)


def _line_token_features(
    lines: list[tuple[str, str]],
) -> list[tuple[set[str], set[tuple[str, str]]]]:
    features: list[tuple[set[str], set[tuple[str, str]]]] = []
    for _, content in lines:
        tokens = _tokens(content)
        features.append((set(tokens), set(zip(tokens, tokens[1:]))))
    return features


def _select_role_evidence(
    lines: list[tuple[str, str]],
    role: str,
    objective: str,
    top_matches: int = 6,
    recent_lines: int = 4,
    line_features: list[tuple[set[str], set[tuple[str, str]]]] | None = None,
) -> str:
    terms, bigrams = _objective_terms(objective)
    term_set = set(terms)
    bigram_set = set(bigrams)
    cached_features = line_features if line_features is not None else _line_token_features(lines)
    candidates = []
    for index, (line_role, content) in enumerate(lines):
        if line_role != role:
            continue
        token_set, line_bigrams = cached_features[index]
        candidates.append(
            (
                index,
                content,
                _objective_score_from_features(
                    token_set, line_bigrams, term_set, bigram_set
                ),
            )
        )
    selected: set[int] = set()
    ranked = sorted(candidates, key=lambda item: (item[2], item[0]), reverse=True)
    selected.update(index for index, _, score in ranked[:top_matches] if score > 0)
    selected.update(index for index, _, _ in candidates[-recent_lines:])
    if candidates and len(selected) < min(4, len(candidates)):
        positions = np.linspace(0, len(candidates) - 1, min(4, len(candidates))).astype(int)
        selected.update(candidates[position][0] for position in positions)
    rendered = [
        f"[{role.upper()}] {_compact(content, max_words=16)}"
        for index, (line_role, content) in enumerate(lines)
        if line_role == role and index in selected
    ]
    return "\n".join([f"[OBJECTIVE] {objective.strip()}", *rendered])


def _cue_bonus(student: str, tutor: str) -> float:
    student_lower = student.lower()
    tutor_lower = tutor.lower()
    score = 0.0
    score += 1.0 * sum(cue in tutor_lower for cue in POSITIVE_FEEDBACK)
    score += 1.0 * sum(cue in tutor_lower for cue in CORRECTIVE_FEEDBACK)
    score += 0.5 * sum(cue in student_lower for cue in STUDENT_REASONING)
    score += 0.5 * sum(cue in student_lower for cue in STUDENT_UNCERTAINTY)
    return score


def _feedback_windows(
    lines: list[tuple[str, str]],
    objective: str,
    top_windows: int = 5,
    recent_windows: int = 2,
    line_features: list[tuple[set[str], set[tuple[str, str]]]] | None = None,
) -> str:
    terms, bigrams = _objective_terms(objective)
    term_set = set(terms)
    bigram_set = set(bigrams)
    cached_features = line_features if line_features is not None else _line_token_features(lines)
    dialogue = [(index, role, content) for index, (role, content) in enumerate(lines) if role in {"student", "tutor"}]
    windows: list[tuple[int, float, str]] = []
    for (left_index, left_role, left), (right_index, right_role, right) in zip(
        dialogue, dialogue[1:]
    ):
        if left_role == right_role:
            continue
        left_tokens, left_bigrams = cached_features[left_index]
        right_tokens, right_bigrams = cached_features[right_index]
        objective_score = _objective_score_from_features(
            left_tokens, left_bigrams, term_set, bigram_set
        ) + _objective_score_from_features(
            right_tokens, right_bigrams, term_set, bigram_set
        )
        if left_role == "student":
            student, tutor = left, right
            rendered = (
                f"[ANSWER] {_compact(left, max_words=14)}\n"
                f"[FEEDBACK] {_compact(right, max_words=14)}"
            )
        else:
            student, tutor = right, left
            rendered = (
                f"[PROMPT] {_compact(left, max_words=14)}\n"
                f"[RESPONSE] {_compact(right, max_words=14)}"
            )
        score = objective_score + _cue_bonus(student, tutor)
        windows.append((max(left_index, right_index), score, rendered))

    selected: set[int] = set()
    ranked = sorted(enumerate(windows), key=lambda item: (item[1][1], item[1][0]), reverse=True)
    selected.update(index for index, (_, score, _) in ranked[:top_windows] if score > 0)
    selected.update(range(max(0, len(windows) - recent_windows), len(windows)))
    if windows and len(selected) < min(4, len(windows)):
        positions = np.linspace(0, len(windows) - 1, min(4, len(windows))).astype(int)
        selected.update(int(position) for position in positions)
    rendered = [windows[index][2] for index in sorted(selected, key=lambda item: windows[item][0])]
    return "\n".join([f"[OBJECTIVE] {objective.strip()}", *rendered])


def _session_trajectory(lines: list[tuple[str, str]]) -> str:
    dialogue = [(role, content) for role, content in lines if role in {"student", "tutor"}]
    if not dialogue:
        return "[SESSION_TRAJECTORY]"
    count = len(dialogue)
    opening = list(range(0, min(5, count)))
    midpoint = count // 2
    middle = list(range(max(0, midpoint - 3), min(count, midpoint + 3)))
    closing = list(range(max(0, count - 8), count))
    selected = sorted(set([*opening, *middle, *closing]))
    rendered = [
        f"[{dialogue[index][0].upper()}] {_compact(dialogue[index][1], max_words=12)}"
        for index in selected
    ]
    return "\n".join(["[SESSION_TRAJECTORY]", *rendered])


def build_multiview_texts(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    output_path = paths.cache_dir / "response_multiview_texts.parquet"
    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet",
        columns=["response_id", "session_id", "learning_objective"],
    )
    if output_path.exists():
        try:
            cached = pd.read_parquet(
                output_path, columns=["response_id", "view_schema_version"]
            )
        except (KeyError, ValueError):
            cached = pd.DataFrame()
        if (
            not cached.empty
            and list(cached["response_id"]) == list(frame["response_id"])
            and set(cached["view_schema_version"]) == {VIEW_SCHEMA_VERSION}
        ):
            return {
                "responses": len(cached),
                "reused": True,
                "view_schema_version": VIEW_SCHEMA_VERSION,
                "path": str(output_path),
            }

    session_texts = pd.read_parquet(
        paths.cache_dir / "session_texts.parquet", columns=["session_id", "full_text"]
    ).set_index("session_id")
    indexed_frame = frame.reset_index(names="response_row")
    rows: list[dict[str, str] | None] = [None] * len(frame)
    completed = 0
    for session_id, group in indexed_frame.groupby("session_id", sort=False):
        session_key = str(session_id)
        lines = _parse_role_lines(str(session_texts.at[session_key, "full_text"]))
        line_features = _line_token_features(lines)
        trajectory = _session_trajectory(lines)
        for response in group.itertuples(index=False):
            objective = str(response.learning_objective)
            rows[int(response.response_row)] = {
                "response_id": str(response.response_id),
                "session_id": session_key,
                "view_schema_version": VIEW_SCHEMA_VERSION,
                "student_evidence": _select_role_evidence(
                    lines, "student", objective, line_features=line_features
                ),
                "tutor_evidence": _select_role_evidence(
                    lines, "tutor", objective, line_features=line_features
                ),
                "feedback_evidence": _feedback_windows(
                    lines, objective, line_features=line_features
                ),
                "session_trajectory": trajectory,
            }
            completed += 1
            if completed % 5000 == 0:
                print(
                    f"Built multiview text for {completed}/{len(frame)} responses.",
                    flush=True,
                )
    if any(row is None for row in rows):
        raise RuntimeError("At least one multiview response was not constructed.")
    result = pd.DataFrame(rows)
    if list(result["response_id"]) != list(frame["response_id"]):
        raise RuntimeError("Multiview text order does not match the modeling frame.")
    result.to_parquet(output_path, index=False, compression="zstd")
    return {
        "responses": len(result),
        "reused": False,
        "view_schema_version": VIEW_SCHEMA_VERSION,
        "path": str(output_path),
    }


def build_multiview_embedding_cache(
    project_root: str | Path,
    batch_size: int = 64,
    chunk_size: int = 512,
) -> dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required for multiview caches.") from error

    paths = discover_project_paths(project_root)
    build_multiview_texts(paths.root)
    views = pd.read_parquet(paths.cache_dir / "response_multiview_texts.parquet")
    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet", columns=["response_id"]
    )
    if list(views["response_id"]) != list(frame["response_id"]):
        raise ValueError("Multiview text rows do not align with modeling data.")

    model_dir = paths.root / "assets" / "pretrained" / "bge-small-en-v1.5"
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    outputs: dict[str, str] = {}
    expected_shape = (len(frame), EMBEDDING_DIMENSION)
    for view in VIEW_COLUMNS:
        output_path = paths.cache_dir / f"bge_small_{view}_256.npy"
        metadata_path = paths.cache_dir / f"bge_small_{view}_256.metadata.json"
        progress_path = paths.cache_dir / f"bge_small_{view}_256.progress.json"
        unique_path = paths.cache_dir / f"bge_small_{view}_256.unique.npy"
        codes, unique_texts = pd.factorize(
            views[view].fillna("").astype(str), sort=False
        )
        unique_shape = (len(unique_texts), EMBEDDING_DIMENSION)

        if (
            output_path.exists()
            and metadata_path.exists()
            and not progress_path.exists()
            and not unique_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("view_schema_version") == VIEW_SCHEMA_VERSION:
                embeddings = np.lib.format.open_memmap(output_path, mode="r")
                if embeddings.shape != expected_shape:
                    raise ValueError(f"Existing {view} embedding cache has the wrong shape.")
                outputs[view] = str(output_path)
                print(
                    f"Reused completed {view} cache ({len(unique_texts)} unique texts).",
                    flush=True,
                )
                continue

        if unique_path.exists():
            unique_embeddings = np.lib.format.open_memmap(unique_path, mode="r+")
            if unique_embeddings.shape != unique_shape:
                raise ValueError(f"Existing {view} unique cache has the wrong shape.")
        else:
            unique_embeddings = np.lib.format.open_memmap(
                unique_path, mode="w+", dtype=np.float32, shape=unique_shape
            )

        start = 0
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            start = int(progress.get("completed_unique_texts", 0))
        for chunk_start in range(start, len(unique_texts), chunk_size):
            chunk_end = min(len(unique_texts), chunk_start + chunk_size)
            encoded = model.encode(
                unique_texts[chunk_start:chunk_end].tolist(),
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            unique_embeddings[chunk_start:chunk_end] = encoded
            unique_embeddings.flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "completed_unique_texts": chunk_end,
                        "total_unique_texts": len(unique_texts),
                        "response_rows": len(frame),
                        "model": MODEL_NAME,
                        "max_sequence_length": MAX_SEQUENCE_LENGTH,
                        "view_schema_version": VIEW_SCHEMA_VERSION,
                        "view": view,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"Encoded {view}: {chunk_end}/{len(unique_texts)} unique texts "
                f"for {len(frame)} responses.",
                flush=True,
            )

        embeddings = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=expected_shape
        )
        for row_start in range(0, len(frame), max(chunk_size * 8, 4096)):
            row_end = min(len(frame), row_start + max(chunk_size * 8, 4096))
            embeddings[row_start:row_end] = unique_embeddings[codes[row_start:row_end]]
        embeddings.flush()
        del embeddings
        del unique_embeddings
        unique_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "embedding_dimension": EMBEDDING_DIMENSION,
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                    "model": MODEL_NAME,
                    "response_rows": len(frame),
                    "unique_texts": len(unique_texts),
                    "view": view,
                    "view_schema_version": VIEW_SCHEMA_VERSION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs[view] = str(output_path)
    return {"responses": len(frame), "model_dir": str(model_dir), "outputs": outputs}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build role-aware multi-view BGE caches.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Build deterministic text views without encoding them.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.text_only:
        result = build_multiview_texts(args.project_root)
    else:
        result = build_multiview_embedding_cache(
            args.project_root, batch_size=args.batch_size, chunk_size=args.chunk_size
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
