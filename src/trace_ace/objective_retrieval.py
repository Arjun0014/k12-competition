from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ROLE_PREFIX_RE = re.compile(r"^\[[A-Z_]+\]\s*")

STOPWORDS = {
    "able",
    "about",
    "after",
    "again",
    "against",
    "all",
    "also",
    "and",
    "another",
    "any",
    "are",
    "around",
    "been",
    "before",
    "being",
    "between",
    "both",
    "can",
    "could",
    "different",
    "each",
    "find",
    "first",
    "for",
    "from",
    "given",
    "have",
    "identify",
    "into",
    "know",
    "knowing",
    "learn",
    "learning",
    "make",
    "more",
    "numbers",
    "number",
    "objects",
    "other",
    "recognise",
    "represent",
    "solve",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "these",
    "they",
    "this",
    "through",
    "understand",
    "understanding",
    "using",
    "value",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
    "work",
    "working",
    "would",
}


def _stem(token: str) -> str:
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(token.lower()) for token in TOKEN_RE.findall(text)]


def _objective_terms(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    raw = _tokens(text)
    terms = [token for token in raw if len(token) >= 3 and token not in STOPWORDS]
    unique_terms = list(dict.fromkeys(terms))
    bigrams = [
        (left, right)
        for left, right in zip(raw, raw[1:])
        if left not in STOPWORDS and right not in STOPWORDS and len(left) >= 3 and len(right) >= 3
    ]
    return unique_terms, list(dict.fromkeys(bigrams))


def retrieve_objective_context(
    full_text: str,
    objective: str,
    top_lines: int = 24,
    neighbor_radius: int = 1,
    max_selected_lines: int = 72,
) -> tuple[str, dict[str, float | int]]:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    terms, bigrams = _objective_terms(objective)
    term_set = set(terms)
    scores = np.zeros(len(lines), dtype=np.float64)
    matched_term_union: set[str] = set()

    for index, line in enumerate(lines):
        line_tokens = _tokens(ROLE_PREFIX_RE.sub("", line))
        line_set = set(line_tokens)
        overlap = term_set.intersection(line_set)
        if overlap:
            matched_term_union.update(overlap)
            scores[index] += sum(1.0 + min(len(token), 10) / 20.0 for token in overlap)
        if bigrams and line_tokens:
            line_bigrams = set(zip(line_tokens, line_tokens[1:]))
            scores[index] += 1.5 * sum(pair in line_bigrams for pair in bigrams)

    positive = np.flatnonzero(scores > 0)
    if len(positive):
        ranked = positive[np.argsort(scores[positive])[::-1][:top_lines]]
        selected: set[int] = set()
        for index in ranked:
            selected.update(
                range(max(0, index - neighbor_radius), min(len(lines), index + neighbor_radius + 1))
            )
        selected_indices = sorted(selected)
        if len(selected_indices) > max_selected_lines:
            priority = sorted(selected_indices, key=lambda item: scores[item], reverse=True)
            selected_indices = sorted(priority[:max_selected_lines])
    else:
        fallback = min(18, len(lines))
        selected_indices = sorted(set(range(fallback)).union(range(max(0, len(lines) - fallback), len(lines))))

    selected_lines = [lines[index] for index in selected_indices]
    coverage = len(matched_term_union) / max(1, len(term_set))
    if len(positive):
        first_position = float(positive.min() / max(1, len(lines) - 1))
        last_position = float(positive.max() / max(1, len(lines) - 1))
        mean_positive_score = float(scores[positive].mean())
        max_score = float(scores[positive].max())
    else:
        first_position = -1.0
        last_position = -1.0
        mean_positive_score = 0.0
        max_score = 0.0
    stats: dict[str, float | int] = {
        "retrieval_total_lines": len(lines),
        "retrieval_positive_lines": int(len(positive)),
        "retrieval_selected_lines": len(selected_indices),
        "retrieval_objective_terms": len(term_set),
        "retrieval_matched_terms": len(matched_term_union),
        "retrieval_term_coverage": float(coverage),
        "retrieval_first_match_position": first_position,
        "retrieval_last_match_position": last_position,
        "retrieval_mean_positive_score": mean_positive_score,
        "retrieval_max_score": max_score,
    }
    header = f"[OBJECTIVE] {objective.strip()}"
    return "\n".join([header, *selected_lines]), stats


def build_objective_retrieval_cache(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    output_path = paths.cache_dir / "response_objective_context.parquet"
    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet",
        columns=["response_id", "session_id", "learning_objective"],
    )
    if output_path.exists():
        cached = pd.read_parquet(output_path, columns=["response_id"])
        if list(cached["response_id"]) == list(frame["response_id"]):
            return {"responses": len(cached), "reused": True, "path": str(output_path)}

    sessions = pd.read_parquet(paths.cache_dir / "session_texts.parquet").set_index("session_id")
    rows: list[dict[str, object]] = []
    for index, response in enumerate(frame.itertuples(index=False), start=1):
        full_text = str(sessions.at[response.session_id, "full_text"])
        context, stats = retrieve_objective_context(full_text, str(response.learning_objective))
        rows.append(
            {
                "response_id": response.response_id,
                "session_id": response.session_id,
                "objective_context": context,
                **stats,
            }
        )
        if index % 5000 == 0:
            print(f"Retrieved objective context for {index}/{len(frame)} responses.", flush=True)
    result = pd.DataFrame(rows)
    if list(result["response_id"]) != list(frame["response_id"]):
        raise RuntimeError("Objective-retrieval cache order is invalid.")
    result.to_parquet(output_path, index=False, compression="zstd")
    return {"responses": len(result), "reused": False, "path": str(output_path)}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build objective-conditioned transcript contexts.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_objective_retrieval_cache(args.project_root)
    print(
        f"Objective contexts ready: {result['responses']} responses; "
        f"reused={result['reused']}; path={result['path']}"
    )


if __name__ == "__main__":
    main()
