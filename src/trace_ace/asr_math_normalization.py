from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sentence_transformers import SentenceTransformer


PROTOCOL_ID = "E860_target_free_asr_math_normalization_v1"
SEED = 20260730
NCTE_ROWS = 2_048
COMPETITION_ROWS = 2_048
BENCHMARK_ROWS = 64
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 16
THREADS = 6
EXPECTED_NCTE_SHA256 = (
    "bbfa37ac857991e5d12b1677216896f32d7cf4c2d95b618b67afa7b3bf23b3d7"
)
EXPECTED_CONTEXT_SHA256 = (
    "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6"
)
EXPECTED_MODEL_SHA256 = (
    "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7"
)

FILLERS = frozenset(
    {"um", "uh", "er", "erm", "hmm", "mm", "ah", "okay", "ok", "right", "so", "well"}
)
ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_WORDS = frozenset({*ONES, *TENS, "hundred", "thousand", "and"})
ROLE_RE = re.compile(r"^\s*(\[[A-Za-z_]+\])\s*")
TOKEN_RE = re.compile(
    r"mathop[a-z0-9_]+|\d+(?:\.\d+)?%?|[a-z]+(?:'[a-z]+)?|[+\-*/=<>]",
    re.IGNORECASE,
)
NUMBER_OR_OPERATOR_RE = re.compile(
    r"(?:\d|[+\-*/=<>%]|\b(?:zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
    r"ninety|hundred|thousand|plus|minus|times|multiplied|divided|equals|"
    r"squared|cubed|fraction|percent)\b)",
    re.IGNORECASE,
)
PHRASE_REPLACEMENTS = (
    (re.compile(r"\bsquare\s+root\s+of\b", re.I), " mathopsquareroot "),
    (re.compile(r"\bsquare\s+root\b", re.I), " mathopsquareroot "),
    (re.compile(r"\bmultiplied\s+by\b", re.I), " mathopmultiply "),
    (re.compile(r"\btimes\s+by\b", re.I), " mathopmultiply "),
    (re.compile(r"\bdivided\s+by\b", re.I), " mathopdivide "),
    (re.compile(r"\bshared\s+between\b", re.I), " mathopdivide "),
    (re.compile(r"\btake\s+away\b", re.I), " mathopsubtract "),
    (re.compile(r"\bsubtract(?:ed)?\s+from\b", re.I), " mathopsubtract "),
    (re.compile(r"\bgreater\s+than\b", re.I), " mathopgreater "),
    (re.compile(r"\bless\s+than\b", re.I), " mathopless "),
    (re.compile(r"\bequal\s+to\b", re.I), " mathopequal "),
    (re.compile(r"\bequals?\b", re.I), " mathopequal "),
    (re.compile(r"\bplus\b|\badded?\s+to\b", re.I), " mathopadd "),
    (re.compile(r"\bminus\b|\bsubtract\b", re.I), " mathopsubtract "),
    (re.compile(r"\btimes\b|\bmultiply\b", re.I), " mathopmultiply "),
    (re.compile(r"\bsquared\b", re.I), " mathoppower2 "),
    (re.compile(r"\bcubed\b", re.I), " mathoppower3 "),
)
SYMBOL_OPERATORS = {
    "+": "mathopadd",
    "-": "mathopsubtract",
    "*": "mathopmultiply",
    "/": "mathopdivide",
    "=": "mathopequal",
    "<": "mathopless",
    ">": "mathopgreater",
}
CORRUPT_OPERATORS = {
    "+": " plus ",
    "-": " minus ",
    "*": " times ",
    "/": " divided by ",
    "=": " equals ",
    "<": " less than ",
    ">": " greater than ",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_integer(value: str, namespace: str) -> int:
    payload = f"E860|{namespace}|{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _parse_number_words(tokens: list[str], start: int) -> tuple[int, int] | None:
    index = start
    total = 0
    current = 0
    consumed_numeric = False
    while index < len(tokens):
        token = tokens[index]
        if token in ONES:
            current += ONES[token]
            consumed_numeric = True
        elif token in TENS:
            current += TENS[token]
            consumed_numeric = True
        elif token == "hundred" and consumed_numeric:
            current = max(current, 1) * 100
        elif token == "thousand" and consumed_numeric:
            total += max(current, 1) * 1_000
            current = 0
        elif token == "and" and consumed_numeric:
            pass
        else:
            break
        index += 1
    if not consumed_numeric:
        return None
    return total + current, index


def normalize_math_speech(text: object) -> str:
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [str(text)]:
        role_match = ROLE_RE.match(raw_line)
        role = role_match.group(1).casefold() if role_match else ""
        content = raw_line[role_match.end() :] if role_match else raw_line
        content = re.sub(r"\[unclear\]", " ", content, flags=re.I)
        for pattern, replacement in PHRASE_REPLACEMENTS:
            content = pattern.sub(replacement, content)
        tokens = [token.casefold() for token in TOKEN_RE.findall(content)]
        while tokens and tokens[0] in FILLERS:
            tokens.pop(0)
        collapsed: list[str] = []
        for token in tokens:
            if not collapsed or collapsed[-1] != token:
                collapsed.append(token)
        output: list[str] = [role] if role else []
        index = 0
        while index < len(collapsed):
            token = collapsed[index]
            parsed = (
                _parse_number_words(collapsed, index)
                if token in NUMBER_WORDS
                else None
            )
            if parsed is not None:
                value, index = parsed
                output.append(f"mathnum{value}")
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?%?", token):
                value = token.replace(".", "point").replace("%", "percent")
                output.append(f"mathnum{value}")
            elif token in SYMBOL_OPERATORS:
                output.append(SYMBOL_OPERATORS[token])
            else:
                output.append(token)
            index += 1
        if output:
            lines.append(" ".join(output))
    return "\n".join(lines)


def _number_to_words(value: int) -> str:
    if value < 20:
        return next(name for name, number in ONES.items() if number == value)
    if value < 100:
        tens, remainder = divmod(value, 10)
        tens_name = next(name for name, number in TENS.items() if number == tens * 10)
        return tens_name if remainder == 0 else f"{tens_name} {_number_to_words(remainder)}"
    if value < 1_000:
        hundreds, remainder = divmod(value, 100)
        prefix = f"{_number_to_words(hundreds)} hundred"
        return prefix if remainder == 0 else f"{prefix} and {_number_to_words(remainder)}"
    return " ".join(_number_to_words(int(digit)) for digit in str(value))


def corrupt_asr_like(text: object, key: str) -> str:
    value = str(text)

    def verbalize(match: re.Match[str]) -> str:
        raw = match.group(0)
        if "." in raw or "%" in raw:
            return raw
        return _number_to_words(int(raw))

    value = re.sub(r"\b\d+\b", verbalize, value)
    for symbol, phrase in CORRUPT_OPERATORS.items():
        value = value.replace(symbol, phrase)
    words = value.split()
    if words:
        duplicate_index = _stable_integer(key, "duplicate") % len(words)
        words.insert(duplicate_index, words[duplicate_index])
        unclear_index = _stable_integer(key, "unclear") % (len(words) + 1)
        words.insert(unclear_index, "[unclear]")
    return "Um, uh, " + " ".join(words)


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(left.casefold()))
    right_tokens = set(TOKEN_RE.findall(right.casefold()))
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _ncte_path(root: Path) -> Path:
    return (
        root
        / "Datasets"
        / "NCTE"
        / "release_19LzXF0IRtOGO62ZUnJeX6rBTjynuW6RR"
        / "ncte_single_utterances.csv"
    )


def _model_path(root: Path) -> Path:
    return root / "submission_src" / "assets" / "bge-base-en-v1.5"


def _load_ncte_sample(root: Path, rows: int) -> pd.DataFrame:
    source = pd.read_csv(
        _ncte_path(root),
        usecols=["speaker", "text", "video_id", "turn_idx", "num_words"],
    )
    eligible = source.loc[
        source["num_words"].fillna(0).astype(float).between(5, 80)
        & source["text"].fillna("").astype(str).str.contains(
            NUMBER_OR_OPERATOR_RE, regex=True
        )
    ].copy()
    eligible["row_key"] = (
        eligible["video_id"].astype(str)
        + "|"
        + eligible["turn_idx"].astype(str)
    )
    eligible["_order"] = eligible["row_key"].map(
        lambda value: _stable_integer(value, "ncte_sample")
    )
    sample = eligible.sort_values("_order", kind="mergesort").head(rows).copy()
    if len(sample) != rows:
        raise ValueError("E860 NCTE sample is smaller than frozen size.")
    sample["clean_text"] = sample["text"].fillna("").astype(str)
    sample["corrupted_text"] = [
        corrupt_asr_like(text, key)
        for text, key in zip(sample["clean_text"], sample["row_key"], strict=True)
    ]
    sample["normalized_clean"] = sample["clean_text"].map(normalize_math_speech)
    sample["normalized_corrupted"] = sample["corrupted_text"].map(
        normalize_math_speech
    )
    return sample.reset_index(drop=True)


def _parse_context(value: object) -> tuple[str, str]:
    lines = [line for line in str(value).splitlines() if line.strip()]
    if not lines or not lines[0].startswith("[OBJECTIVE]"):
        raise ValueError("E860 expected an objective header.")
    objective = lines[0][len("[OBJECTIVE]") :].strip()
    return objective, "\n".join(lines[1:])


def _load_competition_sample(root: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        root / "data_cache" / "response_objective_context.parquet",
        columns=[
            "response_id",
            "objective_context",
            "retrieval_term_coverage",
        ],
    )
    frame["_coverage_bin"] = pd.qcut(
        frame["retrieval_term_coverage"].rank(method="first"),
        q=4,
        labels=False,
    )
    frame["_order"] = frame["response_id"].astype(str).map(
        lambda value: _stable_integer(value, "competition_sample")
    )
    per_bin = rows // 4
    chunks = [
        group.sort_values("_order", kind="mergesort").head(per_bin)
        for _, group in frame.groupby("_coverage_bin", sort=True)
    ]
    sample = pd.concat(chunks, ignore_index=True)
    if len(sample) != rows:
        raise ValueError("E860 competition sample is smaller than frozen size.")
    parsed = sample["objective_context"].map(_parse_context)
    sample["objective"] = parsed.map(lambda value: value[0])
    sample["transcript_context"] = parsed.map(lambda value: value[1])
    sample["normalized_objective"] = sample["objective"].map(normalize_math_speech)
    sample["normalized_transcript_context"] = sample["transcript_context"].map(
        normalize_math_speech
    )
    return sample


def _load_model(root: Path) -> SentenceTransformer:
    import torch

    torch.set_num_threads(THREADS)
    model = SentenceTransformer(str(_model_path(root)), device="cpu")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    return model


def _encode(model: SentenceTransformer, texts: pd.Series | list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1, dtype=np.float64)


def _evaluate(
    root: Path,
    ncte_rows: int,
    competition_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ncte = _load_ncte_sample(root, ncte_rows)
    competition = _load_competition_sample(root, competition_rows)
    model = _load_model(root)

    raw_clean = _encode(model, ncte["clean_text"])
    raw_corrupted = _encode(model, ncte["corrupted_text"])
    normalized_clean = _encode(model, ncte["normalized_clean"])
    normalized_corrupted = _encode(model, ncte["normalized_corrupted"])
    ncte["raw_cosine"] = _cosine(raw_clean, raw_corrupted)
    ncte["normalized_cosine"] = _cosine(
        normalized_clean, normalized_corrupted
    )
    ncte["cosine_gain"] = ncte["normalized_cosine"] - ncte["raw_cosine"]
    ncte["raw_token_jaccard"] = [
        _jaccard(left, right)
        for left, right in zip(
            ncte["clean_text"], ncte["corrupted_text"], strict=True
        )
    ]
    ncte["normalized_token_jaccard"] = [
        _jaccard(left, right)
        for left, right in zip(
            ncte["normalized_clean"],
            ncte["normalized_corrupted"],
            strict=True,
        )
    ]
    ncte["token_jaccard_gain"] = (
        ncte["normalized_token_jaccard"] - ncte["raw_token_jaccard"]
    )

    raw_objective = _encode(model, competition["objective"])
    raw_context = _encode(model, competition["transcript_context"])
    normalized_objective = _encode(model, competition["normalized_objective"])
    normalized_context = _encode(
        model, competition["normalized_transcript_context"]
    )
    competition["raw_objective_context_cosine"] = _cosine(
        raw_objective, raw_context
    )
    competition["normalized_objective_context_cosine"] = _cosine(
        normalized_objective, normalized_context
    )
    competition["objective_context_cosine_gain"] = (
        competition["normalized_objective_context_cosine"]
        - competition["raw_objective_context_cosine"]
    )
    competition["objective_semantic_preservation"] = _cosine(
        raw_objective, normalized_objective
    )
    competition["context_semantic_preservation"] = _cosine(
        raw_context, normalized_context
    )

    lowest = competition["_coverage_bin"].eq(0)
    metrics: dict[str, object] = {
        "ncte": {
            "rows": int(len(ncte)),
            "videos": int(ncte["video_id"].nunique()),
            "exact_normalized_recovery_fraction": float(
                np.mean(
                    ncte["normalized_clean"]
                    == ncte["normalized_corrupted"]
                )
            ),
            "raw_token_jaccard_mean": float(ncte["raw_token_jaccard"].mean()),
            "normalized_token_jaccard_mean": float(
                ncte["normalized_token_jaccard"].mean()
            ),
            "token_jaccard_gain_mean": float(
                ncte["token_jaccard_gain"].mean()
            ),
            "raw_corruption_cosine_mean": float(ncte["raw_cosine"].mean()),
            "normalized_corruption_cosine_mean": float(
                ncte["normalized_cosine"].mean()
            ),
            "corruption_cosine_gain_mean": float(ncte["cosine_gain"].mean()),
            "rows_with_positive_cosine_gain_fraction": float(
                np.mean(ncte["cosine_gain"] > 0.0)
            ),
        },
        "competition": {
            "rows": int(len(competition)),
            "changed_objective_fraction": float(
                np.mean(
                    competition["objective"]
                    != competition["normalized_objective"]
                )
            ),
            "changed_context_fraction": float(
                np.mean(
                    competition["transcript_context"]
                    != competition["normalized_transcript_context"]
                )
            ),
            "objective_semantic_preservation_mean": float(
                competition["objective_semantic_preservation"].mean()
            ),
            "context_semantic_preservation_mean": float(
                competition["context_semantic_preservation"].mean()
            ),
            "raw_objective_context_cosine_mean": float(
                competition["raw_objective_context_cosine"].mean()
            ),
            "normalized_objective_context_cosine_mean": float(
                competition["normalized_objective_context_cosine"].mean()
            ),
            "objective_context_cosine_gain_mean": float(
                competition["objective_context_cosine_gain"].mean()
            ),
            "positive_objective_context_cosine_gain_fraction": float(
                np.mean(competition["objective_context_cosine_gain"] > 0.0)
            ),
            "lowest_coverage_quartile_cosine_gain_mean": float(
                competition.loc[
                    lowest, "objective_context_cosine_gain"
                ].mean()
            ),
        },
    }
    return ncte, competition, metrics


def _gate(metrics: dict[str, object]) -> dict[str, bool]:
    ncte = metrics["ncte"]
    competition = metrics["competition"]
    assert isinstance(ncte, dict)
    assert isinstance(competition, dict)
    return {
        "ncte_at_least_100_videos": int(ncte["videos"]) >= 100,
        "exact_normalized_recovery_at_least_0_90": (
            float(ncte["exact_normalized_recovery_fraction"]) >= 0.90
        ),
        "token_jaccard_gain_at_least_0_20": (
            float(ncte["token_jaccard_gain_mean"]) >= 0.20
        ),
        "corruption_cosine_gain_at_least_0_02": (
            float(ncte["corruption_cosine_gain_mean"]) >= 0.02
        ),
        "normalized_corruption_cosine_at_least_0_95": (
            float(ncte["normalized_corruption_cosine_mean"]) >= 0.95
        ),
        "positive_corruption_cosine_gain_fraction_at_least_0_90": (
            float(ncte["rows_with_positive_cosine_gain_fraction"]) >= 0.90
        ),
        "changed_objective_fraction_at_least_0_25": (
            float(competition["changed_objective_fraction"]) >= 0.25
        ),
        "changed_context_fraction_at_least_0_90": (
            float(competition["changed_context_fraction"]) >= 0.90
        ),
        "objective_semantic_preservation_at_least_0_95": (
            float(competition["objective_semantic_preservation_mean"]) >= 0.95
        ),
        "context_semantic_preservation_at_least_0_90": (
            float(competition["context_semantic_preservation_mean"]) >= 0.90
        ),
        "objective_context_cosine_gain_at_least_0_005": (
            float(competition["objective_context_cosine_gain_mean"]) >= 0.005
        ),
        "positive_objective_context_gain_fraction_at_least_0_60": (
            float(
                competition["positive_objective_context_cosine_gain_fraction"]
            )
            >= 0.60
        ),
        "lowest_coverage_cosine_gain_at_least_0_007": (
            float(competition["lowest_coverage_quartile_cosine_gain_mean"])
            >= 0.007
        ),
    }


def _verify_sources(root: Path) -> dict[str, str]:
    sources = {
        "ncte": _sha256(_ncte_path(root)),
        "competition_context": _sha256(
            root / "data_cache" / "response_objective_context.parquet"
        ),
        "bge_base_model": _sha256(
            _model_path(root) / "model.safetensors"
        ),
    }
    expected = {
        "ncte": EXPECTED_NCTE_SHA256,
        "competition_context": EXPECTED_CONTEXT_SHA256,
        "bge_base_model": EXPECTED_MODEL_SHA256,
    }
    if sources != expected:
        raise ValueError(f"E860 source hashes changed: {sources}")
    return sources


def benchmark(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    _verify_sources(root)
    _, _, metrics = _evaluate(root, BENCHMARK_ROWS, BENCHMARK_ROWS)
    elapsed = time.perf_counter() - started
    scale = (NCTE_ROWS + COMPETITION_ROWS) / (2 * BENCHMARK_ROWS)
    projected = elapsed * scale * 1.25
    return {
        "protocol_id": PROTOCOL_ID,
        "benchmark_rows_per_source": BENCHMARK_ROWS,
        "elapsed_seconds": elapsed,
        "projected_full_seconds_upper_bound": projected,
        "projected_full_hours_upper_bound": projected / 3600.0,
        "projected_below_one_hour": projected < 3600.0,
        "benchmark_metrics": metrics,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def run(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    sources = _verify_sources(root)
    ncte, competition, metrics = _evaluate(
        root, NCTE_ROWS, COMPETITION_ROWS
    )
    clauses = _gate(metrics)
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_asr_math_normalization"
    )
    run_dir = root / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ncte_path = run_dir / "ncte_target_free_scores.parquet"
    competition_path = run_dir / "competition_target_free_scores.parquet"
    report_path = run_dir / "report.json"
    ncte[
        [
            "row_key",
            "video_id",
            "raw_token_jaccard",
            "normalized_token_jaccard",
            "raw_cosine",
            "normalized_cosine",
            "cosine_gain",
        ]
    ].to_parquet(ncte_path, index=False)
    competition[
        [
            "response_id",
            "retrieval_term_coverage",
            "_coverage_bin",
            "raw_objective_context_cosine",
            "normalized_objective_context_cosine",
            "objective_context_cosine_gain",
            "objective_semantic_preservation",
            "context_semantic_preservation",
        ]
    ].to_parquet(competition_path, index=False)
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "configuration": {
            "seed": SEED,
            "ncte_rows": NCTE_ROWS,
            "competition_rows": COMPETITION_ROWS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "threads": THREADS,
            "ambiguous_homophone_correction": False,
        },
        "metrics": metrics,
        "gate_clauses": clauses,
        "passes_target_free_gate": bool(all(clauses.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            "ncte_target_free_scores": {
                "path": str(ncte_path.relative_to(root)),
                "sha256": _sha256(ncte_path),
                "rows": int(len(ncte)),
            },
            "competition_target_free_scores": {
                "path": str(competition_path.relative_to(root)),
                "sha256": _sha256(competition_path),
                "rows": int(len(competition)),
            },
        },
        "competition_outcomes_accessed": False,
        "V_seen_accessed": False,
        "V_objective_accessed": False,
        "V_style_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        "submission_built": False,
        "competition_upload_or_submission": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "gate_clauses": clauses,
        "passes_target_free_gate": report["passes_target_free_gate"],
        "artifact_sha256": {
            name: value["sha256"]
            for name, value in report["artifacts"].items()
        },
        "report_sha256": _sha256(report_path),
        "runtime_seconds": report["runtime_seconds"],
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E860 target-free ASR math normalization."
    )
    parser.add_argument("stage", choices=("benchmark", "run"))
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    result = (
        benchmark(args.project_root)
        if args.stage == "benchmark"
        else run(args.project_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
