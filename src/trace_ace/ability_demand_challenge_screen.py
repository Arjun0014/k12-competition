from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from scipy.special import expit

from trace_ace.bge_base_multiview_screen import assert_runtime
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.objective_retrieval import _objective_terms
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E780_ability_demand_challenge_v1"
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
SEED = 20260728
MAX_EPISODE_TURNS = 10
RASCH_SLOPE_L2 = 0.1
RASCH_MAX_SLOPE = 5.0
RASCH_MAX_ITERATIONS = 200
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6
MAX_PROJECTED_SECONDS = 3_600.0
MIN_RESPONSE_EPISODE_COVERAGE = 0.90
MIN_STYLE_CELL_EPISODE_COVERAGE = 0.70
MIN_STATE_EPISODES = 50
EXPECTED_FEATURE_CONTENT_SHA256 = ""
EXPECTED_FEATURE_PARQUET_SHA256 = ""

EXPECTED_SOURCE_SHA256 = {
    "modeling_base.parquet": (
        "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
    ),
    "response_objective_context.parquet": (
        "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6"
    ),
    "validation_environments_selection.parquet": (
        "600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40"
    ),
    "component_oof_V_seen.parquet": (
        "1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af"
    ),
    "component_oof_V_objective.parquet": (
        "c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7"
    ),
    "component_oof_V_style.parquet": (
        "6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038"
    ),
}

STATE_NAMES = ("under_challenged", "optimally_challenged", "over_challenged")
FEATURE_NAMES = (
    "episode_count_log1p",
    "ability_mean",
    "ability_first",
    "ability_last",
    "ability_shift",
    "demand_mean",
    "demand_first",
    "demand_last",
    "demand_shift",
    "local_gap_mean",
    "local_gap_std",
    "local_gap_first",
    "local_gap_last",
    "local_gap_shift",
    "under_challenged_proportion",
    "optimally_challenged_proportion",
    "over_challenged_proportion",
    "struggle_proportion",
    "recovery_proportion",
    "direct_answer_proportion",
    "unresolved_proportion",
    "rasch_success_mean",
    "session_ability_theta",
    "session_demand_d",
    "rasch_gap",
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NOISE_PATTERN = re.compile(
    r"\[(?:unclear|inaudible|speaker:[^\]]+)\]|\b(?:uh+|um+|erm+|hmm+)\b",
    re.IGNORECASE,
)
TASK_QUESTION_RE = re.compile(
    r"\b(?:what|why|how|which|where|when|can you|could you|"
    r"tell me|show me|explain|solve|calculate|find|work out|"
    r"write|read|complete|choose|try)\b"
)
TASK_DIRECTIVE_RE = re.compile(
    r"^(?:please )?(?:solve|calculate|find|work out|write|read|"
    r"complete|choose|try|show|explain)\b"
)
REASONING_DEMAND_RE = re.compile(
    r"\b(?:why|explain|justify|prove|reason|how do you know|"
    r"compare|what makes)\b"
)
MULTISTEP_DEMAND_RE = re.compile(
    r"\b(?:first|then|next|after that|steps?|part [a-d]|"
    r"two things|both|each)\b"
)
REPRESENTATION_DEMAND_RE = re.compile(
    r"\b(?:diagram|table|graph|number line|equation|model|"
    r"bar model|draw|represent|in words)\b"
)
OPERATION_RE = re.compile(
    r"\b(?:add|subtract|multiply|divide|fraction|decimal|"
    r"ratio|percent|area|perimeter|angle|equation)\b|[+\-*/=]"
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:i don'?t know|not sure|unsure|confused|maybe|"
    r"i guess|can'?t|cannot|no idea|need help)\b"
)
PROCEDURE_CONFUSION_RE = re.compile(
    r"\b(?:what should i (?:type|write|click)|where do i put|"
    r"should i type|do i write|which button)\b"
)
REASONING_RE = re.compile(
    r"\b(?:because|therefore|since|which means|that means|"
    r"first|then|next|after that|i think|the reason)\b"
)
SELF_CORRECTION_RE = re.compile(
    r"\b(?:actually|i mean|no wait|wait no|i made a mistake|"
    r"sorry i meant|instead|rather)\b"
)
CORRECT_FEEDBACK_RE = re.compile(
    r"\b(?:correct|exactly|that'?s right|that is right|"
    r"you got it|well done|perfect)\b"
)
INCORRECT_FEEDBACK_RE = re.compile(
    r"\b(?:incorrect|wrong|not quite|that is not right|"
    r"that'?s not right|almost but|try again|check your)\b"
)
HINT_RE = re.compile(
    r"\b(?:hint|remember|think about|what if|look at|"
    r"start by|consider|check your|try again)\b"
)
DIRECT_ANSWER_RE = re.compile(
    r"\b(?:the answer is|the solution is|it equals|"
    r"you get|that gives us|it is)\s+(?:minus\s+)?\d"
)
MATH_WORDS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "equals",
    "equal",
    "fraction",
    "decimal",
    "number",
    "tens",
    "ones",
    "hundreds",
    "ratio",
    "percent",
    "angle",
    "area",
    "perimeter",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    source_run = paths.experiments_dir / "runs" / SOURCE_RUN_ID
    return {
        "modeling_base.parquet": paths.cache_dir / "modeling_base.parquet",
        "response_objective_context.parquet": (
            paths.cache_dir / "response_objective_context.parquet"
        ),
        "validation_environments_selection.parquet": (
            paths.cache_dir / "validation_environments_selection.parquet"
        ),
        **{
            f"component_oof_{environment}.parquet": (
                source_run / f"component_oof_{environment}.parquet"
            )
            for environment in SELECTION_ENVIRONMENTS
        },
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    actual = {
        name: _sha256(path)
        for name, path in _source_paths(project_root).items()
    }
    if actual != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"E780 immutable source verification failed: {actual}"
        )
    return actual


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    return {
        "features": paths.cache_dir / "ability_demand_e780.parquet",
        "metadata": paths.cache_dir / "ability_demand_e780.metadata.json",
        "benchmark": paths.cache_dir / "ability_demand_e780_benchmark.json",
    }


def _normalize(text: object) -> str:
    value = NOISE_PATTERN.sub(" ", str(text).lower())
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: object) -> list[str]:
    return TOKEN_PATTERN.findall(_normalize(text))


def parse_context(context: object) -> tuple[str, list[tuple[str, str]]]:
    objective = ""
    turns: list[tuple[str, str]] = []
    for raw_line in str(context).splitlines():
        line = raw_line.strip()
        if line.startswith("[OBJECTIVE]"):
            objective = line[len("[OBJECTIVE]") :].strip()
        elif line.startswith("[STUDENT]"):
            content = line[len("[STUDENT]") :].strip()
            if _normalize(content):
                turns.append(("student", content))
        elif line.startswith("[TUTOR]"):
            content = line[len("[TUTOR]") :].strip()
            if _normalize(content):
                turns.append(("tutor", content))
    return objective, turns


def is_task_prompt(text: object) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    return bool(
        ("?" in normalized and TASK_QUESTION_RE.search(normalized))
        or TASK_DIRECTIVE_RE.search(normalized)
    )


def segment_task_episodes(
    turns: Sequence[tuple[str, str]]
) -> list[list[tuple[str, str]]]:
    episodes: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] | None = None
    for role, text in turns:
        if role == "tutor" and is_task_prompt(text):
            if current is not None:
                episodes.append(current[: MAX_EPISODE_TURNS + 1])
            current = [(role, text)]
        elif current is not None and len(current) <= MAX_EPISODE_TURNS:
            current.append((role, text))
    if current is not None:
        episodes.append(current[: MAX_EPISODE_TURNS + 1])
    return episodes


def _sigmoid(value: float) -> float:
    return float(expit(np.clip(value, -20.0, 20.0)))


def estimate_task_demand(prompt: object, objective: object) -> float:
    prompt_text = _normalize(prompt)
    objective_text = _normalize(objective)
    prompt_tokens = _tokens(prompt_text)
    objective_tokens = _tokens(objective_text)
    operation_count = len(set(OPERATION_RE.findall(prompt_text)))
    raw = (
        -1.20
        + 0.20 * math.log1p(len(prompt_tokens))
        + 0.10 * math.log1p(len(objective_tokens))
        + 0.65 * float(bool(REASONING_DEMAND_RE.search(prompt_text)))
        + 0.45 * float(bool(MULTISTEP_DEMAND_RE.search(prompt_text)))
        + 0.45 * float(bool(REPRESENTATION_DEMAND_RE.search(prompt_text)))
        + 0.25 * min(operation_count, 3)
    )
    return _sigmoid(raw)


def _student_attempt_evidence(
    text: object,
    objective: object,
    *,
    correct_feedback: bool,
    incorrect_feedback: bool,
) -> float:
    normalized = _normalize(text)
    tokens = _tokens(normalized)
    token_set = set(tokens)
    objective_terms, _ = _objective_terms(str(objective))
    objective_overlap = len(token_set & set(objective_terms)) / max(
        1, len(set(objective_terms))
    )
    math_evidence = bool(
        token_set & MATH_WORDS
        or re.search(r"(?:\d|[+\-*/=])", normalized)
    )
    raw = (
        -0.70
        + 0.32 * min(math.log1p(len(tokens)), 3.0)
        + 0.65 * float(bool(REASONING_RE.search(normalized)))
        + 0.60 * min(objective_overlap, 1.0)
        + 0.30 * float(math_evidence)
        + 0.80 * float(correct_feedback)
        - 0.85 * float(incorrect_feedback)
        - 0.75 * float(bool(UNCERTAINTY_RE.search(normalized)))
        + 0.35 * float(bool(SELF_CORRECTION_RE.search(normalized)))
    )
    return _sigmoid(raw)


def analyze_episode(
    episode: Sequence[tuple[str, str]], objective: object
) -> dict[str, object]:
    if not episode or episode[0][0] != "tutor":
        raise ValueError("E780 episode must begin with a tutor task prompt.")
    prompt = episode[0][1]
    student_indices = [
        index for index, (role, _) in enumerate(episode) if role == "student"
    ]
    direct_answer = any(
        role == "tutor" and DIRECT_ANSWER_RE.search(_normalize(text))
        for role, text in episode[1:]
    )
    hint = any(
        role == "tutor" and HINT_RE.search(_normalize(text))
        for role, text in episode[1:]
    )
    attempts: list[float] = []
    attempt_details: list[dict[str, object]] = []
    for index in student_indices:
        text = episode[index][1]
        following_tutor = ""
        for role, following_text in episode[index + 1 :]:
            if role == "student":
                break
            if role == "tutor":
                following_tutor += " " + _normalize(following_text)
        correct = bool(CORRECT_FEEDBACK_RE.search(following_tutor))
        incorrect = bool(INCORRECT_FEEDBACK_RE.search(following_tutor))
        evidence = _student_attempt_evidence(
            text,
            objective,
            correct_feedback=correct,
            incorrect_feedback=incorrect,
        )
        attempts.append(evidence)
        attempt_details.append(
            {
                "text": text,
                "correct_feedback": correct,
                "incorrect_feedback": incorrect,
                "uncertainty": bool(UNCERTAINTY_RE.search(_normalize(text))),
                "procedure_confusion": bool(
                    PROCEDURE_CONFUSION_RE.search(_normalize(text))
                ),
                "self_correction": bool(
                    SELF_CORRECTION_RE.search(_normalize(text))
                ),
                "evidence": evidence,
            }
        )
    correct_observed = any(
        bool(detail["correct_feedback"]) for detail in attempt_details
    )
    incorrect_observed = any(
        bool(detail["incorrect_feedback"]) for detail in attempt_details
    )
    content_confusion = any(
        bool(detail["uncertainty"]) and not bool(detail["procedure_confusion"])
        for detail in attempt_details
    )
    self_correction = any(
        bool(detail["self_correction"]) for detail in attempt_details
    )
    robust_unconfirmed_success = bool(
        attempts
        and attempts[-1] >= 0.66
        and not incorrect_observed
        and not direct_answer
    )
    success = correct_observed or robust_unconfirmed_success
    struggle = incorrect_observed or content_confusion
    recovery = bool(
        success
        and struggle
        and (
            len(attempts) >= 2
            or self_correction
            or hint
        )
    )
    unresolved = not success
    if direct_answer or unresolved:
        state = "over_challenged"
    elif struggle and recovery:
        state = "optimally_challenged"
    else:
        state = "under_challenged"
    ability = (
        0.60 * attempts[-1] + 0.40 * float(np.mean(attempts))
        if attempts
        else 0.0
    )
    if recovery:
        ability = min(1.0, ability + 0.10)
    demand = estimate_task_demand(prompt, objective)
    return {
        "state": state,
        "ability": float(ability),
        "demand": float(demand),
        "local_gap": float(ability - demand),
        "struggle": float(struggle),
        "recovery": float(recovery),
        "direct_answer": float(direct_answer),
        "unresolved": float(unresolved),
        "hint": float(hint),
        "student_attempts": len(attempts),
    }


def aggregate_episodes(
    episodes: Sequence[dict[str, object]]
) -> dict[str, float]:
    count = len(episodes)
    ability = np.asarray(
        [float(episode["ability"]) for episode in episodes],
        dtype=np.float64,
    )
    demand = np.asarray(
        [float(episode["demand"]) for episode in episodes],
        dtype=np.float64,
    )
    gap = ability - demand

    def first(values: np.ndarray) -> float:
        return float(values[0]) if len(values) else 0.0

    def last(values: np.ndarray) -> float:
        return float(values[-1]) if len(values) else 0.0

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else 0.0

    def proportion(name: str) -> float:
        return (
            sum(episode["state"] == name for episode in episodes)
            / max(1, count)
        )

    under = proportion("under_challenged")
    optimal = proportion("optimally_challenged")
    over = proportion("over_challenged")
    struggle = (
        sum(float(episode["struggle"]) for episode in episodes)
        / max(1, count)
    )
    recovery = (
        sum(float(episode["recovery"]) for episode in episodes)
        / max(1, count)
    )
    direct_answer = (
        sum(float(episode["direct_answer"]) for episode in episodes)
        / max(1, count)
    )
    unresolved = (
        sum(float(episode["unresolved"]) for episode in episodes)
        / max(1, count)
    )
    ability_shift = last(ability) - first(ability) if count else 0.0
    demand_shift = last(demand) - first(demand) if count else 0.0
    gap_shift = last(gap) - first(gap) if count else 0.0
    theta = mean(ability) + 0.25 * ability_shift + 0.20 * optimal + 0.15 * recovery
    difficulty = mean(demand) + 0.50 * over + 0.20 * unresolved
    rasch_gap = theta - difficulty
    values = {
        "episode_count_log1p": math.log1p(count),
        "ability_mean": mean(ability),
        "ability_first": first(ability),
        "ability_last": last(ability),
        "ability_shift": ability_shift,
        "demand_mean": mean(demand),
        "demand_first": first(demand),
        "demand_last": last(demand),
        "demand_shift": demand_shift,
        "local_gap_mean": mean(gap),
        "local_gap_std": float(gap.std(ddof=0)) if count else 0.0,
        "local_gap_first": first(gap),
        "local_gap_last": last(gap),
        "local_gap_shift": gap_shift,
        "under_challenged_proportion": under,
        "optimally_challenged_proportion": optimal,
        "over_challenged_proportion": over,
        "struggle_proportion": float(struggle),
        "recovery_proportion": float(recovery),
        "direct_answer_proportion": float(direct_answer),
        "unresolved_proportion": float(unresolved),
        "rasch_success_mean": (
            float(expit(gap).mean()) if count else 0.5
        ),
        "session_ability_theta": float(theta),
        "session_demand_d": float(difficulty),
        "rasch_gap": float(rasch_gap),
    }
    if tuple(values) != FEATURE_NAMES or not np.isfinite(list(values.values())).all():
        raise RuntimeError("E780 aggregate feature contract failed.")
    return values


def extract_context_features(context: object) -> dict[str, float]:
    objective, turns = parse_context(context)
    raw_episodes = segment_task_episodes(turns)
    analyzed = [
        analyze_episode(episode, objective)
        for episode in raw_episodes
    ]
    return aggregate_episodes(analyzed)


def _synthetic_episode(
    objective: str, lines: Sequence[tuple[str, str]]
) -> dict[str, object]:
    return analyze_episode(lines, objective)


def synthetic_gate() -> dict[str, object]:
    objective = "Adding and subtracting two digit numbers"
    under = _synthetic_episode(
        objective,
        [
            ("tutor", "What is 24 plus 13?"),
            ("student", "I add the tens and ones because 20 plus 10 is 30, so it is 37."),
            ("tutor", "Exactly, that is correct."),
        ],
    )
    optimal = _synthetic_episode(
        objective,
        [
            ("tutor", "Explain why 24 plus 13 equals 37."),
            ("student", "I am not sure, maybe 35."),
            ("tutor", "Not quite. Here is a hint: add the tens first."),
            ("student", "Actually, 20 plus 10 is 30 and 4 plus 3 is 7, so it is 37."),
            ("tutor", "Correct, well done."),
        ],
    )
    over = _synthetic_episode(
        objective,
        [
            ("tutor", "Prove using a diagram and an equation why 24 plus 13 equals 37."),
            ("student", "I do not know."),
            ("tutor", "The answer is 37."),
        ],
    )
    low_ability = _synthetic_episode(
        objective,
        [
            ("tutor", "What is 24 plus 13?"),
            ("student", "I do not know."),
            ("tutor", "That is not right."),
        ],
    )
    medium_ability = _synthetic_episode(
        objective,
        [
            ("tutor", "What is 24 plus 13?"),
            ("student", "Maybe 37."),
            ("tutor", "Correct."),
        ],
    )
    high_ability = under
    easy_demand = estimate_task_demand("What is 2 plus 2?", objective)
    medium_demand = estimate_task_demand(
        "Calculate 24 plus 13 and write an equation showing the answer.",
        objective,
    )
    hard_demand = estimate_task_demand(
        "Explain and justify with a diagram and equation why both steps work.",
        objective,
    )
    single_summary = aggregate_episodes([optimal])
    duplicate_summary = aggregate_episodes([optimal, optimal])
    invariance_fields = (
        "ability_mean",
        "demand_mean",
        "under_challenged_proportion",
        "optimally_challenged_proportion",
        "over_challenged_proportion",
        "rasch_gap",
    )
    clauses = {
        "under_state_recovered": under["state"] == "under_challenged",
        "optimal_state_recovered": optimal["state"] == "optimally_challenged",
        "over_state_recovered": over["state"] == "over_challenged",
        "ability_order_is_strict": (
            high_ability["ability"]
            > medium_ability["ability"]
            > low_ability["ability"]
        ),
        "ability_adjacent_margin_at_least_0_10": (
            min(
                high_ability["ability"] - medium_ability["ability"],
                medium_ability["ability"] - low_ability["ability"],
            )
            >= 0.10
        ),
        "demand_order_is_strict": hard_demand > medium_demand > easy_demand,
        "demand_adjacent_margin_at_least_0_05": (
            min(hard_demand - medium_demand, medium_demand - easy_demand)
            >= 0.05
        ),
        "rasch_probability_increases_with_ability": bool(
            expit(high_ability["ability"] - medium_demand)
            > expit(low_ability["ability"] - medium_demand)
        ),
        "rasch_probability_decreases_with_demand": bool(
            expit(medium_ability["ability"] - easy_demand)
            > expit(medium_ability["ability"] - hard_demand)
        ),
        "session_duplication_invariant": all(
            np.isclose(single_summary[name], duplicate_summary[name])
            for name in invariance_fields
        ),
        "feature_schema_is_frozen": len(FEATURE_NAMES) == 25,
    }
    return {
        "episodes": {
            "under": under,
            "optimal": optimal,
            "over": over,
            "low_ability": low_ability,
            "medium_ability": medium_ability,
            "high_ability": high_ability,
        },
        "demand": {
            "easy": easy_demand,
            "medium": medium_demand,
            "hard": hard_demand,
        },
        "clauses": clauses,
        "passes": bool(all(clauses.values())),
    }


def benchmark(project_root: str | Path) -> dict[str, object]:
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    paths = discover_project_paths(project_root)
    contexts = pd.read_parquet(
        paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "objective_context"],
    )
    order = contexts["response_id"].astype(str).map(
        lambda value: hashlib.sha256(
            f"E780-benchmark|{value}".encode()
        ).hexdigest()
    )
    sample = contexts.assign(_order=order).sort_values("_order").head(1_024)
    started = time.perf_counter()
    rows = [
        extract_context_features(context)
        for context in sample["objective_context"]
    ]
    elapsed = time.perf_counter() - started
    matrix = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    projected = elapsed * len(contexts) / len(sample)
    synthetic = synthetic_gate()
    clauses = {
        "runtime_is_python_3_12_8": runtime["python"] == "3.12.8",
        "runtime_is_sklearn_1_8_0": runtime["scikit_learn"] == "1.8.0",
        "synthetic_gate_passes": synthetic["passes"],
        "benchmark_features_finite": bool(np.isfinite(matrix).all()),
        "projected_seconds_at_most_3600": projected <= MAX_PROJECTED_SECONDS,
        "competition_outcomes_accessed_is_false": True,
    }
    result: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
        "sample_rows": len(sample),
        "full_rows": len(contexts),
        "feature_count": len(FEATURE_NAMES),
        "elapsed_seconds": elapsed,
        "projected_seconds": projected,
        "synthetic": synthetic,
        "clauses": clauses,
        "proceed": bool(all(clauses.values())),
        "runtime": runtime,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    path = _paths(project_root)["benchmark"]
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _feature_content_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["response_id", *FEATURE_NAMES]].itertuples(
        index=False, name=None
    ):
        digest.update(str(row[0]).encode("utf-8"))
        digest.update(np.asarray(row[1:], dtype="<f4").tobytes())
    return digest.hexdigest()


def build_features(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    project_paths = discover_project_paths(project_root)
    paths = _paths(project_root)
    benchmark_result = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    if (
        benchmark_result.get("protocol_id") != PROTOCOL_ID
        or benchmark_result.get("proceed") is not True
        or benchmark_result.get("competition_outcomes_accessed") is not False
    ):
        raise ValueError("E780 target-free benchmark gate is invalid.")
    contexts = pd.read_parquet(
        project_paths.cache_dir / "response_objective_context.parquet",
        columns=["response_id", "session_id", "objective_context"],
    )
    rows: list[dict[str, object]] = []
    for index, row in enumerate(contexts.itertuples(index=False)):
        values: dict[str, object] = {
            "response_id": str(row.response_id),
            "session_id": str(row.session_id),
        }
        values.update(extract_context_features(row.objective_context))
        rows.append(values)
        if (index + 1) % 5_000 == 0:
            print(f"E780 features: {index + 1}/{len(contexts)}", flush=True)
    frame = pd.DataFrame.from_records(rows)
    frame[list(FEATURE_NAMES)] = frame[list(FEATURE_NAMES)].astype(np.float32)
    if (
        frame["response_id"].duplicated().any()
        or list(frame["response_id"]) != list(contexts["response_id"].astype(str))
        or not np.isfinite(frame[list(FEATURE_NAMES)].to_numpy()).all()
    ):
        raise RuntimeError("E780 full target-free feature contract failed.")
    frame.to_parquet(paths["features"], index=False)
    assignments = pd.read_parquet(
        project_paths.cache_dir / "validation_environments_selection.parquet",
        columns=["environment", "response_id", "style_cell"],
    )
    style = assignments.loc[
        assignments["environment"].eq("V_style")
    ].merge(
        frame[["response_id", "episode_count_log1p"]],
        on="response_id",
        how="left",
        validate="one_to_one",
    )
    style["has_episode"] = style["episode_count_log1p"] > 0.0
    style_coverage = {
        str(int(cell)): float(group["has_episode"].mean())
        for cell, group in style.groupby("style_cell", sort=True)
    }
    episode_count = np.expm1(frame["episode_count_log1p"]).to_numpy()
    state_counts = {
        state: int(
            round(
                float(
                    (
                        frame[f"{state}_proportion"] * episode_count
                    ).sum()
                )
            )
        )
        for state in STATE_NAMES
    }
    nonconstant_features = int(
        (frame[list(FEATURE_NAMES)].nunique(dropna=False) > 1).sum()
    )
    clauses = {
        "overall_episode_coverage_at_least_0_90": (
            float(style["has_episode"].mean())
            >= MIN_RESPONSE_EPISODE_COVERAGE
        ),
        "each_style_cell_episode_coverage_at_least_0_70": (
            min(style_coverage.values())
            >= MIN_STYLE_CELL_EPISODE_COVERAGE
        ),
        "every_challenge_state_has_at_least_50_episodes": (
            min(state_counts.values()) >= MIN_STATE_EPISODES
        ),
        "all_features_are_nonconstant": (
            nonconstant_features == len(FEATURE_NAMES)
        ),
        "rasch_gap_is_finite_and_nonconstant": bool(
            np.isfinite(frame["rasch_gap"]).all()
            and frame["rasch_gap"].nunique() > 1
        ),
        "all_features_are_finite": bool(
            np.isfinite(frame[list(FEATURE_NAMES)].to_numpy()).all()
        ),
    }
    metadata: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
        "rows": len(frame),
        "features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_content_sha256": _feature_content_hash(frame),
        "feature_parquet_sha256": _sha256(paths["features"]),
        "overall_episode_coverage": float(style["has_episode"].mean()),
        "style_cell_episode_coverage": style_coverage,
        "challenge_state_counts": state_counts,
        "nonconstant_features": nonconstant_features,
        "rasch_gap_summary": {
            str(name): float(value)
            for name, value in frame["rasch_gap"].describe().items()
        },
        "clauses": clauses,
        "passes_target_free_gate": bool(all(clauses.values())),
        "runtime_seconds": time.perf_counter() - started,
        "runtime": runtime,
        "competition_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def verify_feature_cache(project_root: str | Path) -> dict[str, object]:
    verify_sources(project_root)
    if not EXPECTED_FEATURE_CONTENT_SHA256 or not EXPECTED_FEATURE_PARQUET_SHA256:
        raise RuntimeError("E780 feature cache hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["features"]) != EXPECTED_FEATURE_PARQUET_SHA256:
        raise ValueError("E780 feature Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["features"])
    if _feature_content_hash(frame) != EXPECTED_FEATURE_CONTENT_SHA256:
        raise ValueError("E780 feature ordered-content SHA-256 changed.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("passes_target_free_gate") is not True
        or metadata.get("competition_outcomes_accessed") is not False
        or metadata.get("V_joint_accessed") is not False
        or metadata.get("V_final_accessed") is not False
    ):
        raise ValueError("E780 feature-cache gate is invalid.")
    return metadata


def align_rasch_gap(
    component: pd.DataFrame, feature_frame: pd.DataFrame
) -> np.ndarray:
    if (
        component["response_id"].duplicated().any()
        or feature_frame["response_id"].duplicated().any()
    ):
        raise ValueError("E780 feature alignment contains duplicate responses.")
    lookup = feature_frame.set_index("response_id")["rasch_gap"]
    aligned = component["response_id"].astype(str).map(lookup)
    if aligned.isna().any():
        raise ValueError("E780 could not align every response exactly once.")
    values = aligned.to_numpy(dtype=np.float64)
    if values.shape != (len(component),) or not np.isfinite(values).all():
        raise RuntimeError("E780 aligned Rasch gap is invalid.")
    return values


def fit_rasch_calibration(
    training_gap: np.ndarray,
    training_target: np.ndarray,
    validation_gap: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    mean = float(training_gap.mean())
    scale = float(training_gap.std(ddof=0))
    if scale <= 1e-8:
        raise ValueError("E780 training Rasch gap is constant.")
    x_train = (training_gap - mean) / scale
    x_validation = (validation_gap - mean) / scale
    prior = float(np.clip(training_target.mean(), 1e-5, 1.0 - 1e-5))
    initial = np.asarray([math.log(prior / (1.0 - prior)), 0.5])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        probability = expit(intercept + slope * x_train)
        probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
        loss = -float(
            np.mean(
                training_target * np.log(probability)
                + (1.0 - training_target) * np.log1p(-probability)
            )
        ) + 0.5 * RASCH_SLOPE_L2 * slope**2
        residual = probability - training_target
        gradient = np.asarray(
            [
                float(residual.mean()),
                float(np.mean(residual * x_train) + RASCH_SLOPE_L2 * slope),
            ]
        )
        return loss, gradient

    result = minimize(
        lambda values: objective(values)[0],
        initial,
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
        bounds=((-6.0, 6.0), (0.0, RASCH_MAX_SLOPE)),
        options={"maxiter": RASCH_MAX_ITERATIONS, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"E780 Rasch calibration failed: {result.message}")
    intercept, slope = result.x
    prediction = expit(intercept + slope * x_validation)
    return prediction, {
        "intercept": float(intercept),
        "slope": float(slope),
        "training_gap_mean": mean,
        "training_gap_std": scale,
        "training_prior": prior,
        "iterations": int(result.nit),
        "optimizer_success": bool(result.success),
    }


def rasch_oof(
    component: pd.DataFrame, gap: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target = component["target"].to_numpy(dtype=np.int8)
    folds = component["fold"].to_numpy(dtype=np.int8)
    prediction = np.full(len(component), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        training_mask, validation_mask = _fold_masks(component, folds, fold)
        training_sessions = set(
            component.loc[training_mask, "session_id"].astype(str)
        )
        validation_sessions = set(
            component.loc[validation_mask, "session_id"].astype(str)
        )
        if training_sessions & validation_sessions:
            raise RuntimeError(f"E780 session leakage in fold {fold}.")
        fold_prediction, summary = fit_rasch_calibration(
            gap[training_mask],
            target[training_mask],
            gap[validation_mask],
        )
        prediction[validation_mask] = fold_prediction
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(training_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "training_sessions": len(training_sessions),
                "validation_sessions": len(validation_sessions),
                **summary,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("E780 OOF predictions are incomplete.")
    return prediction, summaries


def _baseline_prediction(component: pd.DataFrame) -> np.ndarray:
    return (
        0.25 * component["pred_full"].to_numpy(dtype=np.float64)
        + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
        + 0.50 * component["pred_bge_base"].to_numpy(dtype=np.float64)
    )


def build_prediction_rows(
    component: pd.DataFrame,
    rasch_prediction: np.ndarray,
    weights: Sequence[float],
) -> pd.DataFrame:
    baseline = _baseline_prediction(component)
    metadata = component[
        [
            "environment",
            "response_id",
            "session_id",
            "learning_objective_id",
            "semantic_family",
            "fold",
            "evaluation_eligible",
            "target",
        ]
    ].copy()
    rows: list[pd.DataFrame] = []
    for weight in weights:
        candidate = metadata.copy()
        candidate["rasch_weight"] = float(weight)
        candidate["pred_v05_raw"] = baseline
        candidate["pred_rasch_challenge"] = rasch_prediction
        candidate["prediction"] = np.clip(
            (1.0 - weight) * baseline + weight * rasch_prediction,
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        rows.append(candidate)
    return pd.concat(rows, ignore_index=True)


def _metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (environment, weight, fold), group in predictions.groupby(
        ["environment", "rasch_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        candidate = binary_metrics(target, scored["prediction"])
        row: dict[str, object] = {
            "environment": str(environment),
            "rasch_weight": float(weight),
            "fold": int(fold),
            "rows": int(len(scored)),
        }
        for name, value in candidate.items():
            row[name] = value
            row[f"baseline_{name}"] = baseline[name]
            row[f"delta_{name}_vs_v05_raw"] = value - baseline[name]
        rows.append(row)
    folds = pd.DataFrame(rows)
    metric_names = (
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece_10",
        "prediction_mean",
    )
    columns = [
        *metric_names,
        *[f"baseline_{name}" for name in metric_names],
        *[f"delta_{name}_vs_v05_raw" for name in metric_names],
    ]
    environments = (
        folds.groupby(
            ["environment", "rasch_weight"], as_index=False, sort=True
        )[columns]
        .mean()
        .sort_values(["environment", "rasch_weight"], kind="mergesort")
        .reset_index(drop=True)
    )
    environments["aggregation"] = "equal_fold_macro"
    return folds, environments


def select_candidate(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    environment_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    bootstraps: dict[str, dict[str, object]] = {}
    for index, weight in enumerate(BLEND_WEIGHTS):
        metrics = environment_metrics.loc[
            environment_metrics["rasch_weight"].eq(weight)
        ]
        selected_predictions = predictions.loc[
            predictions["rasch_weight"].eq(weight)
        ]
        session_bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="session_id",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + index,
        )
        family_bootstrap = _macro_cluster_bootstrap(
            selected_predictions,
            candidate_column="prediction",
            baseline_column="pred_v05_raw",
            group_column="semantic_family",
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + 100 + index,
        )
        bootstraps[f"{weight:.2f}"] = {
            "session": session_bootstrap,
            "semantic_family": family_bootstrap,
        }
        delta_loss = metrics[
            "delta_log_loss_vs_v05_raw"
        ].to_numpy(dtype=np.float64)
        weight_folds = fold_metrics.loc[
            fold_metrics["rasch_weight"].eq(weight)
        ]
        rows.append(
            {
                "rasch_weight": weight,
                "mean_log_loss": float(metrics["log_loss"].mean()),
                "mean_log_loss_gain_vs_v05_raw": float(-delta_loss.mean()),
                "improved_environments": int(np.sum(delta_loss < 0.0)),
                "worst_environment_delta_log_loss_vs_v05_raw": float(
                    delta_loss.max()
                ),
                "worst_fold_delta_log_loss_vs_v05_raw": float(
                    weight_folds["delta_log_loss_vs_v05_raw"].max()
                ),
                "mean_delta_roc_auc_vs_v05_raw": float(
                    metrics["delta_roc_auc_vs_v05_raw"].mean()
                ),
                "mean_delta_brier_score_vs_v05_raw": float(
                    metrics["delta_brier_score_vs_v05_raw"].mean()
                ),
                "mean_delta_ece_10_vs_v05_raw": float(
                    metrics["delta_ece_10_vs_v05_raw"].mean()
                ),
                "session_bootstrap_support": session_bootstrap[
                    "support_positive_gain"
                ],
                "semantic_family_bootstrap_support": family_bootstrap[
                    "support_positive_gain"
                ],
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "rasch_weight"], kind="mergesort"
    )
    selected = selection.iloc[0].to_dict()
    clauses = {
        "mean_log_loss_gain_at_least_0_0016": (
            selected["mean_log_loss_gain_vs_v05_raw"] >= 0.0016
        ),
        "all_three_environments_improve": (
            selected["improved_environments"] == len(SELECTION_ENVIRONMENTS)
        ),
        "no_environment_log_loss_regression": (
            selected["worst_environment_delta_log_loss_vs_v05_raw"] <= 0.0
        ),
        "worst_fold_regression_at_most_0_0005": (
            selected["worst_fold_delta_log_loss_vs_v05_raw"] <= 0.0005
        ),
        "macro_auroc_non_regression": (
            selected["mean_delta_roc_auc_vs_v05_raw"] >= 0.0
        ),
        "macro_brier_non_regression": (
            selected["mean_delta_brier_score_vs_v05_raw"] <= 0.0
        ),
        "macro_ece_non_regression": (
            selected["mean_delta_ece_10_vs_v05_raw"] <= 0.0
        ),
        "session_bootstrap_support_at_least_0_95": (
            selected["session_bootstrap_support"] >= 0.95
        ),
        "semantic_family_bootstrap_support_at_least_0_95": (
            selected["semantic_family_bootstrap_support"] >= 0.95
        ),
    }
    selected_key = f"{selected['rasch_weight']:.2f}"
    return selection.reset_index(drop=True), {
        "selected_weight": float(selected["rasch_weight"]),
        "selected_row": selected,
        "clauses": clauses,
        "passes_screen": bool(all(clauses.values())),
        "selected_bootstrap": bootstraps[selected_key],
        "all_preregistered_bootstraps": bootstraps,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }


def validate(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    feature_metadata = verify_feature_cache(project_root)
    project_paths = discover_project_paths(project_root)
    feature_frame = pd.read_parquet(_paths(project_root)["features"])
    prediction_frames: list[pd.DataFrame] = []
    fit_summaries: dict[str, list[dict[str, object]]] = {}
    for environment in SELECTION_ENVIRONMENTS:
        print(f"E780 validation: {environment}", flush=True)
        component = load_component_oof(project_root, environment)
        gap = align_rasch_gap(component, feature_frame)
        prediction, summaries = rasch_oof(component, gap)
        fit_summaries[environment] = summaries
        prediction_frames.append(
            build_prediction_rows(component, prediction, BLEND_WEIGHTS)
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics, environment_metrics = _metric_tables(predictions)
    selection, decision = select_candidate(
        predictions, fold_metrics, environment_metrics
    )
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_ability_demand_challenge")
    run_dir = project_paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "development_predictions.parquet": predictions,
        "development_fold_metrics.csv": fold_metrics,
        "development_environment_metrics.csv": environment_metrics,
        "development_selection.csv": selection,
    }
    artifact_paths: list[Path] = []
    for name, value in artifacts.items():
        path = run_dir / name
        if path.suffix == ".parquet":
            value.to_parquet(path, index=False)
        else:
            value.to_csv(path, index=False, lineterminator="\n")
        artifact_paths.append(path)
    bootstrap_path = run_dir / "development_bootstraps.json"
    bootstrap_path.write_text(
        json.dumps(
            decision["all_preregistered_bootstraps"], indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(bootstrap_path)
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "generated_at_utc": timestamp.isoformat(),
        "candidate": "E780_ability_demand_optimal_challenge",
        "source_run_id": SOURCE_RUN_ID,
        "source_hashes": source_hashes,
        "feature_metadata": feature_metadata,
        "lineage": {
            "feature_count": len(FEATURE_NAMES),
            "challenge_states": list(STATE_NAMES),
            "max_episode_turns": MAX_EPISODE_TURNS,
            "session_ability_theta": (
                "mean ability + 0.25 ability shift + "
                "0.20 optimal proportion + 0.15 recovery proportion"
            ),
            "session_demand_d": (
                "mean demand + 0.50 over proportion + "
                "0.20 unresolved proportion"
            ),
            "rasch_gap": "session ability theta - session demand d",
            "estimator": (
                "fold-local two-parameter Rasch calibration with "
                "non-negative scalar slope"
            ),
            "slope_l2": RASCH_SLOPE_L2,
            "max_slope": RASCH_MAX_SLOPE,
            "seed": SEED,
            "raw_v05_weights": {
                "pred_full": 0.25,
                "pred_role": 0.25,
                "pred_bge_base": 0.50,
            },
            "rasch_blend_weights": list(BLEND_WEIGHTS),
        },
        "preregistered_gate": decision,
        "fit_summaries": fit_summaries,
        "runtime": runtime,
        "runtime_metadata": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "validation_runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            path.name: _sha256(path) for path in artifact_paths
        },
        "competition_outcomes_accessed": True,
        "V_joint_accessed": False,
        "V_final_accessed": False,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "passes_screen": decision["passes_screen"],
        "selected": decision["selected_row"],
        "clauses": decision["clauses"],
        "artifact_sha256": report["artifact_sha256"],
        "report_sha256": _sha256(report_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E780 ability-demand challenge screen."
    )
    parser.add_argument(
        "stage", choices=("benchmark", "build-features", "validate")
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage == "benchmark":
        result = benchmark(args.project_root)
    elif args.stage == "build-features":
        result = build_features(args.project_root)
    else:
        result = validate(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
