from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trace_ace.io import discover_project_paths
from trace_ace.multiview_cache import _parse_role_lines
from trace_ace.objective_retrieval import _objective_terms, _tokens


ORDERED_FEATURE_SCHEMA = "2026-07-17-v1"
AFFIRM_RE = re.compile(
    r"\b(?:correct|exactly|excellent|great|good|well done|yes|right|brilliant|perfect|super)\b",
    re.IGNORECASE,
)
CORRECTION_RE = re.compile(
    r"\b(?:incorrect|wrong|not quite|try again|almost|check(?: that| your)?|remember)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:don'?t know|not sure|confused|can'?t|cannot|unsure|maybe|guess)\b",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"\b(?:because|therefore|so|means|equals|i think|first|then|next)\b",
    re.IGNORECASE,
)

ORDERED_FEATURE_NAMES = [
    "ordered_dialogue_turns",
    "ordered_student_turns",
    "ordered_tutor_turns",
    "ordered_role_switches",
    "ordered_student_turn_share",
    "ordered_affirm_early",
    "ordered_affirm_late",
    "ordered_affirm_delta",
    "ordered_correction_early",
    "ordered_correction_late",
    "ordered_correction_delta",
    "ordered_uncertainty_early",
    "ordered_uncertainty_late",
    "ordered_uncertainty_delta",
    "ordered_reasoning_early",
    "ordered_reasoning_late",
    "ordered_reasoning_delta",
    "ordered_student_objective_mean",
    "ordered_student_objective_max",
    "ordered_student_objective_first",
    "ordered_student_objective_last",
    "ordered_student_objective_early",
    "ordered_student_objective_late",
    "ordered_student_objective_delta",
    "ordered_student_objective_positive_share",
    "ordered_tutor_objective_mean",
    "ordered_tutor_objective_max",
    "ordered_tutor_objective_last",
    "ordered_tutor_objective_delta",
    "ordered_tutor_objective_positive_share",
    "ordered_feedback_pairs",
    "ordered_affirm_pairs",
    "ordered_correction_pairs",
    "ordered_objective_affirm_pairs",
    "ordered_objective_correction_pairs",
    "ordered_uncertain_correction_pairs",
    "ordered_uncertain_affirm_pairs",
    "ordered_reasoning_affirm_pairs",
    "ordered_repair_opportunities",
    "ordered_repair_overlap_improved",
    "ordered_repair_later_affirm",
    "ordered_repair_overlap_delta_mean",
    "ordered_repair_success_rate",
    "ordered_student_words_early",
    "ordered_student_words_late",
    "ordered_student_words_delta",
    "ordered_tutor_words_early",
    "ordered_tutor_words_late",
    "ordered_tutor_words_delta",
    "ordered_last_affirm_position",
    "ordered_last_correction_position",
]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _cue(text: str, pattern: re.Pattern[str]) -> float:
    return float(bool(pattern.search(text)))


def extract_ordered_features(
    lines: list[tuple[str, str]], objective: str
) -> dict[str, float]:
    terms, _ = _objective_terms(objective)
    term_set = set(terms)
    term_denominator = max(1, len(term_set))
    dialogue: list[dict[str, object]] = []
    for role, content in lines:
        if role not in {"student", "tutor"}:
            continue
        tokens = _tokens(content)
        overlap = len(term_set.intersection(tokens)) / term_denominator
        dialogue.append(
            {
                "role": role,
                "content": content,
                "tokens": tokens,
                "overlap": float(overlap),
                "words": float(len(str(content).split())),
                "affirm": _cue(content, AFFIRM_RE) if role == "tutor" else 0.0,
                "correction": _cue(content, CORRECTION_RE) if role == "tutor" else 0.0,
                "uncertainty": _cue(content, UNCERTAINTY_RE) if role == "student" else 0.0,
                "reasoning": _cue(content, REASONING_RE) if role == "student" else 0.0,
            }
        )

    count = len(dialogue)
    for index, turn in enumerate(dialogue):
        turn["position"] = float(index / max(1, count - 1))
    student = [turn for turn in dialogue if turn["role"] == "student"]
    tutor = [turn for turn in dialogue if turn["role"] == "tutor"]
    early = [turn for turn in dialogue if float(turn["position"]) <= 1.0 / 3.0]
    late = [turn for turn in dialogue if float(turn["position"]) >= 2.0 / 3.0]

    def cue_count(turns: list[dict[str, object]], name: str) -> float:
        return float(sum(float(turn[name]) for turn in turns))

    def role_values(
        turns: list[dict[str, object]], role: str, field: str
    ) -> list[float]:
        return [float(turn[field]) for turn in turns if turn["role"] == role]

    student_overlap = role_values(dialogue, "student", "overlap")
    tutor_overlap = role_values(dialogue, "tutor", "overlap")
    student_early_overlap = role_values(early, "student", "overlap")
    student_late_overlap = role_values(late, "student", "overlap")
    tutor_early_overlap = role_values(early, "tutor", "overlap")
    tutor_late_overlap = role_values(late, "tutor", "overlap")

    affirm_early = cue_count(early, "affirm")
    affirm_late = cue_count(late, "affirm")
    correction_early = cue_count(early, "correction")
    correction_late = cue_count(late, "correction")
    uncertainty_early = cue_count(early, "uncertainty")
    uncertainty_late = cue_count(late, "uncertainty")
    reasoning_early = cue_count(early, "reasoning")
    reasoning_late = cue_count(late, "reasoning")

    feedback_pairs = 0.0
    affirm_pairs = 0.0
    correction_pairs = 0.0
    objective_affirm_pairs = 0.0
    objective_correction_pairs = 0.0
    uncertain_correction_pairs = 0.0
    uncertain_affirm_pairs = 0.0
    reasoning_affirm_pairs = 0.0
    repair_opportunities = 0.0
    repair_overlap_improved = 0.0
    repair_later_affirm = 0.0
    repair_deltas: list[float] = []

    for index in range(max(0, count - 1)):
        answer = dialogue[index]
        feedback = dialogue[index + 1]
        if answer["role"] != "student" or feedback["role"] != "tutor":
            continue
        feedback_pairs += 1.0
        affirmation = float(feedback["affirm"])
        correction = float(feedback["correction"])
        objective_linked = float(answer["overlap"]) > 0.0
        affirm_pairs += affirmation
        correction_pairs += correction
        objective_affirm_pairs += affirmation * objective_linked
        objective_correction_pairs += correction * objective_linked
        uncertain_correction_pairs += float(answer["uncertainty"]) * correction
        uncertain_affirm_pairs += float(answer["uncertainty"]) * affirmation
        reasoning_affirm_pairs += float(answer["reasoning"]) * affirmation
        if not correction:
            continue
        repair_opportunities += 1.0
        next_students = [
            later
            for later in dialogue[index + 2 :]
            if later["role"] == "student"
        ]
        if next_students:
            delta = float(next_students[0]["overlap"]) - float(answer["overlap"])
            repair_deltas.append(delta)
            repair_overlap_improved += float(delta > 0.0)
        repair_later_affirm += float(
            any(float(later["affirm"]) > 0.0 for later in dialogue[index + 2 :])
        )

    role_switches = float(
        sum(
            left["role"] != right["role"]
            for left, right in zip(dialogue, dialogue[1:])
        )
    )
    last_affirm = max(
        (float(turn["position"]) for turn in tutor if float(turn["affirm"]) > 0.0),
        default=-1.0,
    )
    last_correction = max(
        (float(turn["position"]) for turn in tutor if float(turn["correction"]) > 0.0),
        default=-1.0,
    )
    student_words_early = _mean(role_values(early, "student", "words"))
    student_words_late = _mean(role_values(late, "student", "words"))
    tutor_words_early = _mean(role_values(early, "tutor", "words"))
    tutor_words_late = _mean(role_values(late, "tutor", "words"))
    student_objective_early = _mean(student_early_overlap)
    student_objective_late = _mean(student_late_overlap)
    tutor_objective_early = _mean(tutor_early_overlap)
    tutor_objective_late = _mean(tutor_late_overlap)
    repair_rate = (
        (repair_overlap_improved + repair_later_affirm)
        / (2.0 * repair_opportunities)
        if repair_opportunities
        else 0.0
    )

    values = [
        float(count),
        float(len(student)),
        float(len(tutor)),
        role_switches,
        float(len(student) / max(1, count)),
        affirm_early,
        affirm_late,
        affirm_late - affirm_early,
        correction_early,
        correction_late,
        correction_late - correction_early,
        uncertainty_early,
        uncertainty_late,
        uncertainty_late - uncertainty_early,
        reasoning_early,
        reasoning_late,
        reasoning_late - reasoning_early,
        _mean(student_overlap),
        max(student_overlap, default=0.0),
        student_overlap[0] if student_overlap else 0.0,
        student_overlap[-1] if student_overlap else 0.0,
        student_objective_early,
        student_objective_late,
        student_objective_late - student_objective_early,
        float(np.mean(np.asarray(student_overlap) > 0.0)) if student_overlap else 0.0,
        _mean(tutor_overlap),
        max(tutor_overlap, default=0.0),
        tutor_overlap[-1] if tutor_overlap else 0.0,
        tutor_objective_late - tutor_objective_early,
        float(np.mean(np.asarray(tutor_overlap) > 0.0)) if tutor_overlap else 0.0,
        feedback_pairs,
        affirm_pairs,
        correction_pairs,
        objective_affirm_pairs,
        objective_correction_pairs,
        uncertain_correction_pairs,
        uncertain_affirm_pairs,
        reasoning_affirm_pairs,
        repair_opportunities,
        repair_overlap_improved,
        repair_later_affirm,
        _mean(repair_deltas),
        repair_rate,
        student_words_early,
        student_words_late,
        student_words_late - student_words_early,
        tutor_words_early,
        tutor_words_late,
        tutor_words_late - tutor_words_early,
        last_affirm,
        last_correction,
    ]
    if len(values) != len(ORDERED_FEATURE_NAMES):
        raise RuntimeError("Ordered feature names and values are misaligned.")
    if not np.isfinite(values).all():
        raise RuntimeError("Ordered features contain non-finite values.")
    return dict(zip(ORDERED_FEATURE_NAMES, values))


def build_ordered_feature_cache(project_root: str | Path) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    output_path = paths.cache_dir / "response_ordered_features.parquet"
    frame = pd.read_parquet(
        paths.cache_dir / "modeling_base.parquet",
        columns=["response_id", "session_id", "learning_objective"],
    )
    if output_path.exists():
        try:
            cached = pd.read_parquet(
                output_path, columns=["response_id", "ordered_feature_schema"]
            )
        except (KeyError, ValueError):
            cached = pd.DataFrame()
        if (
            not cached.empty
            and list(cached["response_id"]) == list(frame["response_id"])
            and set(cached["ordered_feature_schema"]) == {ORDERED_FEATURE_SCHEMA}
        ):
            return {"path": str(output_path), "responses": len(cached), "reused": True}

    session_texts = pd.read_parquet(
        paths.cache_dir / "session_texts.parquet", columns=["session_id", "full_text"]
    ).set_index("session_id")
    indexed = frame.reset_index(names="response_row")
    rows: list[dict[str, object] | None] = [None] * len(frame)
    completed = 0
    for session_id, group in indexed.groupby("session_id", sort=False):
        session_key = str(session_id)
        lines = _parse_role_lines(str(session_texts.at[session_key, "full_text"]))
        for response in group.itertuples(index=False):
            row: dict[str, object] = {
                "response_id": str(response.response_id),
                "ordered_feature_schema": ORDERED_FEATURE_SCHEMA,
            }
            row.update(extract_ordered_features(lines, str(response.learning_objective)))
            rows[int(response.response_row)] = row
            completed += 1
            if completed % 5000 == 0:
                print(f"Built ordered features for {completed}/{len(frame)} responses.", flush=True)
    if any(row is None for row in rows):
        raise RuntimeError("At least one ordered feature row was not constructed.")
    result = pd.DataFrame(rows)
    if list(result["response_id"]) != list(frame["response_id"]):
        raise RuntimeError("Ordered feature order does not match modeling data.")
    result.to_parquet(output_path, index=False, compression="zstd")
    return {"path": str(output_path), "responses": len(result), "reused": False}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic ordered tutoring and mastery features."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build_ordered_feature_cache(args.project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
