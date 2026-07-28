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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trace_ace.bge_base_multiview_screen import assert_runtime
from trace_ace.hard_validation import _fold_masks
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics
from trace_ace.objective_retrieval import _objective_terms
from trace_ace.timing_dynamics_validation import (
    _macro_cluster_bootstrap,
    load_component_oof,
)


PROTOCOL_ID = "E770_tutor_move_state_v1"
SOURCE_RUN_ID = "20260720T075633Z_environment_component_validation"
SELECTION_ENVIRONMENTS = ("V_seen", "V_objective", "V_style")
SEED = 20260728
REGULARIZATION_C = 0.1
MAX_ITERATIONS = 500
TOLERANCE = 1e-5
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
BOOTSTRAP_REPLICATES = 5_000
PROBABILITY_CLIP = 1e-6
MAX_PROJECTED_SECONDS = 3_600.0
MIN_GLOBAL_MOVE_EVENTS = 20
MIN_RESPONSE_PAIR_COVERAGE = 0.90
MIN_STYLE_CELL_PAIR_COVERAGE = 0.75
EXPECTED_FEATURE_CONTENT_SHA256 = (
    "d4f0134c4aa57cf2822139f7d4f0f6c5fc93b7002f11e7846c4421ad7e54d675"
)
EXPECTED_FEATURE_PARQUET_SHA256 = (
    "bd1ccd1d84fabbcaf83ae6f0920530585b54657e050f6b91f78abd900c4f748d"
)

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

MOVE_NAMES = (
    "probing_prior_knowledge",
    "probing_understand",
    "prompting_self_explanation",
    "prompting_self_correction",
    "prompting_next_step",
    "feedback_correct",
    "feedback_incorrect",
    "revoicing",
    "restating",
    "giving_hint",
    "giving_example",
    "explaining_conceptual",
    "explaining_procedural",
    "giving_answer",
    "praising_process",
    "praising_outcome",
)
STATE_NAMES = (
    "substantive_attempt",
    "short_guess",
    "uncertainty",
    "expressed_reasoning",
    "self_correction",
    "answer_change",
    "objective_or_math_evidence",
    "scaffolding_acceptance",
    "resistance",
    "bypass",
)
INTERACTION_NAMES = (
    "press_reasoning_after_strong",
    "revoice_after_weak",
    "self_correction_after_error",
    "direct_answer_low_followup",
    "hint_new_attempt",
    "process_praise_elaboration",
)
PHASES = ("all", "early", "middle", "late")

BASE_FEATURE_NAMES = (
    "eligible_pair_log1p",
    "next_student_rate",
    "student_tokens_mean_log1p",
    "tutor_tokens_mean_log1p",
    "move_coverage_rate",
    "state_coverage_rate",
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *tuple(f"move_{name}_{phase}_rate" for name in MOVE_NAMES for phase in PHASES),
    *tuple(f"state_{name}_{phase}_rate" for name in STATE_NAMES for phase in PHASES),
    *tuple(
        f"interaction_{name}_{phase}_rate"
        for name in INTERACTION_NAMES
        for phase in PHASES
    ),
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NOISE_PATTERN = re.compile(
    r"\[(?:unclear|inaudible|speaker:[^\]]+)\]|\b(?:uh+|um+|erm+|hmm+)\b",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"\b(?:because|therefore|since|so that|which means|that means|"
    r"first|then|next|after that|i think|my reason|the reason)\b"
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:i don'?t know|not sure|unsure|confused|maybe|i guess|"
    r"can'?t|cannot|i need help|no idea)\b"
)
SELF_CORRECTION_RE = re.compile(
    r"\b(?:actually|i mean|no wait|wait no|let me correct|"
    r"i made a mistake|sorry i meant|rather)\b"
)
ANSWER_CHANGE_RE = re.compile(
    r"\b(?:actually|i mean|no wait|wait no|instead|rather|sorry i meant)\b"
)
ACCEPTANCE_RE = re.compile(
    r"^[^a-z0-9]*(?:okay|ok|yes|yeah|right|got it|i see|"
    r"that makes sense|makes sense)\b"
)
RESISTANCE_RE = re.compile(
    r"\b(?:don'?t want|do not want|won'?t|will not|not doing|"
    r"no way|stop|leave me alone)\b"
)
BYPASS_RE = re.compile(
    r"\b(?:tell me the answer|give me the answer|just give|"
    r"do it for me|solve it for me|can we skip|skip this)\b"
)
MATH_WORDS = {
    "add",
    "added",
    "addition",
    "subtract",
    "subtracted",
    "subtraction",
    "multiply",
    "multiplied",
    "divide",
    "divided",
    "equals",
    "equal",
    "fraction",
    "decimal",
    "number",
    "tens",
    "ones",
    "hundreds",
    "triangle",
    "angle",
    "area",
    "perimeter",
    "percent",
    "ratio",
}

PRIOR_KNOWLEDGE_RE = re.compile(
    r"\b(?:have you (?:seen|learned|done|used|heard)|"
    r"do you remember|are you familiar|what do you already know|"
    r"have you come across)\b"
)
PROBING_UNDERSTAND_RE = re.compile(
    r"\b(?:do you understand|did you understand|does that make sense|"
    r"are you with me|do you follow|got it|is that clear)\b"
)
SELF_EXPLANATION_RE = re.compile(
    r"\b(?:why|how did you|how do you know|what makes you|"
    r"explain|tell me how|show me how|what is your reason|"
    r"can you justify)\b"
)
SELF_CORRECTION_PROMPT_RE = re.compile(
    r"\b(?:try again|check (?:that|your|it)|rethink|look again|"
    r"can you correct|fix (?:that|it|your)|what did you miss|"
    r"spot (?:the|your) mistake|find (?:the|your) error)\b"
)
NEXT_STEP_RE = re.compile(
    r"\b(?:what(?:'s| is) next|next step|what should (?:we|you) do next|"
    r"continue|go on|carry on|what do you do now|what comes next)\b"
)
FEEDBACK_CORRECT_RE = re.compile(
    r"\b(?:correct|exactly|that'?s right|that is right|you are right|"
    r"yes,? that is|"
    r"right answer|well done|perfect)\b"
)
FEEDBACK_INCORRECT_RE = re.compile(
    r"\b(?:incorrect|wrong|not quite|that is not right|"
    r"that'?s not right|almost but|no that is)\b"
)
REVOICE_CUE_RE = re.compile(
    r"\b(?:so you(?:'re| are) saying|in other words|"
    r"what you mean is|so your idea is|that means)\b"
)
HINT_RE = re.compile(
    r"\b(?:here'?s a hint|hint|remember that|think about|"
    r"what if|look at|start by|consider the)\b"
)
EXAMPLE_RE = re.compile(
    r"\b(?:for example|for instance|suppose|let'?s say|let us say|"
    r"imagine that|take the example)\b"
)
CONCEPT_RE = re.compile(
    r"\b(?:means|definition|concept|because|represents|"
    r"is called|we call|the idea is|property)\b"
)
PROCEDURE_RE = re.compile(
    r"\b(?:first|then|next|after that|step|start by|"
    r"add|subtract|multiply|divide|move the|carry the|"
    r"write down|calculate)\b"
)
ANSWER_RE = re.compile(
    r"\b(?:the answer is|the solution is|it equals|"
    r"that gives us|you get|it is)\s+(?:minus\s+)?\d"
)
PROCESS_PRAISE_RE = re.compile(
    r"\b(?:good thinking|great thinking|good strategy|great strategy|"
    r"nice explanation|good explanation|great reasoning|"
    r"good work(?:ing)?|nice work(?:ing)?|good effort|"
    r"you kept trying|i like how you)\b"
)
OUTCOME_PRAISE_RE = re.compile(
    r"\b(?:correct answer|you got it|you got that right|"
    r"well done|perfect answer|excellent answer|that is correct)\b"
)


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
            f"E770 immutable source verification failed: {actual}"
        )
    return actual


def _paths(project_root: str | Path) -> dict[str, Path]:
    paths = discover_project_paths(project_root)
    return {
        "features": paths.cache_dir / "tutor_move_state_e770.parquet",
        "metadata": paths.cache_dir / "tutor_move_state_e770.metadata.json",
        "benchmark": paths.cache_dir / "tutor_move_state_e770_benchmark.json",
    }


def _normalize(text: object) -> str:
    value = NOISE_PATTERN.sub(" ", str(text).lower())
    value = re.sub(r"\s+", " ", value).strip()
    return value


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


def classify_student_state(
    text: object,
    objective: object,
    *,
    previous_student: object = "",
) -> dict[str, float]:
    normalized = _normalize(text)
    tokens = _tokens(normalized)
    token_set = set(tokens)
    objective_terms, _ = _objective_terms(str(objective))
    objective_overlap = bool(token_set & set(objective_terms))
    numeric_or_operator = bool(
        re.search(r"(?:\d|[+\-*/=])", normalized)
    )
    math_evidence = bool(token_set & MATH_WORDS) or numeric_or_operator
    uncertainty = bool(UNCERTAINTY_RE.search(normalized))
    resistance = bool(RESISTANCE_RE.search(normalized))
    bypass = bool(BYPASS_RE.search(normalized))
    reasoning = bool(REASONING_RE.search(normalized))
    self_correction = bool(SELF_CORRECTION_RE.search(normalized))
    current_numbers = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    previous_numbers = re.findall(
        r"-?\d+(?:\.\d+)?", _normalize(previous_student)
    )
    answer_change = bool(ANSWER_CHANGE_RE.search(normalized)) or bool(
        previous_numbers
        and current_numbers
        and current_numbers != previous_numbers
        and re.search(r"\b(?:no|instead|rather|wait)\b", normalized)
    )
    short_guess = (
        len(tokens) <= 4
        and bool(tokens)
        and (math_evidence or bool(re.match(r"^(?:maybe|i think)", normalized)))
        and not bypass
    )
    substantive = (
        len(tokens) >= 5
        and not resistance
        and not bypass
        and not (
            uncertainty
            and len(tokens) <= 7
            and not math_evidence
            and not objective_overlap
        )
    )
    values = {
        "substantive_attempt": float(substantive),
        "short_guess": float(short_guess),
        "uncertainty": float(uncertainty),
        "expressed_reasoning": float(reasoning),
        "self_correction": float(self_correction),
        "answer_change": float(answer_change),
        "objective_or_math_evidence": float(objective_overlap or math_evidence),
        "scaffolding_acceptance": float(
            bool(ACCEPTANCE_RE.search(normalized))
        ),
        "resistance": float(resistance),
        "bypass": float(bypass),
    }
    if tuple(values) != STATE_NAMES:
        raise RuntimeError("E770 student-state schema drifted.")
    return values


def _overlap(student_text: object, tutor_text: object) -> tuple[float, float]:
    student_tokens = _tokens(student_text)
    tutor_tokens = _tokens(tutor_text)
    student_set = set(student_tokens)
    tutor_set = set(tutor_tokens)
    shared = student_set & tutor_set
    recall = len(shared) / max(1, len(student_set))
    precision = len(shared) / max(1, len(tutor_set))
    return recall, precision


def classify_tutor_move(
    student_text: object, tutor_text: object
) -> dict[str, float]:
    student = _normalize(student_text)
    tutor = _normalize(tutor_text)
    tutor_tokens = _tokens(tutor)
    recall, precision = _overlap(student, tutor)
    exact_restatement = bool(
        len(_tokens(student)) >= 3
        and (
            student in tutor
            or (recall >= 0.80 and precision >= 0.55)
        )
    )
    revoicing = bool(
        REVOICE_CUE_RE.search(tutor)
        or (
            len(_tokens(student)) >= 4
            and 0.35 <= recall < 0.80
            and precision >= 0.20
            and "?" not in tutor
        )
    )
    declarative_long = len(tutor_tokens) >= 8 and "?" not in tutor
    values = {
        "probing_prior_knowledge": float(
            bool(PRIOR_KNOWLEDGE_RE.search(tutor))
        ),
        "probing_understand": float(
            bool(PROBING_UNDERSTAND_RE.search(tutor))
        ),
        "prompting_self_explanation": float(
            bool(SELF_EXPLANATION_RE.search(tutor)) and "?" in tutor
        ),
        "prompting_self_correction": float(
            bool(SELF_CORRECTION_PROMPT_RE.search(tutor))
        ),
        "prompting_next_step": float(bool(NEXT_STEP_RE.search(tutor))),
        "feedback_correct": float(bool(FEEDBACK_CORRECT_RE.search(tutor))),
        "feedback_incorrect": float(
            bool(FEEDBACK_INCORRECT_RE.search(tutor))
        ),
        "revoicing": float(revoicing and not exact_restatement),
        "restating": float(exact_restatement),
        "giving_hint": float(bool(HINT_RE.search(tutor))),
        "giving_example": float(bool(EXAMPLE_RE.search(tutor))),
        "explaining_conceptual": float(
            declarative_long and bool(CONCEPT_RE.search(tutor))
        ),
        "explaining_procedural": float(
            declarative_long and bool(PROCEDURE_RE.search(tutor))
        ),
        "giving_answer": float(bool(ANSWER_RE.search(tutor))),
        "praising_process": float(bool(PROCESS_PRAISE_RE.search(tutor))),
        "praising_outcome": float(bool(OUTCOME_PRAISE_RE.search(tutor))),
    }
    if tuple(values) != MOVE_NAMES:
        raise RuntimeError("E770 tutor-move schema drifted.")
    return values


def _phase(index: int, total: int) -> str:
    position = index / max(1, total - 1)
    if position <= 1.0 / 3.0:
        return "early"
    if position >= 2.0 / 3.0:
        return "late"
    return "middle"


def extract_events(
    turns: Sequence[tuple[str, str]], objective: object
) -> list[dict[str, object]]:
    eligible_indices = [
        index
        for index in range(1, len(turns))
        if turns[index][0] == "tutor" and turns[index - 1][0] == "student"
    ]
    events: list[dict[str, object]] = []
    for sequence, index in enumerate(eligible_indices):
        student_text = turns[index - 1][1]
        tutor_text = turns[index][1]
        prior_student = ""
        for earlier_role, earlier_text in reversed(turns[: index - 1]):
            if earlier_role == "student":
                prior_student = earlier_text
                break
        next_student = ""
        if index + 1 < len(turns) and turns[index + 1][0] == "student":
            next_student = turns[index + 1][1]
        state = classify_student_state(
            student_text, objective, previous_student=prior_student
        )
        move = classify_tutor_move(student_text, tutor_text)
        next_state = classify_student_state(
            next_student, objective, previous_student=student_text
        ) if next_student else {name: 0.0 for name in STATE_NAMES}
        student_token_count = len(_tokens(student_text))
        next_token_count = len(_tokens(next_student))
        strong_mastery = bool(
            state["substantive_attempt"]
            and state["expressed_reasoning"]
            and state["objective_or_math_evidence"]
            and not state["uncertainty"]
        )
        weak_mastery = bool(
            state["uncertainty"]
            or state["short_guess"]
            or state["bypass"]
            or not state["substantive_attempt"]
        )
        interactions = {
            "press_reasoning_after_strong": float(
                strong_mastery and move["prompting_self_explanation"]
            ),
            "revoice_after_weak": float(
                weak_mastery and (move["revoicing"] or move["restating"])
            ),
            "self_correction_after_error": float(
                move["prompting_self_correction"]
                and (
                    move["feedback_incorrect"]
                    or state["uncertainty"]
                    or state["self_correction"]
                )
            ),
            "direct_answer_low_followup": float(
                move["giving_answer"] and next_token_count < 5
            ),
            "hint_new_attempt": float(
                move["giving_hint"]
                and next_state["substantive_attempt"]
                and not next_state["bypass"]
                and not next_state["resistance"]
            ),
            "process_praise_elaboration": float(
                move["praising_process"]
                and (
                    next_state["expressed_reasoning"]
                    or next_token_count > student_token_count
                )
            ),
        }
        if tuple(interactions) != INTERACTION_NAMES:
            raise RuntimeError("E770 interaction schema drifted.")
        events.append(
            {
                "sequence": sequence,
                "phase": _phase(sequence, len(eligible_indices)),
                "student_text": student_text,
                "tutor_text": tutor_text,
                "next_student": next_student,
                "student_tokens": student_token_count,
                "tutor_tokens": len(_tokens(tutor_text)),
                "move": move,
                "state": state,
                "interaction": interactions,
            }
        )
    return events


def aggregate_events(events: Sequence[dict[str, object]]) -> dict[str, float]:
    total = len(events)
    denominators = {
        phase: (
            total
            if phase == "all"
            else sum(event["phase"] == phase for event in events)
        )
        for phase in PHASES
    }

    def rate(family: str, name: str, phase: str) -> float:
        selected = (
            events
            if phase == "all"
            else [event for event in events if event["phase"] == phase]
        )
        return float(
            sum(float(event[family][name]) for event in selected)
            / max(1, denominators[phase])
        )

    student_tokens = [int(event["student_tokens"]) for event in events]
    tutor_tokens = [int(event["tutor_tokens"]) for event in events]
    next_student_rate = (
        sum(bool(event["next_student"]) for event in events) / max(1, total)
    )
    move_coverage = (
        sum(any(event["move"].values()) for event in events) / max(1, total)
    )
    state_coverage = (
        sum(any(event["state"].values()) for event in events) / max(1, total)
    )
    values: dict[str, float] = {
        "eligible_pair_log1p": math.log1p(total),
        "next_student_rate": float(next_student_rate),
        "student_tokens_mean_log1p": math.log1p(
            float(np.mean(student_tokens)) if student_tokens else 0.0
        ),
        "tutor_tokens_mean_log1p": math.log1p(
            float(np.mean(tutor_tokens)) if tutor_tokens else 0.0
        ),
        "move_coverage_rate": float(move_coverage),
        "state_coverage_rate": float(state_coverage),
    }
    for name in MOVE_NAMES:
        for phase in PHASES:
            values[f"move_{name}_{phase}_rate"] = rate("move", name, phase)
    for name in STATE_NAMES:
        for phase in PHASES:
            values[f"state_{name}_{phase}_rate"] = rate("state", name, phase)
    for name in INTERACTION_NAMES:
        for phase in PHASES:
            values[f"interaction_{name}_{phase}_rate"] = rate(
                "interaction", name, phase
            )
    if tuple(values) != FEATURE_NAMES or not np.isfinite(list(values.values())).all():
        raise RuntimeError("E770 aggregate feature contract failed.")
    return values


def extract_context_features(context: object) -> dict[str, float]:
    objective, turns = parse_context(context)
    return aggregate_events(extract_events(turns, objective))


MOVE_SYNTHETIC_CASES = {
    "probing_prior_knowledge": (
        "I am ready.",
        "Have you learned this kind of fraction before?",
        "Are you familiar with this type of fraction?",
        "Um, [unclear] have you seen this kind of fraction?",
        "Please read the fraction.",
    ),
    "probing_understand": (
        "I divided by two.",
        "Does that make sense?",
        "Do you follow that step?",
        "Uh, is that clear [unclear]?",
        "Please write the next line.",
    ),
    "prompting_self_explanation": (
        "I got twelve.",
        "Why did you choose twelve?",
        "Can you explain your reasoning?",
        "Um, how do you know [unclear] it is twelve?",
        "Write twelve in the box.",
    ),
    "prompting_self_correction": (
        "I think it is nine.",
        "Check your calculation and try again.",
        "Can you spot your mistake?",
        "Uh, look [unclear] again and fix it.",
        "The calculation is on the screen.",
    ),
    "prompting_next_step": (
        "I added the tens.",
        "What should you do next?",
        "What comes next?",
        "Um, go on [unclear], what is the next step?",
        "That was the first step.",
    ),
    "feedback_correct": (
        "The answer is twelve.",
        "Exactly, that is correct.",
        "Yes, that is right.",
        "Um, [unclear] correct, well done.",
        "Let us compare both answers.",
    ),
    "feedback_incorrect": (
        "The answer is twelve.",
        "That is not right.",
        "Not quite, there is an error.",
        "Um, [unclear] incorrect this time.",
        "Let us compare both answers.",
    ),
    "revoicing": (
        "I split the number into tens and ones.",
        "So your idea is to separate each place value.",
        "In other words, you decomposed it by place.",
        "Um, [unclear] what you mean is break it by value.",
        "Please solve the next problem.",
    ),
    "restating": (
        "I split the number into tens and ones.",
        "You said, I split the number into tens and ones.",
        "I split the number into tens and ones, yes.",
        "Um, I split [unclear] the number into tens and ones.",
        "You used a place-value strategy.",
    ),
    "giving_hint": (
        "I am stuck.",
        "Here is a hint: think about the tens column.",
        "Remember that each group contains ten.",
        "Um, [unclear] start by looking at the tens.",
        "The answer is forty.",
    ),
    "giving_example": (
        "I do not understand.",
        "For example, suppose we had three groups of ten.",
        "Let us say the number was thirty instead.",
        "Um, for instance [unclear], imagine two equal groups.",
        "Now solve your original problem.",
    ),
    "explaining_conceptual": (
        "What is a denominator?",
        "A denominator means the number of equal parts in the whole.",
        "We call it the denominator because it represents all equal parts.",
        "Um, [unclear] the concept means the whole is split into equal parts.",
        "Write the denominator.",
    ),
    "explaining_procedural": (
        "How do I add these?",
        "First add the ones, then carry the ten to the next column.",
        "Start by multiplying the digits and next write down the result.",
        "Um, first [unclear] subtract the ones, then subtract the tens.",
        "Addition is useful.",
    ),
    "giving_answer": (
        "What do I get?",
        "The answer is 42.",
        "It equals 42.",
        "Um, [unclear] you get 42.",
        "Try to calculate it.",
    ),
    "praising_process": (
        "I checked each step carefully.",
        "Good thinking and a great strategy.",
        "Nice explanation of your reasoning.",
        "Um, [unclear] good effort, you kept trying.",
        "Your answer is correct.",
    ),
    "praising_outcome": (
        "The answer is twelve.",
        "You got it, that is the correct answer.",
        "Perfect answer, well done.",
        "Um, [unclear] you got that right.",
        "I like how you kept trying.",
    ),
}

STATE_SYNTHETIC_CASES = {
    "substantive_attempt": (
        "I added the tens first and got eighty two.",
        "I combined both place values to make seventy three.",
        "Um, [unclear] I divided twelve by three and got four.",
        "Okay.",
    ),
    "short_guess": (
        "Maybe 12.",
        "I think four.",
        "Um, [unclear] 8 perhaps.",
        "I added the two tens and then checked the ones.",
    ),
    "uncertainty": (
        "I am not sure.",
        "Maybe, I do not know.",
        "Um, [unclear] I am confused.",
        "The answer is twelve.",
    ),
    "expressed_reasoning": (
        "I chose twelve because three times four is twelve.",
        "First I added the tens, then the ones.",
        "Um, I think [unclear] it works since both sides are equal.",
        "Twelve.",
    ),
    "self_correction": (
        "Actually, I mean fourteen.",
        "No wait, I made a mistake.",
        "Um, sorry [unclear] I meant sixteen.",
        "The answer is fourteen.",
    ),
    "answer_change": (
        "No, instead it is fourteen.",
        "Actually, I mean sixteen.",
        "Um, wait [unclear] rather it is eighteen.",
        "The answer remains twelve.",
    ),
    "objective_or_math_evidence": (
        "The decimal has two tenths.",
        "I add 7 and 5.",
        "Um, [unclear] the denominator is four.",
        "I like this lesson.",
    ),
    "scaffolding_acceptance": (
        "Okay, that makes sense.",
        "Got it, I see.",
        "Um, [unclear] yes, that makes sense.",
        "I do not understand.",
    ),
    "resistance": (
        "I do not want to do this.",
        "No way, I am not doing it.",
        "Um, [unclear] stop, leave me alone.",
        "I want to try another method.",
    ),
    "bypass": (
        "Just give me the answer.",
        "Can you solve it for me?",
        "Um, [unclear] tell me the answer.",
        "Can you give me a hint?",
    ),
}


def synthetic_gate() -> dict[str, object]:
    move_rows: list[dict[str, object]] = []
    for name, (student, canonical, paraphrase, noisy, negative) in (
        MOVE_SYNTHETIC_CASES.items()
    ):
        detections = {
            "canonical": bool(classify_tutor_move(student, canonical)[name]),
            "paraphrase": bool(classify_tutor_move(student, paraphrase)[name]),
            "asr_noise": bool(classify_tutor_move(student, noisy)[name]),
            "negation_rejected": not bool(
                classify_tutor_move(student, negative)[name]
            ),
        }
        move_rows.append({"name": name, **detections, "passes": all(detections.values())})
    state_rows: list[dict[str, object]] = []
    objective = "Adding and subtracting decimals and understanding denominators."
    for name, (canonical, paraphrase, noisy, negative) in (
        STATE_SYNTHETIC_CASES.items()
    ):
        detections = {
            "canonical": bool(classify_student_state(canonical, objective)[name]),
            "paraphrase": bool(classify_student_state(paraphrase, objective)[name]),
            "asr_noise": bool(classify_student_state(noisy, objective)[name]),
            "negation_rejected": not bool(
                classify_student_state(negative, objective)[name]
            ),
        }
        state_rows.append({"name": name, **detections, "passes": all(detections.values())})
    clauses = {
        "all_move_cases_pass": all(row["passes"] for row in move_rows),
        "all_state_cases_pass": all(row["passes"] for row in state_rows),
        "all_six_interactions_constructed": len(INTERACTION_NAMES) == 6,
        "feature_schema_is_frozen": len(FEATURE_NAMES) == 134,
    }
    return {
        "move_cases": move_rows,
        "state_cases": state_rows,
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
            f"E770-benchmark|{value}".encode()
        ).hexdigest()
    )
    sample = contexts.assign(_order=order).sort_values("_order").head(1_024)
    started = time.perf_counter()
    values = [
        extract_context_features(context)
        for context in sample["objective_context"]
    ]
    elapsed = time.perf_counter() - started
    matrix = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in values],
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
        raise ValueError("E770 target-free benchmark gate is invalid.")
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
            print(f"E770 features: {index + 1}/{len(contexts)}", flush=True)
    frame = pd.DataFrame.from_records(rows)
    frame[list(FEATURE_NAMES)] = frame[list(FEATURE_NAMES)].astype(np.float32)
    if (
        frame["response_id"].duplicated().any()
        or list(frame["response_id"]) != list(contexts["response_id"].astype(str))
        or not np.isfinite(frame[list(FEATURE_NAMES)].to_numpy()).all()
    ):
        raise RuntimeError("E770 full target-free feature contract failed.")
    frame.to_parquet(paths["features"], index=False)
    assignments = pd.read_parquet(
        project_paths.cache_dir / "validation_environments_selection.parquet",
        columns=["environment", "response_id", "style_cell"],
    )
    style = assignments.loc[
        assignments["environment"].eq("V_style")
    ].merge(
        frame[["response_id", "eligible_pair_log1p", "move_coverage_rate"]],
        on="response_id",
        how="left",
        validate="one_to_one",
    )
    style["has_pair"] = style["eligible_pair_log1p"] > 0.0
    style_coverage = {
        str(int(cell)): float(group["has_pair"].mean())
        for cell, group in style.groupby("style_cell", sort=True)
    }
    move_event_estimates = {
        name: int(
            round(
                float(
                    (
                        frame[f"move_{name}_all_rate"]
                        * np.expm1(frame["eligible_pair_log1p"])
                    ).sum()
                )
            )
        )
        for name in MOVE_NAMES
    }
    nonconstant_features = int(
        (frame[list(FEATURE_NAMES)].nunique(dropna=False) > 1).sum()
    )
    clauses = {
        "overall_pair_coverage_at_least_0_90": (
            float(style["has_pair"].mean()) >= MIN_RESPONSE_PAIR_COVERAGE
        ),
        "each_style_cell_pair_coverage_at_least_0_75": (
            min(style_coverage.values()) >= MIN_STYLE_CELL_PAIR_COVERAGE
        ),
        "every_move_has_at_least_20_events": (
            min(move_event_estimates.values()) >= MIN_GLOBAL_MOVE_EVENTS
        ),
        "all_features_are_nonconstant": (
            nonconstant_features == len(FEATURE_NAMES)
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
        "overall_pair_coverage": float(style["has_pair"].mean()),
        "style_cell_pair_coverage": style_coverage,
        "move_event_estimates": move_event_estimates,
        "nonconstant_features": nonconstant_features,
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
        raise RuntimeError("E770 feature cache hashes are not frozen.")
    paths = _paths(project_root)
    if _sha256(paths["features"]) != EXPECTED_FEATURE_PARQUET_SHA256:
        raise ValueError("E770 feature Parquet SHA-256 changed.")
    frame = pd.read_parquet(paths["features"])
    if _feature_content_hash(frame) != EXPECTED_FEATURE_CONTENT_SHA256:
        raise ValueError("E770 feature ordered-content SHA-256 changed.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("passes_target_free_gate") is not True
        or metadata.get("competition_outcomes_accessed") is not False
        or metadata.get("V_joint_accessed") is not False
        or metadata.get("V_final_accessed") is not False
    ):
        raise ValueError("E770 feature-cache gate is invalid.")
    return metadata


def align_features(
    component: pd.DataFrame, feature_frame: pd.DataFrame
) -> np.ndarray:
    if (
        component["response_id"].duplicated().any()
        or feature_frame["response_id"].duplicated().any()
    ):
        raise ValueError("E770 feature alignment contains duplicate responses.")
    lookup = feature_frame.set_index("response_id")
    aligned = lookup.reindex(component["response_id"].astype(str))
    if aligned[list(FEATURE_NAMES)].isna().any().any():
        raise ValueError("E770 could not align every response exactly once.")
    matrix = aligned[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if matrix.shape != (len(component), len(FEATURE_NAMES)):
        raise RuntimeError("E770 aligned feature shape is invalid.")
    return matrix


def move_state_oof(
    component: pd.DataFrame, features: np.ndarray
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
            raise RuntimeError(f"E770 session leakage in fold {fold}.")
        scaler = StandardScaler()
        train = scaler.fit_transform(features[training_mask])
        validation = scaler.transform(features[validation_mask])
        model = LogisticRegression(
            C=REGULARIZATION_C,
            solver="lbfgs",
            max_iter=MAX_ITERATIONS,
            tol=TOLERANCE,
            random_state=SEED,
        )
        model.fit(train, target[training_mask])
        prediction[validation_mask] = model.predict_proba(validation)[:, 1]
        summaries.append(
            {
                "fold": fold,
                "training_rows": int(training_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "training_sessions": len(training_sessions),
                "validation_sessions": len(validation_sessions),
                "iterations": int(model.n_iter_[0]),
                "coefficient_l2": float(np.linalg.norm(model.coef_)),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("E770 OOF predictions are incomplete.")
    return prediction, summaries


def _baseline_prediction(component: pd.DataFrame) -> np.ndarray:
    return (
        0.25 * component["pred_full"].to_numpy(dtype=np.float64)
        + 0.25 * component["pred_role"].to_numpy(dtype=np.float64)
        + 0.50 * component["pred_bge_base"].to_numpy(dtype=np.float64)
    )


def build_prediction_rows(
    component: pd.DataFrame,
    move_state_prediction: np.ndarray,
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
        candidate["move_state_weight"] = float(weight)
        candidate["pred_v05_raw"] = baseline
        candidate["pred_move_state"] = move_state_prediction
        candidate["prediction"] = np.clip(
            (1.0 - weight) * baseline + weight * move_state_prediction,
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
        ["environment", "move_state_weight", "fold"], sort=True
    ):
        scored = group.loc[group["evaluation_eligible"].astype(bool)]
        target = scored["target"].to_numpy(dtype=np.int8)
        baseline = binary_metrics(target, scored["pred_v05_raw"])
        candidate = binary_metrics(target, scored["prediction"])
        row: dict[str, object] = {
            "environment": str(environment),
            "move_state_weight": float(weight),
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
            ["environment", "move_state_weight"], as_index=False, sort=True
        )[columns]
        .mean()
        .sort_values(["environment", "move_state_weight"], kind="mergesort")
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
            environment_metrics["move_state_weight"].eq(weight)
        ]
        selected_predictions = predictions.loc[
            predictions["move_state_weight"].eq(weight)
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
            fold_metrics["move_state_weight"].eq(weight)
        ]
        rows.append(
            {
                "move_state_weight": weight,
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
        ["mean_log_loss", "move_state_weight"], kind="mergesort"
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
    selected_key = f"{selected['move_state_weight']:.2f}"
    return selection.reset_index(drop=True), {
        "selected_weight": float(selected["move_state_weight"]),
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
        print(f"E770 validation: {environment}", flush=True)
        component = load_component_oof(project_root, environment)
        features = align_features(component, feature_frame)
        prediction, summaries = move_state_oof(component, features)
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
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ_tutor_move_state")
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
        "candidate": "E770_tutor_move_by_student_state",
        "source_run_id": SOURCE_RUN_ID,
        "source_hashes": source_hashes,
        "feature_metadata": feature_metadata,
        "lineage": {
            "feature_count": len(FEATURE_NAMES),
            "tutor_moves": list(MOVE_NAMES),
            "student_states": list(STATE_NAMES),
            "interactions": list(INTERACTION_NAMES),
            "phases": list(PHASES),
            "estimator": "fold-local standardized logistic regression",
            "regularization_c": REGULARIZATION_C,
            "solver": "lbfgs",
            "max_iterations": MAX_ITERATIONS,
            "tolerance": TOLERANCE,
            "seed": SEED,
            "raw_v05_weights": {
                "pred_full": 0.25,
                "pred_role": 0.25,
                "pred_bge_base": 0.50,
            },
            "move_state_blend_weights": list(BLEND_WEIGHTS),
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
        description="Run the frozen E770 tutor-move by student-state screen."
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
