from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from trace_ace.bge_base_multiview_screen import assert_runtime


PROTOCOL_ID = "E790_math_validity_target_free_discovery_v1"
MAX_ABSOLUTE_VALUE = Fraction(1_000_000)
MIN_PARSED_RELATIONS = 500
MIN_DISTINCT_RESPONSES = 250
MIN_RELATIONS_PER_VALIDITY_CLASS = 50
MIN_FEEDBACK_LINKED_RELATIONS = 200
MIN_STYLE_CELLS_WITH_TEN_LINKS = 8
MIN_BALANCED_ACCURACY = 0.70
MIN_MATTHEWS_CORRELATION = 0.30
MIN_POSITIVE_FEEDBACK_RATE_GAP = 0.25

EXPECTED_SOURCE_SHA256 = {
    "response_objective_context.parquet": (
        "218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6"
    ),
    "validation_environments_selection.parquet": (
        "600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40"
    ),
    "modeling_base.parquet": (
        "ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede"
    ),
}

MODELING_IDENTITY_COLUMNS = (
    "response_id",
    "session_id",
    "learning_objective_id",
)
CONTEXT_COLUMNS = ("response_id", "session_id", "objective_context")
ASSIGNMENT_COLUMNS = ("response_id", "style_cell")

NUMBER = r"-?(?:\d+/\d+|\d+(?:\.\d+)?)"
ARITHMETIC_PATTERN = re.compile(
    rf"(?<![\w.])(?P<left>{NUMBER})\s*"
    r"(?P<operator>\+|-|\*|×|x|÷|/|plus|add|minus|subtract|"
    r"take\s+away|times|multiplied\s+by|divided\s+by)\s*"
    rf"(?P<right>{NUMBER})\s*"
    r"(?P<comparator>=|equals?|is|>|<|greater\s+than|less\s+than)\s*"
    rf"(?P<stated>{NUMBER})(?![\w/])",
    re.IGNORECASE,
)
DIRECT_COMPARISON_PATTERN = re.compile(
    rf"(?<![\w.])(?P<left>{NUMBER})\s*"
    r"(?P<comparator>>|<|(?:is\s+)?greater\s+than|"
    r"(?:is\s+)?less\s+than)\s*"
    rf"(?P<right>{NUMBER})(?![\w/])",
    re.IGNORECASE,
)
ROLE_LINE_PATTERN = re.compile(
    r"^\[(?P<role>STUDENT|TUTOR|BACKGROUND)\]\s*(?P<text>.*)$",
    re.IGNORECASE,
)
MIXED_NUMBER_PATTERN = re.compile(r"(?<!\d)\d+\s+\d+/\d+(?!\d)")
ISO_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b")
EXTRA_LEFT_OPERATION_PATTERN = re.compile(
    rf"(?:{NUMBER})\s*(?:\+|-|\*|×|x|÷|/|plus|add|minus|subtract|"
    r"take\s+away|times|multiplied\s+by|divided\s+by)\s*$",
    re.IGNORECASE,
)
EXTRA_RIGHT_OPERATION_PATTERN = re.compile(
    rf"^\s*(?:\+|-|\*|×|x|÷|/|plus|add|minus|subtract|"
    rf"take\s+away|times|multiplied\s+by|divided\s+by)\s*(?:{NUMBER})",
    re.IGNORECASE,
)
POSITIVE_FEEDBACK_PATTERN = re.compile(
    r"\b(?:correct|exactly|well done|excellent|very good|good job|"
    r"that(?:'s| is) right|right answer|incredible|fantastic|brilliant)\b",
    re.IGNORECASE,
)
NEGATIVE_FEEDBACK_PATTERN = re.compile(
    r"\b(?:incorrect|wrong|not correct|not right|not quite|"
    r"that(?:'s| is) a mistake|try again|check (?:it|that|your answer) again|"
    r"you made a mistake|the answer (?:should|would) be)\b",
    re.IGNORECASE,
)

OPERATOR_MAP = {
    "+": "+",
    "plus": "+",
    "add": "+",
    "-": "-",
    "minus": "-",
    "subtract": "-",
    "take away": "-",
    "*": "*",
    "×": "*",
    "x": "*",
    "times": "*",
    "multiplied by": "*",
    "/": "/",
    "÷": "/",
    "divided by": "/",
}
COMPARATOR_MAP = {
    "=": "=",
    "equal": "=",
    "equals": "=",
    "is": "=",
    ">": ">",
    "greater than": ">",
    "is greater than": ">",
    "<": "<",
    "less than": "<",
    "is less than": "<",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "root": root,
        "context": root / "data_cache" / "response_objective_context.parquet",
        "assignments": (
            root / "data_cache" / "validation_environments_selection.parquet"
        ),
        "modeling": root / "data_cache" / "modeling_base.parquet",
        "report": (
            root
            / "experiments"
            / "reports"
            / "math_validity_e790_target_free.json"
        ),
    }


def verify_sources(project_root: str | Path) -> dict[str, str]:
    paths = _project_paths(project_root)
    actual = {
        "response_objective_context.parquet": _sha256(paths["context"]),
        "validation_environments_selection.parquet": _sha256(
            paths["assignments"]
        ),
        "modeling_base.parquet": _sha256(paths["modeling"]),
    }
    if actual != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"E790 source hashes changed: {actual}")
    return actual


def _normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _fraction(value: str) -> Fraction:
    result = Fraction(value)
    if abs(result) > MAX_ABSOLUTE_VALUE:
        raise ValueError("numeric value exceeds E790 bound")
    return result


def _operator(value: str) -> str:
    return OPERATOR_MAP[_normalize_space(value).lower()]


def _comparator(value: str) -> str:
    return COMPARATOR_MAP[_normalize_space(value).lower()]


def _apply_operator(left: Fraction, operator: str, right: Fraction) -> Fraction:
    if operator == "+":
        value = left + right
    elif operator == "-":
        value = left - right
    elif operator == "*":
        value = left * right
    elif operator == "/":
        if right == 0:
            raise ZeroDivisionError("division by zero")
        value = left / right
    else:
        raise ValueError(f"unsupported E790 operator: {operator}")
    if abs(value) > MAX_ABSOLUTE_VALUE:
        raise ValueError("computed value exceeds E790 bound")
    return value


def _compare(left: Fraction, comparator: str, right: Fraction) -> bool:
    if comparator == "=":
        return left == right
    if comparator == ">":
        return left > right
    if comparator == "<":
        return left < right
    raise ValueError(f"unsupported E790 comparator: {comparator}")


def _ambiguous_text(text: str) -> bool:
    return bool(
        "%" in text
        or MIXED_NUMBER_PATTERN.search(text)
        or ISO_DATE_PATTERN.search(text)
    )


def _has_extra_adjacent_operation(
    text: str, start: int, end: int
) -> bool:
    prefix = text[max(0, start - 40) : start]
    suffix = text[end : min(len(text), end + 40)]
    return bool(
        EXTRA_LEFT_OPERATION_PATTERN.search(prefix)
        or EXTRA_RIGHT_OPERATION_PATTERN.search(suffix)
    )


def _record_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def extract_relations(text: object) -> list[dict[str, object]]:
    normalized = _normalize_space(text)
    if not normalized or _ambiguous_text(normalized):
        return []

    positioned_relations: list[tuple[int, dict[str, object]]] = []
    occupied: list[tuple[int, int]] = []
    for match in DIRECT_COMPARISON_PATTERN.finditer(normalized):
        if _has_extra_adjacent_operation(normalized, match.start(), match.end()):
            continue
        try:
            left = _fraction(match.group("left"))
            right = _fraction(match.group("right"))
            comparator = _comparator(match.group("comparator"))
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        positioned_relations.append(
            (
                match.start(),
                {
                    "form": "comparison",
                    "raw_relation": match.group(0),
                    "left": _record_fraction(left),
                    "operator": "",
                    "right": _record_fraction(right),
                    "comparator": comparator,
                    "stated": "",
                    "computed": "",
                    "valid": bool(_compare(left, comparator, right)),
                },
            )
        )
        occupied.append(match.span())

    for match in ARITHMETIC_PATTERN.finditer(normalized):
        if any(
            match.start() < occupied_end and match.end() > occupied_start
            for occupied_start, occupied_end in occupied
        ):
            continue
        if _has_extra_adjacent_operation(normalized, match.start(), match.end()):
            continue
        try:
            left = _fraction(match.group("left"))
            right = _fraction(match.group("right"))
            stated = _fraction(match.group("stated"))
            operator = _operator(match.group("operator"))
            comparator = _comparator(match.group("comparator"))
            computed = _apply_operator(left, operator, right)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        positioned_relations.append(
            (
                match.start(),
                {
                    "form": "arithmetic",
                    "raw_relation": match.group(0),
                    "left": _record_fraction(left),
                    "operator": operator,
                    "right": _record_fraction(right),
                    "comparator": comparator,
                    "stated": _record_fraction(stated),
                    "computed": _record_fraction(computed),
                    "valid": bool(_compare(computed, comparator, stated)),
                },
            )
        )
        occupied.append(match.span())

    return [
        relation
        for _, relation in sorted(positioned_relations, key=lambda item: item[0])
    ]


def parse_turns(context: object) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in str(context).splitlines():
        match = ROLE_LINE_PATTERN.match(line.strip())
        if match is None:
            continue
        turns.append(
            (
                match.group("role").lower(),
                _normalize_space(match.group("text")),
            )
        )
    return turns


def feedback_label(text: object) -> str:
    normalized = _normalize_space(text)
    positive = bool(POSITIVE_FEEDBACK_PATTERN.search(normalized))
    negative = bool(NEGATIVE_FEEDBACK_PATTERN.search(normalized))
    if positive == negative:
        return "unknown"
    return "positive" if positive else "negative"


def extract_context_relations(
    response_id: object, context: object, style_cell: int
) -> list[dict[str, object]]:
    turns = parse_turns(context)
    records: list[dict[str, object]] = []
    for turn_index, (role, text) in enumerate(turns):
        if role != "student":
            continue
        relations = extract_relations(text)
        if not relations:
            continue
        next_feedback = "unknown"
        next_tutor_text = ""
        if turn_index + 1 < len(turns) and turns[turn_index + 1][0] == "tutor":
            next_tutor_text = turns[turn_index + 1][1]
            next_feedback = feedback_label(next_tutor_text)
        for relation_index, relation in enumerate(relations):
            records.append(
                {
                    "response_id": str(response_id),
                    "style_cell": int(style_cell),
                    "turn_index": int(turn_index),
                    "relation_index": int(relation_index),
                    "student_text": text,
                    "next_tutor_text": next_tutor_text,
                    "feedback": next_feedback,
                    **relation,
                }
            )
    return records


def synthetic_gate() -> dict[str, object]:
    valid_examples = (
        "2 + 3 = 5",
        "7 minus 4 is 3",
        "2 times 6 equals 12",
        "9 divided by 3 is 3",
        "0.5 plus 0.25 equals 0.75",
        "1/2 + 1/4 = 3/4",
        "4/12 is greater than 3/12",
        "2 < 3",
    )
    invalid_examples = (
        "2 + 3 = 6",
        "7 minus 4 is 2",
        "2 times 6 equals 13",
        "9 divided by 3 is 4",
        "0.5 plus 0.25 equals 0.8",
        "1/2 + 1/4 = 2/4",
        "3/12 is greater than 4/12",
        "3 < 2",
    )
    rejected_examples = (
        "9 divided by 0 is 4",
        "The date is 2026-07-28.",
        "50% plus 25% equals 75%.",
        "1 1/2 plus 1 equals 2 1/2.",
        "1000001 plus 1 equals 1000002.",
        "2 plus 3 plus 4 equals 9.",
        "I have 2 pencils and 3 books.",
    )
    valid_results = [extract_relations(value) for value in valid_examples]
    invalid_results = [extract_relations(value) for value in invalid_examples]
    rejected_results = [extract_relations(value) for value in rejected_examples]

    context = "\n".join(
        [
            "[OBJECTIVE] Adding numbers",
            "[STUDENT] 2 plus 3 equals 5.",
            "[TUTOR] Exactly, that is correct.",
            "[STUDENT] 4 plus 4 equals 9.",
            "[TUTOR] Not quite. Try again.",
            "[STUDENT] 5 plus 5 equals 10.",
            "[BACKGROUND] Noise",
            "[TUTOR] Correct.",
        ]
    )
    linked = extract_context_relations("synthetic", context, 0)
    duplicate = extract_context_relations("synthetic", context, 0)
    clauses = {
        "all_valid_examples_recovered": all(
            len(result) == 1 and result[0]["valid"] is True
            for result in valid_results
        ),
        "all_invalid_examples_recovered": all(
            len(result) == 1 and result[0]["valid"] is False
            for result in invalid_results
        ),
        "ambiguous_and_unsafe_examples_rejected": all(
            not result for result in rejected_results
        ),
        "immediate_positive_feedback_linked": (
            len(linked) == 3 and linked[0]["feedback"] == "positive"
        ),
        "immediate_negative_feedback_linked": (
            len(linked) == 3 and linked[1]["feedback"] == "negative"
        ),
        "background_breaks_immediate_link": (
            len(linked) == 3 and linked[2]["feedback"] == "unknown"
        ),
        "duplicate_input_is_decision_invariant": linked == duplicate,
        "target_column_is_prohibited": (
            "target" not in MODELING_IDENTITY_COLUMNS
            and "target" not in CONTEXT_COLUMNS
            and "target" not in ASSIGNMENT_COLUMNS
        ),
    }
    return {
        "valid_examples": valid_results,
        "invalid_examples": invalid_results,
        "rejected_examples": rejected_results,
        "linked_examples": linked,
        "clauses": clauses,
        "passes": bool(all(clauses.values())),
    }


def _style_lookup(assignments: pd.DataFrame) -> pd.Series:
    unique = assignments.loc[:, ASSIGNMENT_COLUMNS].drop_duplicates()
    counts = unique.groupby("response_id")["style_cell"].nunique()
    if not counts.eq(1).all():
        raise ValueError("E790 style-cell assignments are inconsistent.")
    return unique.drop_duplicates("response_id").set_index("response_id")[
        "style_cell"
    ]


def _load_target_free_frames(
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    paths = _project_paths(project_root)
    contexts = pd.read_parquet(paths["context"], columns=list(CONTEXT_COLUMNS))
    identities = pd.read_parquet(
        paths["modeling"], columns=list(MODELING_IDENTITY_COLUMNS)
    )
    assignments = pd.read_parquet(
        paths["assignments"], columns=list(ASSIGNMENT_COLUMNS)
    )
    if contexts["response_id"].duplicated().any():
        raise ValueError("E790 context source contains duplicate responses.")
    if identities["response_id"].duplicated().any():
        raise ValueError("E790 identity source contains duplicate responses.")
    alignment = contexts.merge(
        identities,
        on=["response_id", "session_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not alignment["_merge"].eq("both").all():
        raise ValueError("E790 context/identity alignment changed.")
    style = _style_lookup(assignments)
    if not contexts["response_id"].isin(style.index).all():
        raise ValueError("E790 is missing style assignments.")
    return contexts, style


def _finite(value: float) -> bool:
    return bool(math.isfinite(float(value)))


def evaluate_relations(
    relations: pd.DataFrame, total_responses: int
) -> dict[str, object]:
    if relations.empty:
        valid_count = 0
        invalid_count = 0
        linked = relations
    else:
        valid_count = int(relations["valid"].sum())
        invalid_count = int((~relations["valid"]).sum())
        linked = relations.loc[relations["feedback"].ne("unknown")].copy()

    distinct_responses = (
        int(relations["response_id"].nunique()) if not relations.empty else 0
    )
    linked_count = int(len(linked))
    style_counts = (
        linked.groupby("style_cell").size().sort_index()
        if not linked.empty
        else pd.Series(dtype=np.int64)
    )
    style_cells_with_ten = int((style_counts >= 10).sum())

    if linked_count and linked["valid"].nunique() == 2:
        validity = linked["valid"].astype(int).to_numpy()
        feedback_positive = (
            linked["feedback"].eq("positive").astype(int).to_numpy()
        )
        balanced_accuracy = float(
            balanced_accuracy_score(validity, feedback_positive)
        )
        mcc = float(matthews_corrcoef(validity, feedback_positive))
        valid_positive_rate = float(
            linked.loc[linked["valid"], "feedback"].eq("positive").mean()
        )
        invalid_positive_rate = float(
            linked.loc[~linked["valid"], "feedback"].eq("positive").mean()
        )
        rate_gap = valid_positive_rate - invalid_positive_rate
    else:
        balanced_accuracy = float("nan")
        mcc = float("nan")
        valid_positive_rate = float("nan")
        invalid_positive_rate = float("nan")
        rate_gap = float("nan")

    clauses = {
        "at_least_500_parsed_relations": len(relations) >= MIN_PARSED_RELATIONS,
        "at_least_250_distinct_responses": (
            distinct_responses >= MIN_DISTINCT_RESPONSES
        ),
        "at_least_50_valid_relations": (
            valid_count >= MIN_RELATIONS_PER_VALIDITY_CLASS
        ),
        "at_least_50_invalid_relations": (
            invalid_count >= MIN_RELATIONS_PER_VALIDITY_CLASS
        ),
        "at_least_200_feedback_linked_relations": (
            linked_count >= MIN_FEEDBACK_LINKED_RELATIONS
        ),
        "at_least_8_style_cells_with_10_links": (
            style_cells_with_ten >= MIN_STYLE_CELLS_WITH_TEN_LINKS
        ),
        "balanced_accuracy_at_least_0_70": (
            _finite(balanced_accuracy)
            and balanced_accuracy >= MIN_BALANCED_ACCURACY
        ),
        "matthews_correlation_at_least_0_30": (
            _finite(mcc) and mcc >= MIN_MATTHEWS_CORRELATION
        ),
        "positive_feedback_rate_gap_at_least_0_25": (
            _finite(rate_gap) and rate_gap >= MIN_POSITIVE_FEEDBACK_RATE_GAP
        ),
    }
    return {
        "total_responses": int(total_responses),
        "parsed_relations": int(len(relations)),
        "distinct_responses": distinct_responses,
        "response_coverage": (
            float(distinct_responses / total_responses) if total_responses else 0.0
        ),
        "valid_relations": valid_count,
        "invalid_relations": invalid_count,
        "feedback_linked_relations": linked_count,
        "style_cell_feedback_linked_counts": {
            str(int(key)): int(value) for key, value in style_counts.items()
        },
        "style_cells_with_at_least_10_links": style_cells_with_ten,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation": mcc,
        "valid_positive_feedback_rate": valid_positive_rate,
        "invalid_positive_feedback_rate": invalid_positive_rate,
        "positive_feedback_rate_gap": rate_gap,
        "clauses": clauses,
        "passes_discovery_gate": bool(all(clauses.values())),
    }


def run_audit(project_root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    runtime = assert_runtime()
    source_hashes = verify_sources(project_root)
    synthetic = synthetic_gate()
    if not synthetic["passes"]:
        raise RuntimeError("E790 synthetic gate failed.")

    contexts, style = _load_target_free_frames(project_root)
    records: list[dict[str, object]] = []
    for index, row in enumerate(contexts.itertuples(index=False), start=1):
        records.extend(
            extract_context_relations(
                row.response_id,
                row.objective_context,
                int(style.loc[row.response_id]),
            )
        )
        if index % 5_000 == 0:
            print(f"E790 target-free audit: {index}/{len(contexts)}", flush=True)
    relations = pd.DataFrame.from_records(records)
    evaluation = evaluate_relations(relations, len(contexts))
    elapsed = time.perf_counter() - started
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
        "runtime": runtime,
        "runtime_seconds": float(elapsed),
        "synthetic_gate": synthetic,
        "evaluation": evaluation,
        "competition_outcomes_accessed": False,
        "V_seen_outcomes_accessed": False,
        "V_objective_outcomes_accessed": False,
        "V_style_outcomes_accessed": False,
        "V_joint_accessed": False,
        "V_final_accessed": False,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    paths = _project_paths(project_root)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the target-free E790 mathematical-validity audit."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_audit(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
