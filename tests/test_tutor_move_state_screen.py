from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.tutor_move_state_screen import (
    FEATURE_NAMES,
    MOVE_NAMES,
    STATE_NAMES,
    aggregate_events,
    align_features,
    classify_student_state,
    classify_tutor_move,
    extract_context_features,
    extract_events,
    parse_context,
    synthetic_gate,
)


def test_synthetic_move_and_state_gate_passes() -> None:
    result = synthetic_gate()
    assert result["passes"]
    assert all(row["passes"] for row in result["move_cases"])
    assert all(row["passes"] for row in result["state_cases"])


def test_parser_ignores_background_and_preserves_dialogue_order() -> None:
    context = "\n".join(
        [
            "[OBJECTIVE] Adding two digit numbers",
            "[BACKGROUND] noise",
            "[STUDENT] I added the tens because they are easier.",
            "[TUTOR] Why did you add the tens first?",
            "[STUDENT] Because four tens plus three tens is seven tens.",
        ]
    )
    objective, turns = parse_context(context)
    assert objective == "Adding two digit numbers"
    assert [role for role, _ in turns] == ["student", "tutor", "student"]
    events = extract_events(turns, objective)
    assert len(events) == 1
    assert events[0]["move"]["prompting_self_explanation"] == 1.0
    assert events[0]["state"]["expressed_reasoning"] == 1.0
    assert events[0]["interaction"]["press_reasoning_after_strong"] == 1.0


def test_asr_noise_does_not_change_expected_detection() -> None:
    clean_move = classify_tutor_move(
        "I got twelve.", "Can you explain why you chose twelve?"
    )
    noisy_move = classify_tutor_move(
        "Um, I got [unclear] twelve.",
        "Uh, can you explain [unclear] why you chose twelve?",
    )
    assert clean_move["prompting_self_explanation"] == 1.0
    assert noisy_move["prompting_self_explanation"] == 1.0
    clean_state = classify_student_state(
        "I chose twelve because three times four is twelve.",
        "Multiplying whole numbers",
    )
    noisy_state = classify_student_state(
        "Um, I chose [unclear] twelve because three times four is twelve.",
        "Multiplying whole numbers",
    )
    assert clean_state["expressed_reasoning"] == noisy_state["expressed_reasoning"] == 1.0


def test_aggregate_feature_schema_is_finite_and_frozen() -> None:
    context = "\n".join(
        [
            "[OBJECTIVE] Adding decimals",
            "[STUDENT] I added 1.2 and 2.3 because I aligned the decimals.",
            "[TUTOR] Good thinking and a great strategy.",
            "[STUDENT] Then I added each place and got 3.5.",
            "[TUTOR] Exactly, that is correct.",
        ]
    )
    values = extract_context_features(context)
    assert tuple(values) == FEATURE_NAMES
    assert len(values) == 134
    assert np.isfinite(list(values.values())).all()
    empty = aggregate_events([])
    assert tuple(empty) == FEATURE_NAMES
    assert np.isfinite(list(empty.values())).all()


def test_role_and_turn_order_change_interaction_features() -> None:
    objective = "Adding numbers"
    first = [
        ("student", "I added the tens because four plus three is seven."),
        ("tutor", "Why did you add the tens first?"),
        ("student", "Because that gives seventy."),
    ]
    reversed_roles = [
        ("tutor", "I added the tens because four plus three is seven."),
        ("student", "Why did you add the tens first?"),
        ("tutor", "Because that gives seventy."),
    ]
    first_values = aggregate_events(extract_events(first, objective))
    reversed_values = aggregate_events(extract_events(reversed_roles, objective))
    assert (
        first_values["interaction_press_reasoning_after_strong_all_rate"]
        != reversed_values["interaction_press_reasoning_after_strong_all_rate"]
    )


def test_feature_alignment_is_sample_independent() -> None:
    component = pd.DataFrame({"response_id": ["b", "a"]})
    rows = []
    for response_id, offset in (("a", 1.0), ("b", 2.0), ("unrelated", 99.0)):
        row = {"response_id": response_id, "session_id": response_id}
        row.update({name: offset for name in FEATURE_NAMES})
        rows.append(row)
    features = pd.DataFrame(rows)
    aligned = align_features(component, features)
    assert aligned.shape == (2, len(FEATURE_NAMES))
    assert np.all(aligned[0] == 2.0)
    assert np.all(aligned[1] == 1.0)
    assert set(classify_tutor_move("twelve", "The answer is 12.")) == set(MOVE_NAMES)
    assert set(classify_student_state("twelve", "numbers")) == set(STATE_NAMES)
