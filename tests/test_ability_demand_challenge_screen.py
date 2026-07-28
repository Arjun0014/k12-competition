from __future__ import annotations

import numpy as np
import pandas as pd

from trace_ace.ability_demand_challenge_screen import (
    FEATURE_NAMES,
    aggregate_episodes,
    align_rasch_gap,
    analyze_episode,
    estimate_task_demand,
    extract_context_features,
    fit_rasch_calibration,
    parse_context,
    segment_task_episodes,
    synthetic_gate,
)


def test_synthetic_gate_recovers_states_and_orderings() -> None:
    result = synthetic_gate()
    assert result["passes"]
    assert all(result["clauses"].values())


def test_episode_segmentation_is_chronological_and_bounded() -> None:
    context = "\n".join(
        [
            "[OBJECTIVE] Adding numbers",
            "[BACKGROUND] noise",
            "[TUTOR] What is 2 plus 3?",
            "[STUDENT] It is 5.",
            "[TUTOR] Correct.",
            "[TUTOR] Explain why 4 plus 5 is 9?",
            "[STUDENT] Because four and five make nine.",
        ]
    )
    objective, turns = parse_context(context)
    episodes = segment_task_episodes(turns)
    assert objective == "Adding numbers"
    assert len(episodes) == 2
    assert episodes[0][0] == ("tutor", "What is 2 plus 3?")
    assert episodes[1][0] == ("tutor", "Explain why 4 plus 5 is 9?")
    assert all(len(episode) <= 11 for episode in episodes)


def test_task_demand_uses_prompt_and_objective_content() -> None:
    objective = "Adding two digit numbers"
    easy = estimate_task_demand("What is 2 plus 2?", objective)
    hard = estimate_task_demand(
        "Explain and justify with a diagram and equation why both steps work.",
        objective,
    )
    assert 0.0 < easy < hard < 1.0


def test_challenge_aggregate_is_duplication_invariant_for_model_gap() -> None:
    objective = "Adding numbers"
    episode = analyze_episode(
        [
            ("tutor", "Explain why 24 plus 13 equals 37?"),
            ("student", "I am not sure."),
            ("tutor", "Not quite. Think about the tens."),
            ("student", "Actually, the tens make thirty and the ones make seven."),
            ("tutor", "Correct."),
        ],
        objective,
    )
    single = aggregate_episodes([episode])
    duplicate = aggregate_episodes([episode, episode])
    for name in (
        "ability_mean",
        "demand_mean",
        "under_challenged_proportion",
        "optimally_challenged_proportion",
        "over_challenged_proportion",
        "rasch_gap",
    ):
        assert np.isclose(single[name], duplicate[name])


def test_feature_schema_is_finite_and_empty_safe() -> None:
    empty = aggregate_episodes([])
    assert tuple(empty) == FEATURE_NAMES
    assert len(empty) == 25
    assert np.isfinite(list(empty.values())).all()
    context = "\n".join(
        [
            "[OBJECTIVE] Adding numbers",
            "[TUTOR] What is 2 plus 3?",
            "[STUDENT] It is 5 because two and three make five.",
            "[TUTOR] Correct.",
        ]
    )
    values = extract_context_features(context)
    assert tuple(values) == FEATURE_NAMES
    assert np.isfinite(list(values.values())).all()


def test_rasch_calibration_has_nonnegative_slope_and_monotonic_prediction() -> None:
    gap = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    target = np.array([0, 0, 0, 1, 1, 1])
    validation = np.array([-1.5, 0.0, 1.5])
    prediction, summary = fit_rasch_calibration(gap, target, validation)
    assert summary["slope"] >= 0.0
    assert np.all(np.diff(prediction) >= 0.0)


def test_alignment_is_response_ordered_and_ignores_unrelated_rows() -> None:
    component = pd.DataFrame({"response_id": ["b", "a"]})
    feature_frame = pd.DataFrame(
        {
            "response_id": ["a", "b", "unrelated"],
            "rasch_gap": [1.0, 2.0, 99.0],
        }
    )
    actual = align_rasch_gap(component, feature_frame)
    np.testing.assert_array_equal(actual, np.array([2.0, 1.0]))
