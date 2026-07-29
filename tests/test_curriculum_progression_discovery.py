from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.curriculum_progression_discovery import (
    build_objective_features,
    curriculum_leave_one_out,
    evaluate_anchors,
    parse_curriculum,
)


def test_parse_curriculum_tracks_stage_and_domain(tmp_path: Path) -> None:
    source = tmp_path / "curriculum.html"
    source.write_text(
        """
        <h2>Year 1 programme of study</h2>
        <h3>Number - fractions</h3>
        <ul><li>recognise and find one half of an object or quantity</li></ul>
        <h2>Year 2 programme of study</h2>
        <h3>Number - multiplication and division</h3>
        <p>Pupils should recall the 2, 5 and 10 multiplication tables.</p>
        """,
        encoding="utf-8",
    )
    frame = parse_curriculum(source)
    assert list(frame["stage"]) == ["year_1", "year_2"]
    assert list(frame["stage_index"]) == [1, 2]
    assert list(frame["domain"]) == [
        "Number - fractions",
        "Number - multiplication and division",
    ]


def test_curriculum_leave_one_out_uses_no_self_match() -> None:
    stages = np.repeat(np.arange(1, 9), 2)
    standards = pd.DataFrame(
        {
            "stage": [f"stage_{stage}" for stage in stages],
            "stage_index": stages,
        }
    )
    similarity = np.full((16, 16), 0.1)
    np.fill_diagonal(similarity, 1.0)
    for row in range(0, 16, 2):
        similarity[row, row + 1] = 0.9
        similarity[row + 1, row] = 0.9
    report = curriculum_leave_one_out(standards, similarity)
    assert report["top1_accuracy"] == 1.0
    assert report["mean_absolute_stage_error"] < 0.01


def test_build_objective_features_has_frozen_stage_columns() -> None:
    objectives = pd.DataFrame(
        {"learning_objective_id": ["a"], "learning_objective": ["fractions"]}
    )
    standards = pd.DataFrame(
        {
            "stage": [
                "year_1",
                "year_2",
                "year_3",
                "year_4",
                "year_5",
                "year_6",
                "key_stage_3",
                "key_stage_4",
            ],
            "stage_index": np.arange(1, 9),
            "domain": ["number"] * 8,
            "statement": [f"statement {index}" for index in range(8)],
        }
    )
    similarity = np.arange(8, dtype=np.float64)[None, :] / 10.0
    features, scores = build_objective_features(
        objectives, standards, similarity
    )
    assert scores.shape == (1, 8)
    assert int(features.loc[0, "curriculum_best_stage"]) == 8
    assert features.loc[0, "nearest_curriculum_stage"] == "key_stage_4"


def test_frozen_anchor_table_has_no_duplicates() -> None:
    from trace_ace.curriculum_progression_discovery import ANCHORS

    rows = pd.DataFrame(
        {
            "learning_objective": [anchor.objective for anchor in ANCHORS],
            "curriculum_best_stage": [
                anchor.minimum_stage for anchor in ANCHORS
            ],
            "curriculum_expected_stage": [
                (anchor.minimum_stage + anchor.maximum_stage) / 2
                for anchor in ANCHORS
            ],
        }
    )
    _, summary = evaluate_anchors(rows)
    assert len({anchor.objective.casefold() for anchor in ANCHORS}) == len(ANCHORS)
    assert summary["anchors"] == 24
    assert summary["in_range_fraction"] == 1.0
