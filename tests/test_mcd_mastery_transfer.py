from __future__ import annotations

import pandas as pd

from trace_ace.mcd_mastery_transfer import (
    TASK_SUFFIX,
    compact_dialogue,
    source_group,
    split_for_group,
)


def test_group_split_is_deterministic() -> None:
    assert source_group("123_abc") == "123"
    assert split_for_group("123") == split_for_group("123")
    assert split_for_group("123") in {"train", "test"}


def test_compaction_keeps_latest_twenty_four_turns() -> None:
    turns = [
        {"who": "teacher" if index % 2 == 0 else "student", "text": f" turn {index} "}
        for index in range(30)
    ]
    compacted = compact_dialogue(turns)
    lines = compacted.splitlines()
    assert len(lines) == 25
    assert lines[0] == "Tutor: turn 6"
    assert lines[-2] == "Student: turn 29"
    assert lines[-1] == TASK_SUFFIX


def test_compaction_rejects_unknown_speaker() -> None:
    try:
        compact_dialogue([{"who": "observer", "text": "hello"}])
    except ValueError as error:
        assert "unknown speaker" in str(error)
    else:
        raise AssertionError("Unknown E610 speaker was accepted.")


def test_real_source_contract() -> None:
    labels = pd.read_csv("Datasets/MCD/data/dataset/df_feature_num_label-3.csv")
    labels["source_group"] = labels["new_id"].map(source_group)
    labels["split"] = labels["source_group"].map(split_for_group)
    assert labels["split"].value_counts().to_dict() == {"train": 4_242, "test": 984}
    assert labels.groupby("split")["source_group"].nunique().to_dict() == {
        "test": 95,
        "train": 399,
    }
