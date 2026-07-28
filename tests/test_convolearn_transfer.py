from __future__ import annotations

import pandas as pd

from trace_ace.convolearn_transfer import (
    TASK_LINE,
    _audit_protocols,
    compact_dialogue,
    protocol_splits,
    subdimension_fold,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for topic_index, topic in enumerate(("a", "b", "c", "d")):
            rows.append(
                {
                    "subdimension": f"sub-{fold}",
                    "topic": topic,
                    "subdimension_fold": fold,
                    "dimension": f"dim-{topic_index}",
                }
            )
    return pd.DataFrame(rows)


def test_compact_dialogue_is_target_free_and_bounded() -> None:
    source = "\n".join(f"Student: line {index} extra words" for index in range(25))
    text = compact_dialogue(source)
    lines = text.splitlines()
    assert lines[-1] == TASK_LINE
    assert len(lines) <= 11
    assert all(len(line.split()) <= 16 for line in lines[:-1])
    assert "rating" not in text.lower()


def test_subdimension_fold_is_stable() -> None:
    assert subdimension_fold("Reflective Growth") == subdimension_fold(
        "Reflective Growth"
    )
    assert 0 <= subdimension_fold("Scaffolding") < 5


def test_protocol_splits_hold_out_the_declared_group() -> None:
    frame = _frame()
    for protocol, column in (
        ("subdimension_disjoint", "subdimension"),
        ("topic_disjoint", "topic"),
    ):
        for _, train, valid in protocol_splits(frame, protocol):
            assert set(frame.loc[train, column]).isdisjoint(
                set(frame.loc[valid, column])
            )


def test_protocol_audit_accepts_disjoint_fixture() -> None:
    audit = _audit_protocols(_frame())
    assert len(audit["subdimension_disjoint"]) == 5
    assert len(audit["topic_disjoint"]) == 4
    assert all(
        row["held_out_group_overlap"] == 0
        for protocol in audit.values()
        for row in protocol
    )
