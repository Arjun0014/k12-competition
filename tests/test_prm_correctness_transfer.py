from __future__ import annotations

import json

import numpy as np

from trace_ace.prm_correctness_transfer import (
    HYPOTHESIS,
    _canonical_record,
    _premise,
    _prm_metrics,
    _question_bootstrap,
)


def test_prm_canonical_record_and_compaction() -> None:
    key, raw, record = _canonical_record(
        " ".join(f"p{i}" for i in range(100)),
        " ".join(f"g{i}" for i in range(100)),
        [" ".join(f"r{i}" for i in range(100))],
        " ".join(f"c{i}" for i in range(100)),
        -1,
    )
    assert len(key) == 32
    assert json.loads(raw)["rating"] == -1
    premise = _premise(record)
    assert premise.startswith("[CANDIDATE STEP] c0")
    assert "c63 [PROBLEM]" in premise
    assert "p63 [GROUND TRUTH SOLUTION]" in premise
    assert "g79 [PRIOR REASONING] r52" in premise
    assert premise.endswith("r99")
    assert "mathematically correct" in HYPOTHESIS


def test_prm_metrics_and_question_bootstrap_are_deterministic() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    base = np.full((6, 3), 1.0 / 3.0)
    candidate = np.eye(3, dtype=np.float64)[labels] * 0.8 + 0.2 / 3.0
    metrics = _prm_metrics(labels, candidate)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    first = _question_bootstrap(
        labels,
        base,
        candidate,
        np.asarray(["a", "a", "b", "b", "c", "c"]),
    )
    second = _question_bootstrap(
        labels,
        base,
        candidate,
        np.asarray(["a", "a", "b", "b", "c", "c"]),
    )
    assert first == second
    assert first["probability_gain_positive"] == 1.0
