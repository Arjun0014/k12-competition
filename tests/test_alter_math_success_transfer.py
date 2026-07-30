from __future__ import annotations

import numpy as np

from trace_ace.alter_math_success_transfer import (
    _applicability,
    _evidence_from_margin,
    _external_bootstrap,
    build_candidate,
    compact_document,
    source_fold,
)


def test_compact_document_preserves_frozen_roles_and_ends() -> None:
    document = (
        "u1: I need help with fractions * "
        "p1: Try a common denominator * "
        "e1: What denominator works for both? * "
        "u1: Six works for both fractions * "
        "e1: Good, now rewrite each numerator"
    )
    text = compact_document(document)
    assert text.startswith("Task: represent whether")
    assert "[STUDENT] I need help with fractions" in text
    assert "[PEER_TUTOR] Try a common denominator" in text
    assert "[EXPERT_TUTOR] Good, now rewrite each numerator" in text
    assert "Earlier mathematics discussion:" not in text


def test_source_fold_is_deterministic_and_bounded() -> None:
    observed = [source_fold(value) for value in ("1187844", "188427", "42")]
    assert observed == [source_fold(value) for value in ("1187844", "188427", "42")]
    assert all(0 <= value < 5 for value in observed)


def test_external_evidence_is_centered_and_candidate_is_monotonic() -> None:
    margin = np.array([-2.0, 0.0, 2.0])
    z, evidence = _evidence_from_margin(margin, median=0.0, iqr=1.0)
    assert np.allclose(z, margin)
    assert evidence[1] == 0.0
    baseline = np.full(3, 0.7)
    candidate = build_candidate(baseline, evidence, gamma=0.3)
    assert candidate[0] < candidate[1] < candidate[2]
    assert np.isclose(candidate[1], baseline[1])


def test_applicability_gate_accepts_noncollapsed_evidence() -> None:
    evidence = np.linspace(-0.8, 0.8, 35_072)
    report = _applicability(evidence)
    assert report["passes_applicability_gate"] is True
    assert all(report["clauses"].values())


def test_external_bootstrap_is_deterministic() -> None:
    target = np.array([0, 1] * 20, dtype=np.int8)
    baseline = np.full(len(target), 0.5)
    candidate = np.where(target == 1, 0.7, 0.3)
    first = _external_bootstrap(
        target, baseline, candidate, replicates=200, seed=7
    )
    second = _external_bootstrap(
        target, baseline, candidate, replicates=200, seed=7
    )
    assert first == second
    assert first["support_positive_gain"] == 1.0
