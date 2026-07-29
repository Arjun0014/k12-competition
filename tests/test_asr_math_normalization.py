from __future__ import annotations

from trace_ace.asr_math_normalization import (
    corrupt_asr_like,
    normalize_math_speech,
)


def test_normalizer_canonicalizes_explicit_math_and_noise() -> None:
    raw = "[STUDENT] Um, [unclear] twelve twelve divided by three equals four."
    assert normalize_math_speech(raw) == (
        "[student] mathnum12 mathopdivide mathnum3 mathopequal mathnum4"
    )


def test_normalizer_leaves_ambiguous_homophones_untouched() -> None:
    normalized = normalize_math_speech(
        "[TUTOR] Go to the board for some practice."
    )
    assert "to" in normalized
    assert "for" in normalized
    assert "some" in normalized
    assert "mathnum" not in normalized


def test_corruption_is_deterministic_and_normalization_recovers_math() -> None:
    clean = "2 + 3 = 5"
    first = corrupt_asr_like(clean, "row-1")
    second = corrupt_asr_like(clean, "row-1")
    assert first == second
    assert normalize_math_speech(first) == normalize_math_speech(clean)
