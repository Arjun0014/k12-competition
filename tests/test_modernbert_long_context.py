import numpy as np
import pytest

from trace_ace.modernbert_long_context import (
    BENCHMARK_ROWS,
    mastery_hypothesis,
    select_benchmark_indices,
    select_context_length,
)


def test_mastery_hypothesis_normalizes_whitespace() -> None:
    assert (
        mastery_hypothesis("  add   unlike fractions ")
        == "The student demonstrates mastery of this learning objective: "
        "add unlike fractions."
    )


def test_benchmark_indices_span_length_quantiles() -> None:
    lengths = np.arange(4096, dtype=np.int32)[::-1]
    selected = select_benchmark_indices(lengths)
    assert selected.shape == (BENCHMARK_ROWS,)
    assert len(np.unique(selected)) == BENCHMARK_ROWS
    selected_lengths = np.sort(lengths[selected])
    assert selected_lengths[0] == 0
    assert selected_lengths[-1] == 4095


def test_context_selection_uses_longest_eligible_length() -> None:
    rows = [
        {
            "max_length": 8192,
            "projected_hours_for_4096": 12.0,
            "peak_rss_bytes": 2_000_000_000,
            "completed": True,
        },
        {
            "max_length": 4096,
            "projected_hours_for_4096": 7.9,
            "peak_rss_bytes": 3_000_000_000,
            "completed": True,
        },
        {
            "max_length": 2048,
            "projected_hours_for_4096": 4.0,
            "peak_rss_bytes": 2_000_000_000,
            "completed": True,
        },
    ]
    assert select_context_length(rows) == 4096


def test_context_selection_rejects_no_eligible_length() -> None:
    rows = [
        {
            "max_length": value,
            "projected_hours_for_4096": 9.0,
            "peak_rss_bytes": 2_000_000_000,
            "completed": True,
        }
        for value in (8192, 4096, 2048)
    ]
    with pytest.raises(RuntimeError, match="No frozen E420 context length"):
        select_context_length(rows)
