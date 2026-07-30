import numpy as np
import pandas as pd
import pytest
import torch

from trace_ace.colbert_late_interaction import (
    BENCHMARK_ROWS,
    colbert_score_matrix,
    insert_marker,
    retrieval_rows_from_scores,
    select_benchmark_rows,
)


def test_insert_marker_after_first_token() -> None:
    values = torch.tensor([[101, 200, 201, 102], [101, 300, 301, 102]])
    marked = insert_marker(values, marker_id=1)
    assert marked.tolist() == [
        [101, 1, 200, 201, 102],
        [101, 1, 300, 301, 102],
    ]


def test_colbert_score_is_sum_of_query_token_maxima() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    documents = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    scores = colbert_score_matrix(query, documents, mask)
    np.testing.assert_allclose(scores, [[2.0, 1.0]])


def test_retrieval_rows_accept_multiple_positive_documents() -> None:
    scores = np.array([[0.1, 0.8, 0.9], [0.7, 0.2, 0.1]])
    rows = retrieval_rows_from_scores(
        scores,
        ["a", "b"],
        ["b", "a", "a"],
    )
    assert rows["reciprocal_rank"].tolist() == [1.0, 1.0]
    assert rows["hit_at_1"].tolist() == [1, 1]


def test_retrieval_rows_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        retrieval_rows_from_scores(
            np.zeros((2, 3)),
            ["a"],
            ["a", "b", "c"],
        )


def test_benchmark_selection_spans_conversation_lengths() -> None:
    frame = pd.DataFrame(
        {"conversation": ["x" * value for value in range(1, 600)]}
    )
    selected = select_benchmark_rows(frame)
    assert selected.shape == (BENCHMARK_ROWS,)
    assert len(np.unique(selected)) == BENCHMARK_ROWS
    assert selected[0] == 0
    assert selected[-1] == 598
