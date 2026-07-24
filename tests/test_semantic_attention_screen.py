import numpy as np

from trace_ace.semantic_attention_screen import (
    ATTENTION_TEMPERATURE,
    EMBEDDING_DIMENSION,
    SEGMENT_COUNT,
    build_segment_frame,
    segment_objective_context,
    semantic_attention,
)


def test_segment_objective_context_is_chronological_and_bounded() -> None:
    lines = ["[OBJECTIVE] divide fractions"] + [
        f"[STUDENT] line {index} extra words" for index in range(40)
    ]
    segments = segment_objective_context("\n".join(lines))
    assert len(segments) == SEGMENT_COUNT
    assert all(segment.startswith("[OBJECTIVE] divide fractions\n") for segment in segments)
    assert all(len(segment.splitlines()) == 9 for segment in segments)
    assert "line 0 " in segments[0]
    assert "line 9 " in segments[0]
    assert "line 10 " in segments[1]
    assert "line 39 " in segments[3]


def test_real_pilot_segment_contract() -> None:
    frame = build_segment_frame(".")
    assert frame.shape == (4_096, 5)
    assert frame["response_id"].is_unique
    assert not frame.drop(columns="response_id").eq("").any().any()


def test_semantic_attention_is_normalized_and_objective_weighted() -> None:
    objective = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
    objective[:, 0] = 1.0
    segments = np.zeros(
        (2, SEGMENT_COUNT, EMBEDDING_DIMENSION), dtype=np.float32
    )
    segments[:, :, 1] = 1.0
    segments[0, 2] = objective[0]
    segments[1, 3] = objective[1]
    pooled, weights, similarities = semantic_attention(segments, objective)
    assert ATTENTION_TEMPERATURE == 10.0
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0].argmax() == 2
    assert weights[1].argmax() == 3
    assert similarities[0].argmax() == 2
    assert similarities[1].argmax() == 3
