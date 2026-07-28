import numpy as np
import pandas as pd
from scipy import sparse

from trace_ace.session_mastery_screen import (
    session_soft_targets,
    soft_label_examples,
)


def test_session_soft_targets_use_only_legal_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["a", "a", "b", "b", "c"],
            "target": [1, 0, 1, 1, 0],
        }
    )
    train_mask = np.array([True, True, True, False, False])
    grouped = session_soft_targets(frame, train_mask)
    assert grouped["session_id"].tolist() == ["a", "b"]
    assert grouped["soft_target"].tolist() == [0.5, 1.0]
    assert grouped["response_count"].tolist() == [2, 1]


def test_soft_label_examples_assign_equal_total_session_weight() -> None:
    matrix = sparse.csr_matrix(np.eye(3, dtype=np.float32))
    examples, labels, weights = soft_label_examples(
        matrix, np.array([0.0, 0.25, 1.0])
    )
    assert examples.shape == (6, 3)
    assert labels.tolist() == [1, 1, 1, 0, 0, 0]
    np.testing.assert_allclose(weights, [0.0, 0.25, 1.0, 1.0, 0.75, 0.0])
    np.testing.assert_allclose(weights[:3] + weights[3:], np.ones(3))
