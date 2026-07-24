import numpy as np

from trace_ace.mpnet_family_screen import DIMENSION, _interaction


def test_mpnet_interaction_shape_and_similarity() -> None:
    context = np.zeros((2, DIMENSION), dtype=np.float32)
    objective = np.zeros_like(context)
    context[:, 0] = 1
    objective[0, 0] = 1
    objective[1, 1] = 1
    features, similarity = _interaction(context, objective)
    assert features.shape == (2, 4 * DIMENSION)
    np.testing.assert_allclose(similarity[:, 0], [1, 0])
