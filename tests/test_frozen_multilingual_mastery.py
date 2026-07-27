from __future__ import annotations

import torch

from trace_ace.frozen_multilingual_mastery import mean_pool


def test_mean_pool_masks_padding_and_normalizes() -> None:
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    assert torch.allclose(torch.linalg.vector_norm(pooled, dim=1), torch.ones(1))
    assert torch.allclose(pooled, torch.tensor([[0.6, 0.8]]))
