import pytest
import torch


@pytest.mark.smoke
def test_tensor_basic():
    assert torch.ones(3, 2).sum().item() == 6.0
