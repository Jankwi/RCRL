"""Common pytest fixtures and utilities for all tests."""

import pytest
import torch


@pytest.fixture(params=['cpu', 'cuda'])
def device_fx(request):
    """Fixture providing CPU and CUDA devices, skipping CUDA if unavailable."""
    device_type = request.param
    if device_type == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available on this system.')
    return torch.device(device_type)
