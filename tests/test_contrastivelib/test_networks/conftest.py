import pytest
import torch


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    """Provides available compute devices, skipping CUDA if unavailable."""
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device(request.param)


@pytest.fixture(params=[8, 16])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 8])
def embed_dim(request):
    return request.param
