import pytest


@pytest.fixture(params=[8, 16])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 8])
def embed_dim(request):
    return request.param
