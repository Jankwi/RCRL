import math

import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.contrastivelib.metrics.sims import (
    DotProductSimilarityMeasure,
    L1SimilarityMeasure,
    L2SimilarityMeasure,
    L2SquaredSimilarityMeasure,
)


@pytest.fixture(params=[1, 3, 10])
def latent_dim(request):
    """Provides latent dimension sizes for cross-similarity tests."""
    return request.param


@pytest.fixture(params=[(1, 4), (3, 5), (5, 3), (4, 1)])
def cross_dims(request):
    """Provides tuples of a.shape[0], b.shape[0] for cross-similarity tests"""
    return request.param


@pytest.fixture(params=[True, False])
def normalize(request):
    """Provides normalization flag for cross-similarity tests."""
    return request.param


@pytest.fixture
def cross_sim_data(device_fx, cross_dims, latent_dim, normalize):
    """Provides deterministic tensors and a build_expected callable for cross-similarity tests."""
    torch.manual_seed(42)
    len_a, len_b = cross_dims
    a = torch.rand((len_a, latent_dim), device=device_fx)
    b = torch.rand((len_b, latent_dim), device=device_fx)

    def build_expected(sim_fn):
        expected = torch.zeros((len_a, len_b), device=device_fx)
        for i in range(len_a):
            for j in range(len_b):
                expected[i][j] = sim_fn(a[i].unsqueeze(0), b[j].unsqueeze(0), 'elementwise').item()
        return expected

    return a, b, normalize, build_expected


class TestL2SimilarityMeasure:
    """Tests for the L2 (Euclidean distance) similarity measure."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize(
        'a, b, normalize, expected',
        [
            ([[1.0, 0.0]], [[0.0, 1.0]], False, [math.sqrt(2)]),
            ([[1.0, 0.0]], [[0.0, 1.0]], True, [math.sqrt(2)]),
            (
                [[3.0, 2.0, 5.0], [2.0, 1.0, 7.0]],
                [[3.0, 0.0, 5.0], [1.0, -2.0, 5.0]],
                False,
                [2.0, math.sqrt(14)],
            ),
            (
                [[2 * math.sqrt(2), 2 * math.sqrt(2)], [0.0, 1.0]],
                [[1.0, 1.0], [1, 0]],
                True,
                [0.0, math.sqrt(2)],
            ),
        ],
    )
    def test_l2_elementwise_similarity(self, a, b, normalize, expected, device_fx):
        a = torch.tensor(a, device=device_fx)
        b = torch.tensor(b, device=device_fx)
        expected = torch.tensor(expected, device=device_fx)

        sim_fn = L2SimilarityMeasure(normalize)
        assert_close(sim_fn(a, b, 'elementwise'), expected)

    def test_l2_cross_similarity(self, cross_sim_data):
        """Verifies cross-mode L2 similarity against repeated elementwise calls."""
        a, b, normalize, build_expected = cross_sim_data
        sim_fn = L2SimilarityMeasure(normalize)
        expected = build_expected(sim_fn)
        assert_close(sim_fn(a, b, 'cross'), expected)


class TestDotProductSImilarityMeasure:
    """Tests for the negative dot product similarity measure."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize(
        'a, b, normalize, expected',
        [
            ([[1.0, 0.0]], [[1.0, 1.0]], False, [-1.0]),
            (
                # The angles are 45, 180, and 90 degrees respesctively
                [[1.0, 0.0], [-math.sqrt(2), math.sqrt(2)], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, -1.0], [0.0, -1.0]],
                True,
                [-math.sqrt(2) / 2, 1, 0],
            ),
            (
                # The angles are 30 and 90 degrees respectively
                [[1.0, 0.0, 0.0], [1, 0, math.sqrt(3)]],
                [[0.0, 1.0, 0.0], [math.sqrt(3), 0, 1]],
                False,
                [0.0, -2 * math.sqrt(3)],
            ),
        ],
    )
    def test_dot_product_elementwise_similarity(self, a, b, normalize, expected, device_fx):
        a = torch.tensor(a, device=device_fx)
        b = torch.tensor(b, device=device_fx)
        expected = torch.tensor(expected, device=device_fx)

        sim_fn = DotProductSimilarityMeasure(normalize)
        assert_close(sim_fn(a, b, 'elementwise'), expected)

    def test_dot_product_cross_similarity(self, cross_sim_data):
        """Verifies cross-mode dot-product similarity against repeated elementwise calls."""
        a, b, normalize, build_expected = cross_sim_data
        sim_fn = DotProductSimilarityMeasure(normalize)
        expected = build_expected(sim_fn)
        assert_close(sim_fn(a, b, 'cross'), expected)


class TestL1SimilarityMeasure:
    """Tests for the L1 (Manhattan distance) similarity measure."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize(
        'a, b, normalize, expected',
        [
            ([[1.0, 0.0]], [[0.0, 1.0]], False, [2.0]),
            (
                [[3.0, 2.0, 5.0], [2.0, 1.0, 7.0]],
                [[3.0, 0.0, 5.0], [1.0, -2.0, 5.0]],
                False,
                [2.0, 6.0],
            ),
            ([[2.0, 2.0], [0.0, 3.0]], [[1.0, 0.0], [3.0, 0.0]], True, [1.0, 2.0]),
        ],
    )
    def test_l1_elementwise_similarity(self, a, b, normalize, expected, device_fx):
        """Verifies elementwise L1 similarity against hand-crafted expected values."""
        a = torch.tensor(a, device=device_fx)
        b = torch.tensor(b, device=device_fx)
        expected = torch.tensor(expected, device=device_fx)

        sim_fn = L1SimilarityMeasure(normalize)
        assert_close(sim_fn(a, b, 'elementwise'), expected)

    def test_l1_cross_similarity(self, cross_sim_data):
        """Verifies cross-mode L1 similarity against repeated elementwise calls."""
        a, b, normalize, build_expected = cross_sim_data
        sim_fn = L1SimilarityMeasure(normalize)
        expected = build_expected(sim_fn)
        assert_close(sim_fn(a, b, 'cross'), expected)


class TestL2SquaredSimilarityMeasure:
    """Tests for the squared Euclidean distance similarity measure."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize(
        'a, b, normalize, expected',
        [
            ([[1.0, 0.0]], [[0.0, 1.0]], False, [2.0]),
            (
                [[3.0, 2.0, 5.0], [2.0, 1.0, 7.0]],
                [[3.0, 0.0, 5.0], [1.0, -2.0, 5.0]],
                False,
                [4.0, 14.0],
            ),
            (
                [[2 * math.sqrt(2), 2 * math.sqrt(2)], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                True,
                [0.0, 2.0],
            ),
        ],
    )
    def test_l2_squared_elementwise_similarity(self, a, b, normalize, expected, device_fx):
        """Verifies elementwise squared L2 similarity against hand-crafted expected values."""
        a = torch.tensor(a, device=device_fx)
        b = torch.tensor(b, device=device_fx)
        expected = torch.tensor(expected, device=device_fx)

        sim_fn = L2SquaredSimilarityMeasure(normalize)
        assert_close(sim_fn(a, b, 'elementwise'), expected)

    def test_l2_squared_cross_similarity(self, cross_sim_data):
        """Verifies cross-mode squared L2 similarity against repeated elementwise calls."""
        a, b, normalize, build_expected = cross_sim_data
        sim_fn = L2SquaredSimilarityMeasure(normalize)
        expected = build_expected(sim_fn)
        assert_close(sim_fn(a, b, 'cross'), expected)
