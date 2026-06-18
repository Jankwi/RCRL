import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler


def _marginal_t1_expected(length: int, gamma: float, geom_type: str, device) -> torch.Tensor:
    """Analytically computes P(t1 = t) for t in [1, length-1].

    t0 ~ uniform[0, length-2]; given t0=k, delta ~ geom_type over [1, length-1-k].
    """

    def _bounded_pmf(n):
        probs = torch.tensor(
            [(1 - gamma) * gamma ** (k - 1) for k in range(1, n + 1)],
            dtype=torch.float,
            device=device,
        )
        return probs / probs.sum()

    def _clamped_pmf(n):
        probs = [(1 - gamma) * gamma ** (k - 1) for k in range(1, n)]
        return torch.tensor(probs + [gamma ** (n - 1)], dtype=torch.float, device=device)

    pmf_fn = _bounded_pmf if geom_type == 'bounded' else _clamped_pmf
    n_t0 = length - 1
    expected = torch.zeros(length, dtype=torch.float, device=device)
    for k in range(n_t0):
        delta_max = length - 1 - k
        pmf = pmf_fn(delta_max)
        for d in range(1, delta_max + 1):
            expected[k + d] += pmf[d - 1].item() / n_t0
    return expected[1:]  # drop unused index 0


class _Stub(AbstractContrastiveDataset):
    def contrastive_pick(self, **kwargs):
        pass


def _make_stub(geom_type: str, device, gamma: float = 0.9):
    stub = object.__new__(_Stub)
    stub.sampler = DistributionSampler(device=device, gamma=gamma)
    stub.geom_type = geom_type
    return stub


class TestGetT01:
    """Range and convergence tests for AbstractContrastiveDataset._get_t01"""

    def setup_method(self, method):
        self.seed = 42
        self.num_samples = 2_000
        self.atol = 0.05

    @pytest.mark.parametrize('geom_type', ['bounded', 'clamped'])
    def test_t0_range(self, geom_type, device_fx):
        """t0 must be in [0, lengths-2] for every trajectory"""
        torch.manual_seed(self.seed)
        ds = _make_stub(geom_type, device_fx)
        lengths = torch.tensor([5, 10, 3, 7], device=device_fx)
        for _ in range(50):
            t0, _ = ds._get_t01(lengths)
            assert (t0 >= 0).all()
            assert (t0 <= lengths - 2).all()

    @pytest.mark.parametrize('geom_type', ['bounded', 'clamped'])
    def test_t1_range(self, geom_type, device_fx):
        """t1 must be in [t0+1, lengths-1] for every trajectory"""
        torch.manual_seed(self.seed)
        ds = _make_stub(geom_type, device_fx)
        lengths = torch.tensor([5, 10, 3, 7], device=device_fx)
        for _ in range(50):
            t0, t1 = ds._get_t01(lengths)
            assert (t1 >= t0 + 1).all()
            assert (t1 <= lengths - 1).all()

    @pytest.mark.parametrize('geom_type', ['bounded', 'clamped'])
    def test_ordering(self, geom_type, device_fx):
        """t1 > t0 must hold unconditionally"""
        torch.manual_seed(self.seed)
        ds = _make_stub(geom_type, device_fx)
        lengths = torch.full((32,), 4, device=device_fx)
        for _ in range(100):
            t0, t1 = ds._get_t01(lengths)
            assert (t1 > t0).all()

    def test_unknown_geom_type_raises(self, device_fx):
        ds = _make_stub('invalid', device_fx)
        lengths = torch.tensor([5], device=device_fx)
        with pytest.raises(ValueError):
            ds._get_t01(lengths)

    @pytest.mark.parametrize('geom_type', ['bounded', 'clamped'])
    def test_t0_uniform(self, geom_type, device_fx):
        """t0 should be uniformly distributed over [0, lengths-2]"""
        torch.manual_seed(self.seed)
        ds = _make_stub(geom_type, device_fx)
        length = 5
        lengths = torch.full((self.num_samples,), length, device=device_fx)
        t0, _ = ds._get_t01(lengths)

        n_vals = length - 1  # t0 ∈ {0, ..., length-2}
        expected = torch.full((n_vals,), 1.0 / n_vals, device=device_fx)
        freq = torch.zeros(n_vals, device=device_fx)
        for k in range(n_vals):
            freq[k] = (t0 == k).float().mean()
        assert_close(freq, expected, atol=self.atol, rtol=0.0)

    @pytest.mark.parametrize('gamma', [0.5, 0.9])
    @pytest.mark.parametrize('geom_type', ['bounded', 'clamped'])
    def test_t1_marginal(self, gamma, geom_type, device_fx):
        """t1 marginal should match the analytically-computed mixture over t0"""
        torch.manual_seed(self.seed)
        ds = _make_stub(geom_type, device_fx, gamma=gamma)
        length = 5  # t1 ∈ {1, ..., 4}
        lengths = torch.full((self.num_samples,), length, device=device_fx)
        _, t1 = ds._get_t01(lengths)

        expected = _marginal_t1_expected(length, gamma, geom_type, device_fx)
        freq = torch.zeros(length - 1, device=device_fx)
        for t in range(1, length):
            freq[t - 1] = (t1 == t).float().mean()
        assert_close(freq, expected, atol=self.atol, rtol=0.0)
