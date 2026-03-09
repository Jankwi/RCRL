import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.utils.distributions import DistributionSampler


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return t / t.sum()


class TestDistributionSampler:
    """Convergence tests for the DistributionSampler class"""
    def setup_method(self, method):
        self.num_samples = 2_000
        self.num_interleave_reps = 1_000
        self.seed = 42
        self.uniform_atol = 0.05
        self.geometric_atol = 0.025

    @pytest.mark.parametrize('gamma', [0.5, 0.9])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('inclusive_high_list', [[3, 2], [1, 1]])
    def test_bounded_geometric(self, gamma, device, inclusive_high_list):
        """Tests DistributionSampler's bounded_geometric method's convergence"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        torch.manual_seed(self.seed)
        sampler = DistributionSampler(gamma=gamma, device=device)

        inclusive_high = torch.tensor(inclusive_high_list, device=device)
        batch = inclusive_high.unsqueeze(0).expand(self.num_samples, -1)
        samples = sampler.bounded_geometric(batch)

        for idx, n in enumerate(inclusive_high_list):
            expected = _normalize(torch.tensor(
                [gamma**k for k in range(1, n + 1)], device=device,
            ))
            col = samples[:, idx]
            freq = torch.zeros(n, device=device)
            for j in range(1, n + 1):
                freq[j - 1] = (col == j).float().mean()
            print(f'{freq=} {expected=}')
            assert_close(freq, expected, atol=self.geometric_atol, rtol=0.0)

    @pytest.mark.parametrize('gamma', [None, 0.5, 0.9])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('n_list', [[3, 6, 3, 2], [2, 4]])
    def test_bounded_uniform(self, gamma, device, n_list):
        """Tests DistributionSampler's bounded_uniform method's convergence"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        torch.manual_seed(self.seed)
        sampler = DistributionSampler(gamma=gamma, device=device)

        exclusive_high = torch.tensor(n_list, device=device)
        batch = exclusive_high.unsqueeze(0).expand(self.num_samples, -1)
        samples = sampler.bounded_uniform(batch)

        for idx, n in enumerate(n_list):
            expected = torch.full((n,), 1.0 / n, device=device)
            col = samples[:, idx]
            freq = torch.zeros(n, device=device)
            for j in range(n):
                freq[j] = (col == j).float().mean()
            assert_close(freq, expected, atol=self.uniform_atol, rtol=0.0)

    @pytest.mark.parametrize('gamma', [0.5, 0.9])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('n_list', [[3, 1, 3, 2], [2, 4]])
    @pytest.mark.parametrize('inclusive_high_list', [[2, 1, 7], [5, 3]])
    def test_interleave(self, gamma, device, n_list, inclusive_high_list):
        """
            Tests the converegnce of DistributionSampler's bounded_geometric and bounded_uniform
            methods, while interlaving the calls
        """
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        torch.manual_seed(self.seed)
        sampler = DistributionSampler(gamma=gamma, device=device)

        exclusive_high = torch.tensor(n_list, device=device)
        inclusive_high = torch.tensor(inclusive_high_list, device=device)

        u_samples = []
        g_samples = []
        for _ in range(self.num_interleave_reps):
            u_samples.append(sampler.bounded_uniform(exclusive_high))
            g_samples.append(sampler.bounded_geometric(inclusive_high))
            u_samples.append(sampler.bounded_uniform(exclusive_high))
            g_samples.append(sampler.bounded_geometric(inclusive_high))

        u_all = torch.stack(u_samples)
        g_all = torch.stack(g_samples)

        # Check uniform frequencies
        for idx, n in enumerate(n_list):
            expected = torch.full((n,), 1.0 / n, device=device)
            col = u_all[:, idx]
            freq = torch.zeros(n, device=device)
            for j in range(n):
                freq[j] = (col == j).float().mean()
            print(f'{freq=} {expected=}')
            assert_close(freq, expected, atol=self.uniform_atol, rtol=0.0)

        # Check geometric frequencies
        for idx, n in enumerate(inclusive_high_list):
            expected = _normalize(torch.tensor(
                [gamma**k for k in range(1, n + 1)], device=device,
            ))
            col = g_all[:, idx]
            freq = torch.zeros(n, device=device)
            for j in range(1, n + 1):
                freq[j - 1] = (col == j).float().mean()
            assert_close(freq, expected, atol=self.geometric_atol, rtol=0.0)
