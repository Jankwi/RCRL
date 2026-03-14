import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.utils import fix_seed


class FlattenModule(nn.Module):
    """A simple module that flattens input tensors to float."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x.to(torch.float))


class TestSingleGoalValueEstimator:
    """Tests for SingleGoalValueEstimator.get_solved_distance."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)
        self.sim_fn = get_similarity_measure('manhattan', False)

    @pytest.mark.parametrize('state_shape', [(3,), (8, 8), (2, 2, 2)])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_get_solved_distance(self, n, state_shape, device):
        """Tests that get_solved_distance returns per-sample Manhattan distances."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        states = torch.randint(0, 10, (n, *state_shape), device=device)
        goals = torch.randint(0, 10, (n, *state_shape), device=device)

        expected = torch.sum(
            torch.abs(states.reshape(n, -1).float() - goals.reshape(n, -1).float()),
            dim=-1,
        )

        ve = SingleGoalValueEstimator(FlattenModule().to(device), self.sim_fn)
        assert_close(ve.get_solved_distance(states, goals), expected)


class TestManyGoalsValueEstimator:
    """Tests for ManyGoalsValueEstimator.get_solved_distance."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)
        self.sim_fn = get_similarity_measure('manhattan', False)

    @pytest.mark.parametrize('state_shape', [(3,), (8, 8), (2, 2, 2)])
    @pytest.mark.parametrize('k', [1, 4])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_get_solved_distance(self, n, k, state_shape, device):
        """Tests that get_solved_distance returns the min Manhattan dist over k goals per sample."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        states = torch.randint(0, 10, (n, *state_shape), device=device)
        goals = torch.randint(0, 10, (n, k, *state_shape), device=device)

        dists = torch.sum(
            torch.abs(states.reshape(n, -1).float().unsqueeze(1) - goals.reshape(n, k, -1).float()),
            dim=-1,
        )
        expected = torch.min(dists, dim=1).values

        ve = ManyGoalsValueEstimator(FlattenModule().to(device), self.sim_fn)
        assert_close(ve.get_solved_distance(states, goals), expected)
