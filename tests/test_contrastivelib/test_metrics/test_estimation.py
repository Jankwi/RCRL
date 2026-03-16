import math

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.utils import fix_seed
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper


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

        ve = SingleGoalValueEstimator(FlattenModule().to(device), self.sim_fn, device=device)
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

        ve = ManyGoalsValueEstimator(FlattenModule().to(device), self.sim_fn, device=device)
        assert_close(ve.get_solved_distance(states, goals), expected)


def _make_spec_ve(ve_class, state_shape, embed_dim, special_val, sim_fn, device):
    """Helper: builds a ValueEstimator with SpecialStateModelWrapper."""
    spec_mgr = SpecialStateManager(val=special_val, shape=state_shape, device=device)
    flat_size = math.prod(state_shape)
    base_model = nn.Sequential(nn.Flatten(), nn.Linear(flat_size, embed_dim)).to(device)
    model = SpecialStateModelWrapper(base_model, spec_mgr, embed_dim).to(device)
    return ve_class(model, sim_fn, device=device)


class TestSingleGoalValueEstimatorWithSpecMgr:
    """Tests for SingleGoalValueEstimator with SpecialStateModelWrapper."""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)
        self.sim_fn = get_similarity_measure('manhattan', False)
        self.special_val = -1
        self.embed_dim = 16

    @pytest.mark.parametrize('state_shape', [(3,), (8, 8), (2, 2, 2)])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_with_special_goals(self, n, state_shape, device):
        """All goals are the special state (as from spec_mgr.create_many)."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        ve = _make_spec_ve(
            SingleGoalValueEstimator,
            state_shape,
            self.embed_dim,
            self.special_val,
            self.sim_fn,
            device,
        )
        states = torch.randint(0, 10, (n, *state_shape), device=device).float()
        goals = torch.full((n, *state_shape), self.special_val, device=device).float()

        result = ve.get_solved_distance(states, goals)
        assert result.shape == (n,)

    @pytest.mark.parametrize('state_shape', [(3,), (8, 8)])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_with_mixed_special_and_normal(self, n, state_shape, device):
        """Some states are special, some are normal."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        ve = _make_spec_ve(
            SingleGoalValueEstimator,
            state_shape,
            self.embed_dim,
            self.special_val,
            self.sim_fn,
            device,
        )
        states = torch.randint(0, 10, (n, *state_shape), device=device).float()
        goals = torch.randint(0, 10, (n, *state_shape), device=device).float()
        if n > 1:
            states[0] = self.special_val  # first state is special

        result = ve.get_solved_distance(states, goals)
        assert result.shape == (n,)


class TestManyGoalsValueEstimatorWithSpecMgr:
    """Tests for ManyGoalsValueEstimator with SpecialStateModelWrapper.

    Reproduces the bug where goals without a k dimension (as returned by
    spec_mgr.create_many in eval_no_search) cause ManyGoalsValueEstimator
    to misinterpret state spatial dims as k and corrupt the reshape.
    """

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)
        self.sim_fn = get_similarity_measure('manhattan', False)
        self.special_val = -1
        self.embed_dim = 16

    @pytest.mark.parametrize('state_shape', [(3,), (8, 8), (2, 2, 2)])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_single_goal_per_state(self, n, state_shape, device):
        """Goals without k dim (as from spec_mgr.create_many) should not crash."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        ve = _make_spec_ve(
            ManyGoalsValueEstimator,
            state_shape,
            self.embed_dim,
            self.special_val,
            self.sim_fn,
            device,
        )
        states = torch.randint(0, 10, (n, *state_shape), device=device).float()
        goals = torch.full((n, *state_shape), self.special_val, device=device).float()

        result = ve.get_solved_distance(states, goals)
        assert result.shape == (n,)

    @pytest.mark.parametrize('state_shape', [(3,), (8, 8)])
    @pytest.mark.parametrize('k', [1, 4])
    @pytest.mark.parametrize('n', [1, 3])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_many_goals_with_special_states(self, n, k, state_shape, device):
        """Goals with k dim where some goals are special should work correctly."""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        ve = _make_spec_ve(
            ManyGoalsValueEstimator,
            state_shape,
            self.embed_dim,
            self.special_val,
            self.sim_fn,
            device,
        )
        states = torch.randint(0, 10, (n, *state_shape), device=device).float()
        goals = torch.randint(0, 10, (n, k, *state_shape), device=device).float()
        goals[:, 0] = self.special_val  # first goal per state is special

        result = ve.get_solved_distance(states, goals)
        assert result.shape == (n,)
