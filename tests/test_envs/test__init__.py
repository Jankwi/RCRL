import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from tests.test_envs.mock import MockVectorizedEnvironment


class TestVectorizedEnvironment:
    def setup_method(self, method):
        self.max_idx = 20
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize('n_states', [1, 5, 10])
    def test_preview_all_actions(self, n_states: int, device_fx):
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device_fx)
        states = torch.randint(0, self.max_idx, (n_states, 2), device=device_fx)

        expected_new_states = states.clone().unsqueeze(1).repeat(1, 2, 1)
        expected_new_states[:, 0, 1] += 1
        expected_new_states[:, 1, 1] += 2
        expected_legality = (expected_new_states[:, :, 1] <= self.max_idx).reshape(n_states, 2)

        new_states, legality = env.preview_all_actions(states)
        assert_close(new_states, expected_new_states)
        assert_close(legality, expected_legality)
