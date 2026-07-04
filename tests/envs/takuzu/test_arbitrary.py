import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.takuzu.arbitrary import ArbitraryGoalTakuzu


class TestArbitraryGoalTakuzu:
    """Tests for ArbitraryGoalTakuzu"""

    def setup_method(self, method):
        self.seed = 42
        self.atol = 0.01
        fix_seed(self.seed)

    @pytest.mark.parametrize('n', [8])
    @pytest.mark.parametrize('m_boards', [2])
    @pytest.mark.parametrize('samples', [10_000])
    def test_goal_randomness(self, n, m_boards, samples, device_fx):
        env = ArbitraryGoalTakuzu(n=n, device=device_fx)
        boards = torch.randint(0, 3, (m_boards, n, n), dtype=torch.uint8, device=device_fx)

        expected_distrib = 0.5 * torch.ones(
            (m_boards, 2, n, n), dtype=torch.float, device=device_fx
        )
        expected_distrib[:, 0, :, :] += (0.5) * (boards == 1).float() + (-0.5) * (
            boards == 2
        ).float()
        expected_distrib[:, 1, :, :] += (-0.5) * (boards == 1).float() + (0.5) * (
            boards == 2
        ).float()

        goals = env.generate_goals(boards.repeat_interleave(samples, dim=0))
        goals_view = goals.view(m_boards, samples, n, n)

        distrib_1 = (goals_view == 1).float().mean(dim=1)
        distrib_2 = (goals_view == 2).float().mean(dim=1)
        distrib = torch.stack([distrib_1, distrib_2], dim=1)

        assert_close(distrib, expected_distrib, atol=0.1, rtol=0.0)
