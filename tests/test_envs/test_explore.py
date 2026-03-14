import pytest
import torch

from src.contrastivelib.utils import fix_seed
from src.envs.explore import eval_no_search
from tests.test_envs.mock import MockValueEstimator, MockVectorizedEnvironment


class TestEvalNoSearch:
    """
    Tests for eval_no_search's solved rate on the mock env.

    Actions are +1 and +2. The MockValueEstimator prefers lower (pos % 3),
    so the greedy policy is:
        rem 0 -> +1 (to rem 1)
        rem 1 -> +2 (to rem 0)
        rem 2 -> +1 (to rem 0)
    Interior positions with rem 2 are never visited (the boundary MAX_IDX
    can be forced when +2 is illegal), so (goal, pos) is solvable iff
    goal == pos, or goal > pos and (goal % 3 != 2 or goal == MAX_IDX).
    """

    MAX_IDX = 20
    SEED = 42

    def _solvable_states(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample states the greedy solves: goal > pos with goal % 3 != 2."""
        if n == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        pos = torch.randint(0, self.MAX_IDX - 9, (n,), device=device)
        goal = pos + torch.randint(1, 11, (n,), device=device)
        # Bump rem-2 goals down by 1 (to rem 1); still >= pos since offset >= 1
        goal = torch.where(goal % 3 == 2, goal - 1, goal)
        return torch.stack([goal, pos], dim=1)

    def _unsolvable_states(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample unsolvable states: goal < pos, or goal > pos with goal % 3 == 2."""
        if n == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        # Mode 1: goal behind pos
        behind_pos = torch.randint(1, self.MAX_IDX + 1, (n,), device=device)
        behind_goal = torch.randint(0, self.MAX_IDX, (n,), device=device) % behind_pos
        # Mode 2: goal ahead but on a rem-2 position (skipped by greedy)
        # Sample goal from {2, 5, 8, 11, 14, 17} (interior rem-2 positions; exclude
        # MAX_IDX=20 which is reachable via forced +1 at pos=19)
        two_goal = torch.randint(0, 6, (n,), device=device) * 3 + 2
        two_pos = torch.randint(0, self.MAX_IDX - 9, (n,), device=device) % torch.clamp(
            two_goal, min=1
        )
        # Randomly pick which failure mode to use per element
        use_behind = torch.randint(0, 2, (n,), device=device).bool()
        pos = torch.where(use_behind, behind_pos, two_pos)
        goal = torch.where(use_behind, behind_goal, two_goal)
        return torch.stack([goal, pos], dim=1)

    @pytest.mark.parametrize('num_unsolved, num_solved', [(4, 0), (3, 1), (10, 10), (1, 3), (0, 4)])
    @pytest.mark.parametrize(
        'device',
        [
            'cpu',
            pytest.param(
                'cuda',
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason='CUDA not available'
                ),
            ),
        ],
    )
    def test_eval_no_search(self, num_unsolved, num_solved, device):
        fix_seed(self.SEED)
        device = torch.device(device)
        env = MockVectorizedEnvironment(max_idx=self.MAX_IDX, device=device)
        ve = MockValueEstimator(device=device)

        total = num_unsolved + num_solved
        states = torch.cat(
            [
                self._solvable_states(num_solved, device),
                self._unsolvable_states(num_unsolved, device),
            ],
            dim=0,
        )

        result = eval_no_search(states, env, ve)
        assert result['solved_rate'] == pytest.approx(num_solved / total)
