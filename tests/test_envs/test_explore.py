import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.utils import fix_seed
from src.envs.explore import eval_no_search
from src.envs.storage import DeepStateStorage
from tests.test_envs.mock import (
    MockValueEstimator,
    MockVectorizedEnvironment,
)


def _solvable_states(n: int, max_idx: int, device: torch.device) -> torch.Tensor:
    """Sample states the greedy solves: goal > pos with goal % 3 != 2."""
    if n == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    pos = torch.randint(0, max_idx - 9, (n,), device=device)
    goal = pos + torch.randint(1, 11, (n,), device=device)
    # Bump rem-2 goals down by 1 (to rem 1); still >= pos since offset >= 1
    goal = torch.where(goal % 3 == 2, goal - 1, goal)
    return torch.stack([goal, pos], dim=1)


def _unsolvable_states(n: int, max_idx: int, device: torch.device) -> torch.Tensor:
    """Sample unsolvable states: goal < pos, or goal > pos with goal % 3 == 2."""
    if n == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    # Mode 1: goal behind pos
    behind_pos = torch.randint(1, max_idx + 1, (n,), device=device)
    behind_goal = torch.randint(0, max_idx, (n,), device=device) % behind_pos
    # Mode 2: goal ahead but on a rem-2 position (skipped by greedy)
    # Sample goal from {2, 5, 8, 11, 14, 17} (interior rem-2 positions; exclude
    # max_idx=20 which is reachable via forced +1 at pos=19)
    two_goal = torch.randint(0, 6, (n,), device=device) * 3 + 2
    two_pos = torch.randint(0, max_idx - 9, (n,), device=device) % torch.clamp(two_goal, min=1)
    # Randomly pick which failure mode to use per element
    use_behind = torch.randint(0, 2, (n,), device=device).bool()
    pos = torch.where(use_behind, behind_pos, two_pos)
    goal = torch.where(use_behind, behind_goal, two_goal)
    return torch.stack([goal, pos], dim=1)


def _overflow_states(n: int, max_idx_high: int, device: torch.device) -> torch.Tensor:
    """States where goal is too far to reach within max_len steps."""
    if n == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    pos = torch.zeros(n, dtype=torch.long, device=device)
    goal = torch.randint(50, max_idx_high, (n,), device=device)
    return torch.stack([goal, pos], dim=1)


def _simulate_greedy(goal: int, pos: int, max_idx: int, max_len: int, mod_base: int = 3):
    """Simulate the greedy policy for a single game. Returns (trajectory, solved)."""
    traj = [(goal, pos)]
    visited = {pos}
    current = pos
    solved = current == goal

    while not solved and len(traj) < max_len:
        candidates = []
        for offset in [1, 2]:
            new_pos = current + offset
            if new_pos <= max_idx and new_pos not in visited:
                candidates.append((new_pos % mod_base, new_pos))

        if not candidates:
            break

        candidates.sort()
        best = candidates[0][1]
        visited.add(best)
        current = best
        traj.append((goal, current))
        solved = current == goal

    return traj, solved


class TestEvalNoSearchShallow:
    """
    Tests for eval_no_search's solved rate on the mock env.

    Actions are +1 and +2. The MockValueEstimator prefers lower (pos % 3),
    so the greedy policy is:
        rem 0 -> +1 (to rem 1)
        rem 1 -> +2 (to rem 0)
        rem 2 -> +1 (to rem 0)
    Interior positions with rem 2 are never visited (the boundary max_idx
    can be forced when +2 is illegal), so (goal, pos) is solvable iff
    goal == pos, or goal > pos and (goal % 3 != 2 or goal == max_idx).
    """

    def setup_method(
        self,
    ):
        self.max_idx = 20
        self.seed = 42
        fix_seed(self.seed)

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
        device = torch.device(device)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device)
        ve = MockValueEstimator(device=device)

        total = num_unsolved + num_solved
        states = torch.cat(
            [
                _solvable_states(num_solved, self.max_idx, device),
                _unsolvable_states(num_unsolved, self.max_idx, device),
            ],
            dim=0,
        )

        result = eval_no_search(states, env, ve)
        assert result['solved_rate'] == pytest.approx(num_solved / total)


class TestEvalNoSearchDeep:
    """
    Tests for eval_no_search with DeepStateStorage.

    Verifies that trajectories, lengths, and outcomes match
    the manually-simulated greedy policy.
    """

    def setup_method(
        self,
    ):
        self.max_idx = 20
        self.seed = 42
        fix_seed(self.seed)

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
    def test_all_fit(self, num_unsolved, num_solved, device):
        max_len = 32
        device = torch.device(device)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device)
        ve = MockValueEstimator(device=device)

        total = num_unsolved + num_solved
        states = torch.cat(
            [
                _solvable_states(num_solved, self.max_idx, device),
                _unsolvable_states(num_unsolved, self.max_idx, device),
            ],
            dim=0,
        )

        tensors, metrics = eval_no_search(
            states, env, ve, storage_class=DeepStateStorage, max_len={'max_len': max_len}
        )

        # Build expected tensors by simulating the greedy policy per game
        expected_traj = torch.zeros((total, max_len, 2), device=device, dtype=states.dtype)
        expected_lengths = torch.zeros((total,), device=device, dtype=torch.long)
        expected_outcome = torch.zeros((total,), device=device, dtype=torch.bool)

        for i in range(total):
            g, p = states[i, 0].item(), states[i, 1].item()
            traj, solved = _simulate_greedy(g, p, self.max_idx, max_len)
            for j, (tg, tp) in enumerate(traj):
                expected_traj[i, j] = torch.tensor([tg, tp], device=device)
            expected_lengths[i] = len(traj)
            expected_outcome[i] = solved

        assert metrics['solved_rate'] == pytest.approx(num_solved / total)
        torch.testing.assert_close(tensors['trajectories'], expected_traj)
        torch.testing.assert_close(tensors['lengths'], expected_lengths)
        assert_close(tensors['outcome'], expected_outcome)

    @pytest.mark.parametrize(
        'num_unsolved, num_solved, num_overflow',
        [(0, 0, 4), (2, 2, 2), (0, 4, 2), (4, 0, 2)],
    )
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
    def test_some_overflow(self, num_unsolved, num_solved, num_overflow, device):
        max_len = 32
        max_idx_high = 100
        device = torch.device(device)
        env = MockVectorizedEnvironment(max_idx=max_idx_high, device=device)
        ve = MockValueEstimator(device=device)

        total = num_unsolved + num_solved + num_overflow
        states = torch.cat(
            [
                _solvable_states(num_solved, self.max_idx, device),
                _unsolvable_states(num_unsolved, self.max_idx, device),
                _overflow_states(num_overflow, max_idx_high, device),
            ],
            dim=0,
        )

        tensors, metrics = eval_no_search(
            states, env, ve, storage_class=DeepStateStorage, max_len={'max_len': max_len}
        )

        # Build expected tensors by simulating the greedy policy per game
        expected_traj = torch.zeros((total, max_len, 2), device=device, dtype=states.dtype)
        expected_lengths = torch.zeros((total,), device=device, dtype=torch.long)
        expected_outcome = torch.zeros((total,), device=device, dtype=torch.bool)

        for i in range(total):
            g, p = states[i, 0].item(), states[i, 1].item()
            traj, solved = _simulate_greedy(g, p, max_idx_high, max_len)
            for j, (tg, tp) in enumerate(traj):
                expected_traj[i, j] = torch.tensor([tg, tp], device=device)
            expected_lengths[i] = len(traj)
            expected_outcome[i] = solved

        assert metrics['solved_rate'] == pytest.approx(num_solved / total)
        torch.testing.assert_close(tensors['trajectories'], expected_traj)
        torch.testing.assert_close(tensors['lengths'], expected_lengths)
        assert (tensors['outcome'] == expected_outcome).all()

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
    def test_eval_no_search_one_unsolvable(self, device):
        max_len = 32
        device = torch.device(device)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device)
        ve = MockValueEstimator(device=device)

        states = torch.tensor([[0, self.max_idx - 5]], device=device)
        tensors, metrics = eval_no_search(
            states, env, ve, storage_class=DeepStateStorage, max_len={'max_len': max_len}
        )

        assert metrics['solved_rate'] == 0.0
        assert_close(tensors['outcome'], torch.tensor([False], device=device))
