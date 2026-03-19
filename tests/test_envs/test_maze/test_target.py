import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.utils import fix_seed
from src.envs.maze.target import VectorizedTargetMaze


class TestVectorizedTargetMaze:
    def setup_method(self, method):
        self.pos_marker = 3
        self.target_marker = 2
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize('n', [3, 5, 80])
    @pytest.mark.parametrize('grid_dim', [3, 4, 9])
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_generate_goals(self, n, grid_dim, device):
        """Tests if generate_goals provides a list of all non-wall positions on the border"""
        device = torch.device(device)

        states = torch.randint(0, 2, (n, grid_dim, grid_dim), device=device)
        target = torch.randint(0, grid_dim, (n, 2), device=device)
        pos = torch.randint(0, grid_dim, (n, 2), device=device)
        states[:, pos[:, 0], pos[:, 1]] = 0
        states[:, target[:, 0], target[:, 1]] = 0
        batch_idx = torch.arange(n, device=device)

        expected_goals = states.clone()
        expected_goals[batch_idx, target[:, 0], target[:, 1]] = self.pos_marker

        states[batch_idx, target[:, 0], target[:, 1]] = self.target_marker
        states[batch_idx, pos[:, 0], pos[:, 1]] = self.pos_marker

        env = VectorizedTargetMaze(device=device, grid_dim=grid_dim, pos_marker=self.pos_marker)
        goals = env.generate_goals(states.clone())
        assert_close(goals, expected_goals)

    @pytest.mark.parametrize('n', [3, 5, 80])
    @pytest.mark.parametrize('grid_dim', [3, 4, 9])
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_is_solved_at_generation(self, n, grid_dim, device):
        device = torch.device(device)

        states = torch.randint(0, 2, (n, grid_dim, grid_dim), device=device)
        target = torch.randint(0, grid_dim, (n, 2), device=device)
        pos = torch.randint(0, grid_dim, (n, 2), device=device)
        batch_idx = torch.arange(n, device=device)

        states[batch_idx, target[:, 0], target[:, 1]] = self.target_marker
        states[batch_idx, pos[:, 0], pos[:, 1]] = self.pos_marker

        is_solved = torch.all(pos == target, dim=1)
        env = VectorizedTargetMaze(device=device, grid_dim=grid_dim, pos_marker=self.pos_marker)
        assert_close(is_solved, env.is_solved(states, env.generate_goals(states)))

    @pytest.mark.parametrize(
        'states, action, expected_solved',
        [
            (
                [[[3, 0, 1], [2, 1, 0], [1, 0, 1]], [[0, 0, 1], [2, 3, 0], [1, 0, 1]]],
                2,
                [True, False],
            )
        ],
    )
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_is_solved_after_action(self, states, action, expected_solved, device):
        device = torch.device(device)
        states = torch.tensor(states, device=device)
        expected_solved = torch.tensor(expected_solved, device=device, dtype=torch.bool)

        env = VectorizedTargetMaze(
            device=device,
            grid_dim=states.shape[-1],
            target_marker=self.target_marker,
            pos_marker=self.pos_marker,
        )

        goals = env.generate_goals(states)
        new_states, legality = env.preview_action(states, action_no=action)
        solved = env.is_solved(new_states, goals)
        print(f'{solved.shape=} {solved.dtype=} {solved=}')
        assert_close(solved, expected_solved)
