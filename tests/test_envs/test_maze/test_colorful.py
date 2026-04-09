import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.maze.colorful import VectorizedColorfulMaze


class TestVectorizedColorfulMaze:
    def setup_method(self, method):
        self.wall_marker = 0
        self.floor_marker = 1
        self.pos_marker = 10
        self.seed = 42
        fix_seed(self.seed)

    def sample_targets(self, n: int, k: int, grid_dim: int, device: torch.device):
        """Samples k distinct positions for each board from a grid_dim x grid_dim grid."""
        n_cells = grid_dim * grid_dim
        assert k <= n_cells

        flat_idx = torch.multinomial(
            torch.ones((n, n_cells), device=device, dtype=torch.float32),
            num_samples=k,
            replacement=False,
        )
        rows = flat_idx // grid_dim
        cols = flat_idx % grid_dim
        return torch.stack((rows, cols), dim=-1)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('k_targets', [3, 4])
    @pytest.mark.parametrize('grid_dim', [3, 4, 9])
    def test_generate_goals(self, n, k_targets, grid_dim, device_fx):
        """
        Tests if generate_goals changes (k-1) targets to solved,
        and the last one to the position marker
        """
        states = torch.randint(0, 2, (n, grid_dim, grid_dim), device=device_fx)
        targets_pos = self.sample_targets(
            n=n,
            k=k_targets + 1,
            grid_dim=grid_dim,
            device=device_fx,
        )
        targets = targets_pos[:, :-1]
        pos = targets_pos[:, -1]

        batch_idx = torch.arange(n, device=device_fx)
        for i in range(k_targets):
            states[batch_idx, targets[:, i, 0], targets[:, i, 1]] = i + 2

        expected_goals = states.clone()
        states[batch_idx, pos[:, 0], pos[:, 1]] = self.pos_marker
        expected_goals[batch_idx, pos[:, 0], pos[:, 1]] = self.floor_marker

        for i in range(k_targets - 1):
            expected_goals[batch_idx, targets[:, i, 0], targets[:, i, 1]] += self.pos_marker
        expected_goals[batch_idx, targets[:, -1, 0], targets[:, -1, 1]] = self.pos_marker

        env = VectorizedColorfulMaze(
            device=device_fx,
            grid_dim=grid_dim,
            k_targets=k_targets,
            pos_marker=self.pos_marker,
            floor_marker=self.floor_marker,
            wall_marker=self.wall_marker,
        )
        goals = env.generate_goals(states.clone())
        print('states', states)
        print('goals', goals)
        print('epxected_goals', expected_goals)
        assert_close(goals, expected_goals)

    @pytest.mark.parametrize(
        'k_targets, states, expected_covered',
        [
            # 1a) - agent is covering the floor (1)
            # 1b) - agent is covering the first target (2), which will be solved when uncovered (12)
            (2, [[[10, 1], [2, 3]], [[10, 1], [3, 1]]], [1, 12]),
            # 2a) - agent is covering the already solved first target (12)
            # 2b) - agent is covering the second target (3), which will remain unsolved (3)
            (3, [[[10, 1], [13, 4]], [[10, 1], [2, 4]]], [12, 3]),
        ],
    )
    def test_get_covered_pos(self, k_targets, states, expected_covered, device_fx):
        states = torch.tensor(states, device=device_fx)
        expected_covered = torch.tensor(expected_covered, device=device_fx)
        env = VectorizedColorfulMaze(
            device=device_fx, grid_dim=states.shape[-1], k_targets=k_targets
        )
        covered = env.get_covered_pos(states)
        assert_close(covered, expected_covered)

    @pytest.mark.parametrize(
        'k_targets, states, action, expected_solved',
        [
            (
                2,  # 2 targets marked 12 and 13
                [[[10, 3, 0], [12, 0, 1], [0, 1, 0]], [[1, 1, 0], [10, 3, 1], [2, 1, 0]]],
                1,  # action 1 right
                [True, False],
            ),
            (
                1,  # 1 target marked as 11
                [[[10, 1], [1, 0]]],
                2,  # action 2 down
                [False],
            ),
        ],
    )
    def test_is_solved_after_action(self, k_targets, states, action, expected_solved, device_fx):
        states = torch.tensor(states, device=device_fx)
        expected_solved = torch.tensor(expected_solved, device=device_fx, dtype=torch.bool)

        env = VectorizedColorfulMaze(
            device=device_fx,
            grid_dim=states.shape[-1],
            k_targets=k_targets,
            pos_marker=self.pos_marker,
            floor_marker=self.floor_marker,
            wall_marker=self.wall_marker,
        )

        goals = env.generate_goals(states)
        new_states, legality = env.preview_action(states, action_no=action)
        solved = env.is_solved(new_states, goals)
        assert_close(solved, expected_solved)
