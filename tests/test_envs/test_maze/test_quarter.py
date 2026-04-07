from random import shuffle

import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.maze.quarter import VectorizedQuarterMaze
from src.envs.maze.state_sets import MazeVectorizedStateSet


class TestVectorizedQuarterMaze:
    def setup_method(self, method):
        self.pos_marker = 3
        self.seed = 42
        fix_seed(self.seed)

    def _get_true_false_perm(self, num_false: int, num_true: int, device: torch.device):
        bool_list = num_false * [False] + num_true * [True]
        shuffle(bool_list)
        return torch.tensor(bool_list, dtype=torch.bool, device=device)

    def _get_pos_list(self, grid_dim: int, mode: str):
        """Generates the list of allowed positions for the agent"""
        quarter_dim = int(grid_dim / 2)
        if mode == 'border':
            return [
                [i, j]
                for i in range(quarter_dim)
                for j in range(quarter_dim)
                if (
                    (i < quarter_dim and j < quarter_dim)
                    and (i == (quarter_dim - 1) or (j == quarter_dim - 1))
                )
            ]
        elif mode == 'unsolved':
            return [
                [i, j]
                for i in range(grid_dim)
                for j in range(grid_dim)
                if not (i < quarter_dim and j < quarter_dim)
            ]

    def _get_n_states(
        self,
        n: int,
        solved: bool,
        grid_dim: int,
        device: torch.device,
        clear_border: bool = True,
    ):
        """Generates n solved/unsolved board states"""
        n_idx = torch.arange(n, device=device)
        allowed_pos = self._get_pos_list(grid_dim, 'border' if solved else 'unsolved')
        pos = torch.tensor(allowed_pos, device=device)[
            torch.randint(high=len(allowed_pos), size=(n,), device=device)
        ]

        states = torch.randint(0, 2, (n, grid_dim, grid_dim), device=device)
        if clear_border:
            # draw a random idx from the border, and set the board to 0 there
            border_positions = self._get_pos_list(grid_dim, 'border')
            clear_pos = torch.tensor(border_positions, device=device)[
                torch.randint(high=len(border_positions), size=(n,), device=device)
            ]
            states[n_idx, clear_pos[:, 0], clear_pos[:, 1]] = 0
        states[n_idx, pos[:, 0], pos[:, 1]] = self.pos_marker
        return states

    def build_sets(self, many_states):
        """Builds a maze vectorized state set adding many_states[i] to the i-th one"""
        n, k, grid_dim = many_states.shape[0], many_states.shape[1], many_states.shape[2]
        sets = MazeVectorizedStateSet(
            n, grid_dim=grid_dim, device=many_states.device, pos_marker=self.pos_marker
        )
        idx = torch.arange(n, device=many_states.device)

        for i in range(k):
            sets.add_to_sets(many_states[:, i], idx=idx)
        return sets

    @pytest.mark.parametrize('n', [2, 3, 5])
    @pytest.mark.parametrize('grid_dim', [4, 8, 20])
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_generate_goals(self, n, grid_dim, device):
        """Tests if generate_goals provides a list of all non-wall positions on the border"""
        device = torch.device(device)
        states = self._get_n_states(n=n, solved=False, grid_dim=grid_dim, device=device)
        env = VectorizedQuarterMaze(device=device, grid_dim=grid_dim, pos_marker=self.pos_marker)
        goals = env.generate_goals(states.clone())

        border_pos = self._get_pos_list(grid_dim, 'border')
        idx = torch.arange(n, device=device)
        # Exploiting that MazeVectorizedStateSet tracks positions on a grid:
        # checking if a border position is "missing" from the set tells us
        # whether any goal placed the agent there (i.e. whether a wall blocked it).
        sets = self.build_sets(goals)
        for pos in border_pos:
            mask = states[:, pos[0], pos[1]] != 0
            candidates = torch.zeros(n, grid_dim, grid_dim, device=device)
            candidates[:, pos[0], pos[1]] = self.pos_marker
            missing = sets.missing_from_sets(candidates, idx)
            assert_close(missing, mask)

    @pytest.mark.parametrize(
        'grid_dim, n_unsolved, n_solved', [[4, 10, 0], [4, 5, 5], [4, 0, 10], [20, 3, 5]]
    )
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_is_solved(self, grid_dim, n_unsolved, n_solved, device):
        n = n_unsolved + n_solved
        device = torch.device(device)

        solved = self._get_n_states(n=n, solved=True, grid_dim=grid_dim, device=device)
        unsolved = self._get_n_states(n=n, solved=False, grid_dim=grid_dim, device=device)

        is_solved = self._get_true_false_perm(n_unsolved, n_solved, device=device)
        is_solved_bool = is_solved.reshape(-1, 1, 1)  # Reshape for torch.where broadcasting
        states = torch.where(is_solved_bool, solved, unsolved)
        env = VectorizedQuarterMaze(device=device, grid_dim=grid_dim, pos_marker=self.pos_marker)
        assert_close(is_solved, env.is_solved(states, None))
