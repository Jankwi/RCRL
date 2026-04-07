import pytest
import torch
from torch.testing import assert_close

from src.envs.maze.utils import generate_border_positions, get_pos_indices


class TestGetPosIndices:
    """Tests AbstractVectorizedMaze's methods related to positions, and making moves"""

    def setup_method(self, method):
        self.pos_marker = 3
        self.seed = 42

    @pytest.mark.parametrize(
        'wall_layout, agent_positions',
        [
            # two 3x3 states
            (
                [[[0, 0, 1], [0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0], [0, 1, 0]]],
                [(1, 1), (2, 2)],
            ),
            # three 2x2 states
            ([[[0, 1], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]], [(0, 0), (1, 0), (1, 0)]),
            # two 3x3 states where the pos_marker is not the highest value
            (
                [[[0, 0, 7], [0, 0, 0], [9, 0, 1]], [[1, 0, 0], [0, 0, 0], [0, 1, 0]]],
                [(1, 1), (2, 2)],
            ),
        ],
    )
    def test_get_pos_indices(self, wall_layout, agent_positions, device_fx):
        """Tests the get_pos_indices method on sample board states"""
        grid_dim = len(wall_layout[0])
        states = torch.tensor(wall_layout, device=device_fx)
        for i, (r, c) in enumerate(agent_positions):
            states[i, r, c] = self.pos_marker

        expected = torch.tensor(agent_positions, device=device_fx)
        result = get_pos_indices(states, grid_dim, self.pos_marker)
        assert_close(result, expected)

    def test_generate_border_positions(self, device_fx):
        """Tests the generate_border_positions method on sample board states"""
        expected = torch.tensor(
            [
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
            device=device_fx,
        ).to(torch.bool)
        assert_close(generate_border_positions(border=3, grid_dim=5, device=device_fx), expected)
