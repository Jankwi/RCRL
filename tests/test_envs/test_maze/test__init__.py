import pytest
import torch
from torch.testing import assert_close

from src.envs.maze import AbstractVectorizedMaze
from src.envs.maze.utils import get_pos_indices


class AbstractVectorizedMazeTestingSubclass(AbstractVectorizedMaze):
    """Subclass for testing purposes - overrides abstract methods with non-abstract pass"""

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        return states

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)

    def get_covered_pos(self, states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape[0], device=states.device, dtype=states.dtype)


class SevensTestMaze(AbstractVectorizedMaze):
    """
    AbstractVectorizedMaze subclass, where all free positions are filled with 7.
    The intention of this class is to test how the AbstractVectorizedMaze operates
    with a non standard get_covered_pos
    """

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        return None

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return None

    def get_covered_pos(self, states: torch.Tensor) -> torch.Tensor:
        return 7 * torch.ones(states.shape[0], device=states.device, dtype=states.dtype)


class TestAbstractVectorizedMazePositions:
    """Tests AbstractVectorizedMaze's methods related to positions, and making moves"""

    def setup_method(self, method):
        self.pos_marker = 3
        self.target_marker = 2
        self.seed = 42

    @pytest.mark.parametrize(
        'states_data, action_no, expected_legal, expected_legal_states, env_class',
        [
            # action 0 (up): batch of 2, both legal
            #   board 0: agent (1,0)→(0,0) free   board 1: agent (1,1)→(0,1) free
            (
                [[[0, 0], [3, 0]], [[0, 0], [0, 3]]],
                0,
                [True, True],
                [[[3, 0], [0, 0]], [[0, 3], [0, 0]]],
                AbstractVectorizedMazeTestingSubclass,
            ),
            # Same us above, but with a target in the second board
            (
                [[[0, 0], [3, 0]], [[0, 0], [2, 3]]],
                0,
                [True, True],
                [[[3, 0], [0, 0]], [[0, 3], [2, 0]]],
                AbstractVectorizedMazeTestingSubclass,
            ),
            # action 1 (right): board 0 legal (0,0)→(0,1) free, board 1 illegal boundary clamped
            (
                [[[3, 0], [0, 1]], [[0, 3], [0, 0]]],
                1,
                [True, False],
                [[[0, 3], [0, 1]]],
                AbstractVectorizedMazeTestingSubclass,
            ),
            # action 2 (down): single board, boundary hit (1,0) clamped → illegal
            ([[[0, 1], [3, 0]]], 2, [False], [], AbstractVectorizedMazeTestingSubclass),
            # action 1 (right): agent blocked by wall to the right, (1,0)→(1,1)=wall → illegal
            ([[[0, 0], [3, 1]]], 1, [False], [], AbstractVectorizedMazeTestingSubclass),
            # Left (illegal, legal), but with sevens
            (
                [[[1, 3], [7, 7]], [[7, 7], [7, 3]]],
                3,
                [False, True],
                [[[7, 7], [3, 7]]],
                SevensTestMaze,
            ),
        ],
    )
    def test_preview_action(
        self, states_data, action_no, expected_legal, expected_legal_states, env_class, device_fx
    ):
        """Tests the preview action method on various board states, testing for blocks"""
        grid_dim = len(states_data[0])
        env = env_class(device=device_fx, grid_dim=grid_dim, pos_marker=self.pos_marker)

        states = torch.tensor(states_data, device=device_fx)
        expected_legal_t = torch.tensor(expected_legal, device=device_fx)
        expected_legal_states_t = (
            torch.tensor(expected_legal_states, device=device_fx)
            if (expected_legal_states != [])
            else None
        )

        new_states, legal_move = env.preview_action(states.clone(), action_no)
        assert_close(legal_move, expected_legal_t)
        if expected_legal_t.any():
            assert_close(new_states[expected_legal_t], expected_legal_states_t)

    @pytest.mark.parametrize(
        'states_data, action_no, expected_legal',
        [
            # 3x3 all-sevens, agent at centre (1,1), move right → legal
            ([[[7, 7, 7], [7, 3, 7], [7, 7, 7]]], 1, [True]),
            # 3x3 all-sevens, agent at left edge (1,0), move left → boundary clamp → illegal
            ([[[7, 7, 7], [3, 7, 7], [7, 7, 7]]], 3, [False]),
        ],
    )
    def test_preview_action_on_covered_positions(
        self, states_data, action_no, expected_legal, device_fx
    ):
        """Tests the preview_action method on the SevensTestMaze env"""
        grid_dim = len(states_data[0])
        env = SevensTestMaze(device=device_fx, grid_dim=grid_dim, pos_marker=self.pos_marker)
        states = torch.tensor(states_data, device=device_fx)
        old_positions = get_pos_indices(states, grid_dim, self.pos_marker)

        new_states, legal_move = env.preview_action(states.clone(), action_no)

        assert_close(legal_move, torch.tensor(expected_legal, device=device_fx))
        if legal_move.any():
            legal_old_pos = old_positions[legal_move]
            legal_new_pos = torch.clamp(
                legal_old_pos + env.action_tensor[action_no],
                min=0,
                max=grid_dim - 1,
            )
            expected_legal_new = states[legal_move].clone()
            b = torch.arange(legal_move.sum(), device=device_fx)
            expected_legal_new[b, legal_old_pos[:, 0], legal_old_pos[:, 1]] = 7
            expected_legal_new[b, legal_new_pos[:, 0], legal_new_pos[:, 1]] = self.pos_marker
            assert_close(new_states[legal_move], expected_legal_new)
