import pytest
import torch
from torch.testing import assert_close

from src.envs.maze import AbstractVectorizedMaze


class AbstractVectorizedMazeTestingSubclass(AbstractVectorizedMaze):
    """Subclass for testing purposes - overrides abstract methods with non-abstract pass"""

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        return states

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)


class TestAbstractVectorizedMazePositions:
    """Tests AbstractVectorizedMaze's methods related to positions, and making moves"""

    def setup_method(self, method):
        self.pos_marker = 3
        self.target_marker = 2
        self.seed = 42

    @pytest.mark.parametrize(
        'states_data, action_no, expected_legal, expected_legal_states',
        [
            # action 0 (up): batch of 2, both legal
            #   board 0: agent (1,0)→(0,0) free   board 1: agent (1,1)→(0,1) free
            (
                [[[0, 0], [3, 0]], [[0, 0], [0, 3]]],
                0,
                [True, True],
                [[[3, 0], [0, 0]], [[0, 3], [0, 0]]],
            ),
            # Same us above, but with a target in the second board
            (
                [[[0, 0], [3, 0]], [[0, 0], [2, 3]]],
                0,
                [True, True],
                [[[3, 0], [0, 0]], [[0, 3], [2, 0]]],
            ),
            # action 1 (right): board 0 legal (0,0)→(0,1) free, board 1 illegal boundary clamped
            (
                [[[3, 0], [0, 1]], [[0, 3], [0, 0]]],
                1,
                [True, False],
                [[[0, 3], [0, 1]]],
            ),
            # action 2 (down): single board, boundary hit (1,0) clamped → illegal
            (
                [[[0, 1], [3, 0]]],
                2,
                [False],
                [],
            ),
            # action 3 (left): board 0 wall hit, agent (0,1) blocked by wall at (0,0)=1 → illegal,
            #                   board 1 legal (1,1)→(1,0) free (agent reaches the target)
            (
                [[[1, 3], [0, 0]], [[0, 0], [2, 3]]],
                3,
                [False, True],
                [[[0, 0], [3, 0]]],
            ),
            # action 0 (up): agent blocked by wall above, (1,1)→(0,1)=wall → illegal
            (
                [[[0, 1], [2, 3]]],
                0,
                [False],
                [],
            ),
            # action 1 (right): agent blocked by wall to the right, (1,0)→(1,1)=wall → illegal
            (
                [[[0, 0], [3, 1]]],
                1,
                [False],
                [],
            ),
        ],
    )
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_preview_action(
        self, states_data, action_no, expected_legal, expected_legal_states, device
    ):
        """Tests the preview action method on various board states, testing for blocks"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        grid_dim = len(states_data[0])
        env = AbstractVectorizedMazeTestingSubclass(
            device=device, grid_dim=grid_dim, pos_marker=self.pos_marker
        )

        states = torch.tensor(states_data, device=device)
        expected_legal_t = torch.tensor(expected_legal, device=device)
        expected_legal_states_t = (
            torch.tensor(expected_legal_states, device=device)
            if (expected_legal_states != [])
            else None
        )

        new_states, legal_move = env.preview_action(states.clone(), action_no)
        assert_close(legal_move, expected_legal_t)
        if expected_legal_t.any():
            assert_close(new_states[expected_legal_t], expected_legal_states_t)
