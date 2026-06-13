import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.maze.target import VectorizedTargetMaze

_FLR = 0
_WLL = 1
_POS = 2
_TGT = 3


class TestVectorizedTargetMaze:
    def setup_method(self, method):
        self.pos_marker = _POS
        self.target_marker = _TGT
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize('n', [3, 5, 80])
    @pytest.mark.parametrize('grid_dim', [3, 4, 9])
    def test_generate_goals(self, n, grid_dim, device_fx):
        """Tests if generate_goals provides a list of all non-wall positions on the border"""
        states = torch.randint(0, 2, (n, grid_dim, grid_dim), device=device_fx)
        target = torch.randint(0, grid_dim, (n, 2), device=device_fx)
        pos = torch.randint(0, grid_dim, (n, 2), device=device_fx)
        states[:, pos[:, 0], pos[:, 1]] = _FLR
        states[:, target[:, 0], target[:, 1]] = _FLR
        batch_idx = torch.arange(n, device=device_fx)

        expected_goals = states.clone()
        expected_goals[batch_idx, target[:, 0], target[:, 1]] = _POS

        states[batch_idx, target[:, 0], target[:, 1]] = _TGT
        states[batch_idx, pos[:, 0], pos[:, 1]] = _POS

        env = VectorizedTargetMaze(device=device_fx, grid_dim=grid_dim, pos_marker=self.pos_marker)
        goals = env.generate_goals(states.clone())
        assert_close(goals, expected_goals)

    @pytest.mark.parametrize('n', [3, 5, 80])
    @pytest.mark.parametrize('grid_dim', [3, 4, 9])
    def test_is_solved_at_generation(self, n, grid_dim, device_fx):
        states = torch.randint(0, 2, (n, grid_dim, grid_dim), device=device_fx)
        target = torch.randint(0, grid_dim, (n, 2), device=device_fx)
        pos = torch.randint(0, grid_dim, (n, 2), device=device_fx)
        batch_idx = torch.arange(n, device=device_fx)

        states[batch_idx, target[:, 0], target[:, 1]] = _TGT
        states[batch_idx, pos[:, 0], pos[:, 1]] = _POS

        is_solved = torch.all(pos == target, dim=1)
        env = VectorizedTargetMaze(device=device_fx, grid_dim=grid_dim, pos_marker=self.pos_marker)
        assert_close(is_solved, env.is_solved(states, env.generate_goals(states)))

    @pytest.mark.parametrize(
        'states, action, expected_solved',
        [
            (
                [
                    [[_POS, _FLR, _WLL], [_TGT, _WLL, _FLR], [_WLL, _FLR, _WLL]],
                    [[_FLR, _FLR, _WLL], [_TGT, _POS, _FLR], [_WLL, _FLR, _WLL]],
                ],
                2,
                [True, False],
            )
        ],
    )
    def test_is_solved_after_action(self, states, action, expected_solved, device_fx):
        states = torch.tensor(states, device=device_fx)
        expected_solved = torch.tensor(expected_solved, device=device_fx, dtype=torch.bool)

        env = VectorizedTargetMaze(
            device=device_fx,
            grid_dim=states.shape[-1],
            target_marker=self.target_marker,
            pos_marker=self.pos_marker,
        )

        goals = env.generate_goals(states)
        new_states, legality = env.preview_action(states, action_no=action)
        solved = env.is_solved(new_states, goals)
        assert_close(solved, expected_solved)

    @pytest.mark.parametrize(
        'states, action, expected_states',
        [
            (
                # agent sits on the target cell (0,0), so target is absent;
                # action 1 (right) vacates (0,0) -> target restored, agent now at (0,1)
                [[[_POS, _FLR], [_FLR, _FLR]]],
                1,
                [[[_TGT, _POS], [_FLR, _FLR]]],
            )
        ],
    )
    def test_preview_action_restores_target(self, states, action, expected_states, device_fx):
        """When the agent leaves a covered target, the target marker reappears."""
        states = torch.tensor(states, device=device_fx)
        expected_states = torch.tensor(expected_states, device=device_fx)

        env = VectorizedTargetMaze(
            device=device_fx,
            grid_dim=states.shape[-1],
            target_marker=_TGT,
            pos_marker=_POS,
        )
        new_states, legality = env.preview_action(states.clone(), action_no=action)
        assert legality.all()
        assert_close(new_states, expected_states)
