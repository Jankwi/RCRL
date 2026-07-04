import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.maze.arbitrary import ArbitraryPositionMaze

_FLR = 0
_WLL = 1
_POS = 2
_TGT = 3


class TestArbitraryPositionMaze:
    def setup_method(self, method):
        self.pos_marker = _POS
        self.wall_marker = _WLL
        self.floor_marker = _FLR
        self.seed = 42
        fix_seed(self.seed)
        self.atol = 0.02
        self.rtol = 0.0

    @pytest.mark.parametrize('n_boards, reps', [(1, 8_000), (2, 8_000)])
    @pytest.mark.parametrize('target_marker', [_TGT, None])
    @pytest.mark.parametrize('grid_dim', [4])
    def test_generate_goals(self, n_boards, reps, target_marker, grid_dim, device_fx):
        """
        Tests that all floor-marked positions have equal probability of being sampled,
        as the arbitrary position goals.
        """
        states = torch.randint(0, 2, (n_boards, grid_dim, grid_dim), device=device_fx)

        free = torch.randint(0, grid_dim, (n_boards, 2), device=device_fx)
        pos = torch.randint(0, grid_dim, (n_boards, 2), device=device_fx)
        states[:, free[:, 0], free[:, 1]] = self.floor_marker
        states[:, pos[:, 0], pos[:, 1]] = self.pos_marker
        if target_marker is not None:
            target = torch.randint(0, grid_dim, (n_boards, 2), device=device_fx)
            states[:, target[:, 0], target[:, 1]] = target_marker

        floor_mask = (states != self.wall_marker).float()
        expected_distrib = floor_mask / floor_mask.sum(dim=(-1, -2), keepdim=True)

        env = ArbitraryPositionMaze(
            device=device_fx,
            grid_dim=grid_dim,
            floor_marker=self.floor_marker,
            wall_marker=self.wall_marker,
            pos_marker=self.pos_marker,
            target_marker=target_marker,
        )
        tiled = states.repeat_interleave(reps, dim=0)  # (n_boards * reps, H, W)
        goals = env.generate_goals(tiled)
        distrib = (
            (goals == self.pos_marker)
            .float()
            .reshape(n_boards, reps, grid_dim, grid_dim)
            .mean(dim=1)
        )
        assert_close(distrib, expected_distrib, atol=self.atol, rtol=self.rtol)

    @pytest.mark.parametrize(
        'states, goals, expected_solved',
        [
            (
                # board 0: state matches goal exactly -> solved
                # board 1: agent in a different cell than the goal -> unsolved
                [[[0, 0], [2, 0]], [[0, 0], [2, 0]]],
                [[[0, 0], [2, 0]], [[2, 0], [0, 0]]],
                [True, False],
            )
        ],
    )
    def test_is_solved(self, states, goals, expected_solved, device_fx):
        """is_solved is exact board equality between state and goal."""
        states = torch.tensor(states, device=device_fx)
        goals = torch.tensor(goals, device=device_fx)
        expected_solved = torch.tensor(expected_solved, device=device_fx, dtype=torch.bool)

        env = ArbitraryPositionMaze(
            device=device_fx,
            grid_dim=states.shape[-1],
            floor_marker=self.floor_marker,
            wall_marker=self.wall_marker,
            pos_marker=self.pos_marker,
            target_marker=_TGT,
        )
        assert_close(env.is_solved(states, goals), expected_solved)

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

        env = ArbitraryPositionMaze(
            device=device_fx,
            grid_dim=states.shape[-1],
            floor_marker=self.floor_marker,
            wall_marker=self.wall_marker,
            target_marker=_TGT,
            pos_marker=self.pos_marker,
        )
        new_states, legality = env.preview_action(states.clone(), action_no=action)
        assert legality.all()
        assert_close(new_states, expected_states)
