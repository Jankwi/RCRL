import pytest
import torch
from torch.testing import assert_close

from src.envs.maze.augment import RandomDirectionMazeAugmentation

_FLR, _WLL, _POS, _TGT = 0, 1, 2, 3


def _build_traj(boards: torch.Tensor, pivot_idx: torch.Tensor, temporal_dist: torch.Tensor):
    """All time steps hold the same board so prev_pos is initialised to curr_pos."""
    T = int((pivot_idx + temporal_dist).max().item()) + 1
    traj = boards.unsqueeze(1).expand(-1, T, -1, -1).clone()
    lengths = pivot_idx + temporal_dist + 1
    return traj, lengths


def _run(boards, pivot_idx, temporal_dist):
    traj, lengths = _build_traj(boards, pivot_idx, temporal_dist)
    aug = RandomDirectionMazeAugmentation()
    return aug(traj, lengths, pivot_idx, temporal_dist)


class TestRandomDirectionMazeAugmentation:
    @pytest.mark.parametrize('m, n_steps', [(32, 3)])
    def test_exactly_one_agent(self, m, n_steps, device_fx):
        """Exactly one agent marker per board after any augmentation."""
        boards = torch.zeros(m, 5, 5, dtype=torch.long, device=device_fx)
        boards[:, 2, 2] = _POS
        pivot_idx = torch.zeros(m, dtype=torch.long, device=device_fx)
        temporal_dist = torch.full((m,), n_steps, dtype=torch.long, device=device_fx)

        result = _run(boards, pivot_idx, temporal_dist)

        assert (result == _POS).sum(dim=(1, 2)).eq(1).all().item()

    @pytest.mark.parametrize('m', [32])
    def test_walls_never_modified(self, m, device_fx):
        """Wall cells in the input are always wall cells in the output."""
        boards = torch.zeros(m, 5, 5, dtype=torch.long, device=device_fx)
        boards[:, 0, :] = _WLL
        boards[:, -1, :] = _WLL
        boards[:, :, 0] = _WLL
        boards[:, :, -1] = _WLL
        boards[:, 1, 1] = _POS
        pivot_idx = torch.zeros(m, dtype=torch.long, device=device_fx)
        temporal_dist = torch.ones(m, dtype=torch.long, device=device_fx)

        result = _run(boards, pivot_idx, temporal_dist)

        assert (result[boards == _WLL] == _WLL).all().item()

    @pytest.mark.parametrize('m', [64])
    def test_agent_never_lands_on_wall(self, m, device_fx):
        """The agent's final position is never a wall cell."""
        board = torch.tensor(
            [
                [_FLR, _WLL, _FLR, _FLR, _FLR],
                [_FLR, _POS, _FLR, _WLL, _FLR],
                [_FLR, _WLL, _FLR, _FLR, _FLR],
                [_FLR, _FLR, _FLR, _FLR, _FLR],
                [_FLR, _FLR, _FLR, _FLR, _FLR],
            ],
            dtype=torch.long,
            device=device_fx,
        )
        boards = board.unsqueeze(0).expand(m, -1, -1).clone()
        pivot_idx = torch.zeros(m, dtype=torch.long, device=device_fx)
        temporal_dist = torch.full((m,), 4, dtype=torch.long, device=device_fx)

        result = _run(boards, pivot_idx, temporal_dist)

        assert ((result == _POS) & (boards == _WLL)).sum().item() == 0

    def test_no_backtracking_in_corridor(self, device_fx):
        """In a one-way corridor, anti-backtracking forces the agent to advance monotonically.

        Agent starts at column 0. The only reachable direction is right; once a step is taken
        the previous cell is excluded, so the agent must reach column 3 after 3 steps.
        """
        board = torch.tensor(
            [
                [_WLL, _WLL, _WLL, _WLL, _WLL],
                [_POS, _FLR, _FLR, _FLR, _FLR],
                [_WLL, _WLL, _WLL, _WLL, _WLL],
                [_WLL, _WLL, _WLL, _WLL, _WLL],
                [_WLL, _WLL, _WLL, _WLL, _WLL],
            ],
            dtype=torch.long,
            device=device_fx,
        )
        boards = board.unsqueeze(0)
        pivot_idx = torch.zeros(1, dtype=torch.long, device=device_fx)
        temporal_dist = torch.tensor([3], dtype=torch.long, device=device_fx)

        result = _run(boards, pivot_idx, temporal_dist)

        expected = board.clone()
        expected[1, 0] = _FLR
        expected[1, 3] = _POS
        assert_close(result[0], expected)

    @pytest.mark.parametrize('m', [16])
    def test_target_preserved_when_not_reached(self, m, device_fx):
        """Target markers that the agent cannot reach in 1 step survive intact."""
        board = torch.tensor(
            [
                [_POS, _FLR, _FLR],
                [_FLR, _FLR, _FLR],
                [_FLR, _FLR, _TGT],
            ],
            dtype=torch.long,
            device=device_fx,
        )
        boards = board.unsqueeze(0).expand(m, -1, -1).clone()
        pivot_idx = torch.zeros(m, dtype=torch.long, device=device_fx)
        temporal_dist = torch.ones(m, dtype=torch.long, device=device_fx)

        result = _run(boards, pivot_idx, temporal_dist)

        assert (result[:, 2, 2] == _TGT).all().item()

    def test_target_covered_by_agent(self, device_fx):
        """When the agent walks onto a target cell, the target value is overwritten by POS."""
        board = torch.tensor(
            [
                [_WLL, _WLL, _WLL],
                [_POS, _TGT, _WLL],
                [_WLL, _WLL, _WLL],
            ],
            dtype=torch.long,
            device=device_fx,
        )
        boards = board.unsqueeze(0)
        pivot_idx = torch.zeros(1, dtype=torch.long, device=device_fx)
        temporal_dist = torch.tensor([1], dtype=torch.long, device=device_fx)

        result = _run(boards, pivot_idx, temporal_dist)

        assert result[0, 1, 0].item() == _FLR  # vacated cell cleared to floor
        assert result[0, 1, 1].item() == _POS  # agent covers target
