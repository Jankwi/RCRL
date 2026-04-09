import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.validation import count_solved_trajectories, count_valid_transitions, get_neighbours
from tests.test_envs.mock import MockVectorizedEnvironment


def _to_trajectories(trajectory_list, device):
    """Pad ragged trajectory lists to uniform length and convert to tensor."""
    max_len = max(len(t) for t in trajectory_list)
    state_dim = len(trajectory_list[0][0])
    padded = [t + [[0] * state_dim] * (max_len - len(t)) for t in trajectory_list]
    return torch.tensor(padded, device=device)


class TestValidation:
    def setup_method(self, method):
        self.max_idx = 20
        self.seed = 42
        fix_seed(self.seed)

    @pytest.mark.parametrize('n_states', [1, 5, 10])
    @pytest.mark.parametrize('max_length', [7])
    def test_get_neighbours(self, n_states, max_length, device_fx):
        trajectories = torch.randint(0, self.max_idx, (n_states, max_length, 2), device=device_fx)
        lengths = torch.randint(2, max_length, (n_states,), device=device_fx)

        neighbours_list = []
        for idx in range(len(trajectories)):
            for i in range(lengths[idx].item() - 1):
                neighbours_list.append(trajectories[idx, i : i + 2])
        expected_neighbours = torch.stack(neighbours_list)

        neighbours = get_neighbours(trajectories, lengths)
        assert_close(neighbours, expected_neighbours)

    @pytest.mark.parametrize(
        'trajectories, lengths, expected_valid',
        [
            # All valid
            ([[[2, 2], [2, 4], [2, 5], [2, 6]], [[1, 0], [1, 1]]], [4, 2], 4),
            # [7, 4] -> [7, 10] is invalid
            ([[[7, 2], [7, 4], [7, 10], [7, 11]]], [4], 2),
            # [1, 0] -> [8, 1] is invalid
            ([[[2, 2], [2, 4], [2, 5], [2, 6]], [[1, 0], [8, 1]]], [4, 2], 3),
            # [2, 10] -> [2, 4] and [1, 0] -> [8, 1] are invalid
            ([[[2, 10], [2, 4], [2, 5], [2, 6]], [[1, 0], [8, 1]]], [4, 2], 2),
        ],
    )
    def test_count_valid_transitions(self, trajectories, lengths, expected_valid, device_fx):
        trajectories = _to_trajectories(trajectories, device=device_fx)
        lengths = torch.tensor(lengths, device=device_fx)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device_fx)
        expected_total = torch.sum(lengths - 1).item()

        valid_count, total_count = count_valid_transitions(trajectories, lengths, env)
        assert valid_count == expected_valid
        assert total_count == expected_total

    @pytest.mark.parametrize(
        'trajectories, lengths, expected_solved',
        [
            # All solved
            ([[[6, 2], [6, 4], [6, 5], [6, 6]], [[1, 0], [1, 1]]], [4, 2], 2),
            # unsolved (also invalid, but that doesn't matter here)
            ([[[7, 2], [7, 4], [7, 10], [7, 11]]], [4], 0),
            # one solved, one not
            ([[[6, 2], [6, 4], [6, 5], [6, 6]], [[1, 0], [8, 1]]], [4, 2], 1),
            # Both solved
            ([[[6, 2], [6, 4], [6, 5], [6, 6]], [[1, 0], [1, 1]]], [4, 2], 2),
        ],
    )
    def test_count_solved_trajectories(self, trajectories, lengths, expected_solved, device_fx):
        trajectories = _to_trajectories(trajectories, device=device_fx)
        lengths = torch.tensor(lengths, device=device_fx)
        env = MockVectorizedEnvironment(max_idx=self.max_idx, device=device_fx)
        expected_total = len(trajectories)

        valid_count, total_count = count_solved_trajectories(trajectories, lengths, env)
        assert valid_count == expected_solved
        assert total_count == expected_total
