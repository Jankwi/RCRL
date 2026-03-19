import pytest
import torch
from torch.testing import assert_close

from src.envs.maze.state_sets import MazeVectorizedStateSet


class TestMazeVectorizedStateSet:
    """Tests VectorizedEnvironment's methods related to sets"""

    def setup_method(self, method):
        """Initialize shared constants and maze templates"""
        torch.manual_seed(42)
        self.pos_marker = 3
        self.tol = 1e-5

    def add_and_check(self, sets, states, idx):
        """Helper: check missing BEFORE, add to sets, check NOT missing AFTER"""
        missing_before = sets.missing_from_sets(states, idx=idx)
        sets.add_to_sets(states, idx=idx)
        missing_after = sets.missing_from_sets(states, idx=idx)

        assert_close(torch.ones_like(missing_before), missing_before)
        assert_close(torch.zeros_like(missing_after), missing_after)

    @pytest.mark.parametrize(
        'wall_layout',
        [
            [[[1, 0], [0, 0]], [[1, 1], [0, 1]], [[1, 1], [0, 1]]],  # (3x) 2x2
            [[[1, 1, 1], [1, 2, 1], [1, 0, 0]], [[2, 0, 0], [1, 0, 1], [1, 0, 1]]],  # (2x) 3x3
            [[[1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]]],  # (1x) 4x4
        ],
    )
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_get_empty_set(self, wall_layout, device):
        """Verify get_empty_sets returns a set, from which every legal position is missing"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        wall_layout_tensor = torch.tensor(wall_layout, device=device)
        n_sets, grid_dim = wall_layout_tensor.shape[0], wall_layout_tensor.shape[-1]
        empty_sets = MazeVectorizedStateSet(n_sets=n_sets, device=device, grid_dim=grid_dim)

        for i in range(grid_dim):
            for j in range(grid_dim):
                idx = wall_layout_tensor[:, i, j] != 1

                if torch.any(idx).item():
                    states = torch.clone(wall_layout_tensor[idx])
                    states[:, i, j] = self.pos_marker

                    is_missing = empty_sets.missing_from_sets(states, idx)
                    assert_close(torch.ones_like(is_missing), is_missing)

    @pytest.mark.parametrize(
        'states_list',
        [
            [[[3, 1], [0, 1]], [[3, 1], [2, 1]], [[0, 0], [0, 3]]],  # (3x) 2x2
            [[[1, 1, 1], [1, 3, 1], [1, 0, 0]], [[0, 3, 0], [1, 0, 1], [1, 0, 1]]],  # (2x) 3x3
            [[[0, 1, 1, 0], [0, 1, 3, 0], [0, 0, 0, 0], [1, 1, 0, 1]]],  # (1x) 4x4
        ],
    )
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_add_to_sets_basic(self, states_list, device):
        """Verify add_to_sets marks positions correctly"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        states_tensor = torch.tensor(states_list, device=device)
        n_sets, grid_dim = states_tensor.shape[0], states_tensor.shape[-1]
        sets = MazeVectorizedStateSet(n_sets=n_sets, device=device, grid_dim=grid_dim)

        idx = torch.tensor([i for i in range(n_sets)], device=device)
        self.add_and_check(sets, states_tensor, idx=idx)

    @pytest.mark.parametrize(
        'n_sets, additions_idxs, checks_idxs_expected',
        (
            [
                (
                    2,
                    [
                        # Additions are designed, so that after the end:
                        #  Set 0 is full (all legal positions)
                        #  Set 1 is missing only [[0, 0], [0, 0]]
                        ([[[3, 1], [0, 1]], [[3, 0], [0, 0]]], [True, True]),
                        ([[[0, 3], [0, 0]]], [False, True]),
                        ([[[0, 1], [3, 1]]], [True, False]),
                        ([[[0, 0], [0, 3]]], [False, True]),
                    ],
                    [
                        (
                            [[[3, 1], [0, 1]], [[0, 0], [3, 0]]],
                            [True, True],
                            [False, True],
                        ),  # the second one is missing
                        ([[[0, 0], [3, 0]]], [False, True], [True]),  # is missing
                        (
                            [[[0, 1], [3, 1]]],
                            [True, False],
                            [False],
                        ),  # is not missing (present in set)
                    ],
                ),
                (  # Same as above, but with a target marker (shouldn't change anything)
                    2,
                    [
                        ([[[3, 1], [2, 1]], [[3, 0], [2, 0]]], [True, True]),
                        ([[[0, 3], [2, 0]]], [False, True]),
                        ([[[0, 1], [3, 1]]], [True, False]),
                        ([[[0, 0], [2, 3]]], [False, True]),
                    ],
                    [
                        (
                            [[[3, 1], [2, 1]], [[0, 0], [3, 0]]],
                            [True, True],
                            [False, True],
                        ),  # the second one is missing
                        ([[[0, 0], [3, 0]]], [False, True], [True]),  # is missing
                        (
                            [[[2, 1], [3, 1]]],
                            [True, False],
                            [False],
                        ),  # is not missing (present in set)
                    ],
                ),
            ]
        ),
    )
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_sequential_addition(self, n_sets, additions_idxs, checks_idxs_expected, device):
        grid_dim = len(additions_idxs[0][0][0])
        sets = MazeVectorizedStateSet(n_sets=n_sets, device=device, grid_dim=grid_dim)

        for addition_list, idx_list in additions_idxs:
            addition, idx = (
                torch.tensor(addition_list, device=device),
                torch.tensor(idx_list, device=device),
            )
            self.add_and_check(sets=sets, states=addition, idx=idx)

        for check_list, idx_list, expected_list in checks_idxs_expected:
            (
                check,
                idx,
            ) = torch.tensor(check_list, device=device), torch.tensor(idx_list, device=device)
            expected = torch.tensor(expected_list, device=device)
            assert_close(expected, sets.missing_from_sets(check, idx))
