import pytest
import torch
from torch.testing import assert_close

from contrastivelib.utils.special import SpecialStateManager
from src.common import fix_seed
from src.envs.takuzu import VectorizedTakuzu

_BLUE = 1
_YELLOW = 2


class TestVectorizedTakuzu:
    """Tests for VectorizedTakuzu"""

    def setup_method(self, method):
        self.seed = 42
        fix_seed(self.seed)

    def _make_env(self, n: int, device: torch.device) -> VectorizedTakuzu:
        spec_mgr = SpecialStateManager(val=(-1), shape=(n, n), device=device)
        return VectorizedTakuzu(
            n=n, spec_mgr=spec_mgr, device=device, blue_marker=_BLUE, yellow_marker=_YELLOW
        )

    @pytest.mark.parametrize('n', [4, 8])
    @pytest.mark.parametrize('m_boards', [30])
    def test_preview_actions(self, n, m_boards, device_fx):
        base_boards = torch.randint(0, 3, (m_boards, n, n), dtype=torch.uint8, device=device_fx)
        env = self._make_env(n, device_fx)

        ij = torch.randint(0, n, (m_boards, 2), device=device_fx)
        color_idx = torch.randint(0, 2, (m_boards,), device=device_fx)

        action_nos = (ij[:, 0] * n + ij[:, 1]) * 2 + color_idx
        place_coords = (torch.arange(m_boards), ij[:, 0], ij[:, 1])

        expected_boards = base_boards.clone()
        expected_boards[place_coords] = ((1 - color_idx) * _BLUE + color_idx * _YELLOW).to(
            torch.uint8
        )
        expected_legality = base_boards[place_coords] == 0

        action_nos_long = action_nos.to(torch.long)
        boards, legality = env.preview_actions(base_boards, env.action_tensor[action_nos_long])
        assert_close(legality, expected_legality)
        assert_close(boards[legality], expected_boards[legality])

    @pytest.mark.parametrize(
        'boards, expected_solved',
        [
            (
                [
                    [  # solved board
                        [_BLUE, _YELLOW, _BLUE, _YELLOW],
                        [_YELLOW, _BLUE, _BLUE, _YELLOW],
                        [_BLUE, _YELLOW, _YELLOW, _BLUE],
                        [_YELLOW, _BLUE, _YELLOW, _BLUE],
                    ],
                    [  # 3 pixels are yellow in one row/column
                        [_YELLOW, _YELLOW, _BLUE, _YELLOW],
                        [_YELLOW, _BLUE, _BLUE, _YELLOW],
                        [_BLUE, _YELLOW, _YELLOW, _BLUE],
                        [_YELLOW, _BLUE, _YELLOW, _BLUE],
                    ],
                    [  # rows/columns are the same
                        [_BLUE, _BLUE, _YELLOW, _YELLOW],
                        [_BLUE, _BLUE, _YELLOW, _YELLOW],
                        [_YELLOW, _YELLOW, _BLUE, _BLUE],
                        [_YELLOW, _YELLOW, _BLUE, _BLUE],
                    ],
                    [  # solved board
                        [_YELLOW, _BLUE, _BLUE, _YELLOW],
                        [_BLUE, _BLUE, _YELLOW, _YELLOW],
                        [_BLUE, _YELLOW, _YELLOW, _BLUE],
                        [_YELLOW, _YELLOW, _BLUE, _BLUE],
                    ],
                    [  # everything wrong
                        4 * [_BLUE],
                        4 * [_BLUE],
                        4 * [_BLUE],
                        4 * [_BLUE],
                    ],
                ],
                [True, False, False, True, False],
            ),
            (
                [
                    [  # 3 of the same color in a row (But perfectly balanced and unique)
                        [_YELLOW, _YELLOW, _YELLOW, _BLUE, _BLUE, _BLUE],
                        [_BLUE, _BLUE, _BLUE, _YELLOW, _YELLOW, _YELLOW],
                        [_YELLOW, _BLUE, _YELLOW, _BLUE, _YELLOW, _BLUE],
                        [_BLUE, _YELLOW, _BLUE, _YELLOW, _BLUE, _YELLOW],
                        [_YELLOW, _BLUE, _BLUE, _YELLOW, _YELLOW, _BLUE],
                        [_BLUE, _YELLOW, _YELLOW, _BLUE, _BLUE, _YELLOW],
                    ],
                    [  # solved
                        [_BLUE, _YELLOW, _BLUE, _YELLOW, _YELLOW, _BLUE],
                        [_YELLOW, _BLUE, _BLUE, _YELLOW, _BLUE, _YELLOW],
                        [_BLUE, _BLUE, _YELLOW, _BLUE, _YELLOW, _YELLOW],
                        [_BLUE, _YELLOW, _YELLOW, _BLUE, _YELLOW, _BLUE],
                        [_YELLOW, _YELLOW, _BLUE, _YELLOW, _BLUE, _BLUE],
                        [_YELLOW, _BLUE, _YELLOW, _BLUE, _BLUE, _YELLOW],
                    ],
                    [  # almost solved = unsolved
                        [_BLUE, _YELLOW, _BLUE, _YELLOW, _YELLOW, _BLUE],
                        [_YELLOW, _BLUE, _BLUE, _YELLOW, _BLUE, _YELLOW],
                        [_BLUE, 0, _YELLOW, _BLUE, _YELLOW, _YELLOW],
                        [_BLUE, _YELLOW, _YELLOW, _BLUE, _YELLOW, _BLUE],
                        [_YELLOW, _YELLOW, _BLUE, _YELLOW, _BLUE, _BLUE],
                        [_YELLOW, _BLUE, _YELLOW, _BLUE, _BLUE, _YELLOW],
                    ],
                ],
                [False, True, False],
            ),
        ],
    )
    def test_is_solved(self, boards, expected_solved, device_fx=None):
        boards = torch.tensor(boards, dtype=torch.uint8, device=device_fx)
        env = self._make_env(boards.shape[-1], device_fx)
        is_solved = env.is_solved(boards, None)
        assert_close(is_solved, torch.tensor(expected_solved, device=device_fx))
