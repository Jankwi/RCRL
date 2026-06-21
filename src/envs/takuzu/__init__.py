from abc import abstractmethod

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import VectorizedEnvironment
from src.envs.state_sets import AbstractVectorizedStateSet
from src.envs.takuzu.utils import AlwaysMissingStateSet

_VALUE_TO_COLOR = {
    0: np.array([255, 255, 255, 255], dtype=np.uint8),  # white
    1: np.array([52, 89, 149, 255], dtype=np.uint8),  # deep blue
    2: np.array([242, 196, 75, 255], dtype=np.uint8),  # warm yellow
}

_BG_COLOR = (0.08, 0.09, 0.11, 1.0)  # near-black background
_GRID_COLOR = (0.15, 0.16, 0.18, 1.0)  # subtle dark grid
_BORDER_COLOR = (0.25, 0.26, 0.3, 1.0)  # slightly lighter frame


@gin.configurable
class AbstractVectorizedTakuzu(VectorizedEnvironment):
    """
    Abstract Vectorized environment for the game of Takuzu,
    where the goal is to fill an n x n grid with 2 colors so that all 3 rules are satisfied:
    1. No more than 2 of the same color can be adjacent in any row or column
    2. Each row and column must have an equal number of each color
    3. All rows and columns must be unique
    """

    def __init__(
        self,
        n: int,
        device: torch.device,
        spec_mgr: SpecialStateManager = None,
        blue_marker: int = 1,
        yellow_marker: int = 2,
    ) -> None:
        self.n = n
        self.spec_mgr = spec_mgr
        self.blue_marker = blue_marker
        self.yellow_marker = yellow_marker
        action_tensor = torch.zeros((2 * n * n, n, n), dtype=torch.uint8, device=device)
        for i in range(n):
            for j in range(n):
                action_tensor[2 * (i * n + j)][i][j] = self.blue_marker
                action_tensor[2 * (i * n + j) + 1][i][j] = self.yellow_marker
        super().__init__(action_tensor=action_tensor)

    def env_string(self) -> str:
        return 'takuzu_goal'

    def preview_actions(
        self, states: torch.Tensor, action_tensors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_states = states + action_tensors
        legality = ((states * action_tensors) == 0).all(dim=(1, 2))
        return new_states, legality

    def _check_row_constraints(self, states: torch.Tensor) -> torch.Tensor:
        """Checks all 3 Takuzu constraints for rows, returning a boolean mask with results."""
        # 1. Check for 3 in a row
        same_as_right = states[:, :, :-1] == states[:, :, 1:]  # (M, n, n-1)
        same_as_both_right = same_as_right[:, :, :-1] & same_as_right[:, :, 1:]  # (M, n, n-2)
        mask1 = ~same_as_both_right.any(dim=(1, 2))  # (M)

        # 2. Check if each row has equal number of blue and yellow
        blue_counts = (states == self.blue_marker).sum(dim=2)
        yellow_counts = (states == self.yellow_marker).sum(dim=2)
        mask2 = (blue_counts == yellow_counts).all(dim=1)

        # 3. Check if all rows are unique, by comparing all pairs of rows
        same_rows = (states.unsqueeze(2) == states.unsqueeze(1)).all(dim=-1)  # (M, n, n)
        eye = torch.eye(states.shape[-1], dtype=torch.bool, device=states.device)
        same_rows.masked_fill_(eye, False)
        mask3 = ~same_rows.any(dim=(1, 2))

        return mask1 & mask2 & mask3

    @abstractmethod
    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        pass

    # goals are not used, because they are always abstract
    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        # Only consider tensors with all fields filled
        is_all_filled = ((states == self.blue_marker) | (states == self.yellow_marker)).all(
            dim=(1, 2)
        )
        solved = torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)

        valid_states = states[is_all_filled]
        if valid_states.shape[0] > 0:
            mask_rows = self._check_row_constraints(valid_states)
            mask_cols = self._check_row_constraints(valid_states.transpose(1, 2))
            solved[is_all_filled] = mask_rows & mask_cols

        return solved

    def get_state_sets(self, n_sets: int) -> AbstractVectorizedStateSet:
        return AlwaysMissingStateSet()

    def render_state(self, state: np.ndarray) -> Figure:
        h, w = state.shape

        image = np.zeros((h, w, 4), dtype=np.uint8)
        for value, color in _VALUE_TO_COLOR.items():
            image[state == value] = color

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(_BG_COLOR)
        ax.set_facecolor(_BG_COLOR)

        ax.imshow(image)

        # grid
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color=_GRID_COLOR, linestyle='-', linewidth=1.2)

        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # outer border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color(_BORDER_COLOR)

        ax.set_aspect('equal')

        return fig


@gin.configurable
class VectorizedTakuzu(AbstractVectorizedTakuzu):
    """
    This subclass requires a special manager, to generate goals.
    It does not condition the searcg on explicit boards.
    """

    def __init__(
        self,
        n: int,
        device: torch.device,
        spec_mgr: SpecialStateManager,  # not None by default anymore
        blue_marker: int = 1,
        yellow_marker: int = 2,
    ) -> None:
        super().__init__(
            n=n,
            device=device,
            spec_mgr=spec_mgr,
            blue_marker=blue_marker,
            yellow_marker=yellow_marker,
        )

    def env_string(self) -> str:
        return 'takuzu_compass'

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        return self.spec_mgr.create_many(states.shape[0])
