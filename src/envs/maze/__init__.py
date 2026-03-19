from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from src.envs import VectorizedEnvironment
from src.envs.maze.state_sets import MazeVectorizedStateSet
from src.envs.maze.utils import get_pos_indices


class AbstractVectorizedMaze(VectorizedEnvironment):
    """A Vectorized RL environment for a grid_dim x grid_dim maze"""

    # important: the pos_marker should have the highest value of all values used
    def __init__(
        self, device: torch.device, grid_dim: int = 20, pos_marker: int = 3, target_marker: int = 2
    ) -> None:
        super().__init__(action_space_size=4)
        self.device = device
        self.grid_dim = grid_dim
        self.lower_bound = torch.tensor([0], device=self.device)
        self.upper_bound = torch.tensor([grid_dim - 1], device=self.device)
        self.pos_marker = torch.tensor([pos_marker], device=self.device)
        self.target_marker = torch.tensor([target_marker], device=self.device)

        self.action_tensors = [
            torch.tensor([-1, 0], device=self.device),
            torch.tensor([0, 1], device=self.device),
            torch.tensor([1, 0], device=self.device),
            torch.tensor([0, -1], device=self.device),
        ]

    def preview_action(
        self, states: torch.Tensor, action_no: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a given action on each of the states.
        Note: illegal moves (hitting a wall or boundary) leave the board in a
        corrupted state. Callers must pass a clone and use the returned
        `legal_move` mask to discard results for illegal moves.
        """
        batch_idx = torch.arange(states.shape[0], device=self.device)
        action_tensor = self.action_tensors[action_no]
        old_pos = get_pos_indices(states, self.grid_dim)
        new_pos = torch.clamp(old_pos + action_tensor, min=self.lower_bound, max=self.upper_bound)

        legal_move = torch.logical_or(
            states[batch_idx, new_pos[:, 0], new_pos[:, 1]] == 0,
            states[batch_idx, new_pos[:, 0], new_pos[:, 1]] == self.target_marker,
        )
        states[batch_idx, old_pos[:, 0], old_pos[:, 1]] -= self.pos_marker
        states[batch_idx, new_pos[:, 0], new_pos[:, 1]] = self.pos_marker
        return states, legal_move

    def get_state_sets(self, n_sets: int) -> MazeVectorizedStateSet:
        return MazeVectorizedStateSet(n_sets=n_sets, grid_dim=self.grid_dim, device=self.device)

    def render_state(self, state: np.ndarray) -> Figure:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(state, cmap='viridis', vmin=-0.5, vmax=3.0)
        ax.set_aspect('equal')
        ax.axis('off')
        return fig
