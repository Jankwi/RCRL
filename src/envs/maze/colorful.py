import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.maze import AbstractVectorizedMaze
from src.envs.state_sets import AbstractVectorizedStateSet, SlowVectorizedStateSet

# constants used for rendering
_VALUE_TO_COLOR = {
    # Unachieved targets (bright colors)
    2: np.array([255, 0, 0, 255], dtype=np.uint8),  # Red
    3: np.array([0, 255, 0, 255], dtype=np.uint8),  # Green
    4: np.array([0, 0, 255, 255], dtype=np.uint8),  # Blue
    5: np.array([255, 255, 0, 255], dtype=np.uint8),  # Yellow
    6: np.array([255, 0, 255, 255], dtype=np.uint8),  # Magenta
    7: np.array([0, 255, 255, 255], dtype=np.uint8),  # Cyan
    8: np.array([255, 165, 0, 255], dtype=np.uint8),  # Orange
    9: np.array([128, 0, 128, 255], dtype=np.uint8),  # Purple
    # Achieved targets (darker/muted versions)
    12: np.array([128, 0, 0, 255], dtype=np.uint8),  # Dark Red
    13: np.array([0, 128, 0, 255], dtype=np.uint8),  # Dark Green
    14: np.array([0, 0, 128, 255], dtype=np.uint8),  # Dark Blue
    15: np.array([128, 128, 0, 255], dtype=np.uint8),  # Dark Yellow
    16: np.array([128, 0, 128, 255], dtype=np.uint8),  # Dark Magenta
    17: np.array([0, 128, 128, 255], dtype=np.uint8),  # Dark Cyan
    18: np.array([128, 82, 0, 255], dtype=np.uint8),  # Dark Orange
    19: np.array([64, 0, 64, 255], dtype=np.uint8),  # Dark Purple
    # Agent (white)
    10: np.array([255, 255, 255, 255], dtype=np.uint8),
    # Walls (gray)
    0: np.array([150, 150, 150, 255], dtype=np.uint8),
    # Floor (black)
    1: np.array([0, 0, 0, 255], dtype=np.uint8),
}


@gin.configurable
class VectorizedColorfulMaze(AbstractVectorizedMaze):
    """
    A VectorizedMazeEnvironment, in which the goal is to reach all targets in a specified order.
    By default:
        * Walls are marked with 0's
        * The floor is marked with 1's
        * The agent's position is marked with a 10
        * Unsolved targets are marked with 2, 3, 4, ... (up to 9)
        * Solved targets are marked with 12, 13, 14, ... (up to 19)
    """

    def __init__(
        self,
        device: torch.device,
        grid_dim: int = 10,
        k_targets: int = 3,
        pos_marker: int = 10,
        floor_marker: int = 1,
        wall_marker: int = 0,
        spec_mgr: SpecialStateManager = None,
    ) -> None:
        assert k_targets <= (pos_marker - 2)
        super().__init__(
            device=device,
            grid_dim=grid_dim,
            pos_marker=pos_marker,
            floor_marker=floor_marker,
            wall_marker=wall_marker,
        )
        self.k_targets = k_targets
        self.spec_mgr = spec_mgr

    def get_covered_pos(self, states: torch.Tensor) -> torch.Tensor:
        """If the agent is covering the minimal unsolved target, uncovers that target as solved."""
        batch = len(states)
        n_solved = (states > self.pos_marker).view(batch, -1).sum(dim=-1)
        next_target = (n_solved + 2).view(batch, 1, 1)
        target_visible = (states == next_target).view(batch, -1).any(dim=-1)
        result = torch.full((batch,), self.floor_marker, device=self.device, dtype=states.dtype)
        result[~target_visible] = (next_target.view(batch) + self.pos_marker)[~target_visible]
        return result

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        """
        If special manager is configured:
            Returns the special state.
        else:
            Returns the actual goal state.
        Assumes that the states are not solved
        """
        if self.spec_mgr is not None:
            return self.spec_mgr.create_many(len(states))

        goals = states.clone()
        goals[states == self.pos_marker] = self.floor_marker
        goals[(1 < states) & (states < self.pos_marker)] += self.pos_marker
        goals[goals == (self.k_targets + 1 + self.pos_marker)] = self.pos_marker
        return goals

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        n_solved = (states > self.pos_marker).view(len(states), -1).sum(dim=-1)
        smaller_solved = n_solved == (self.k_targets - 1)
        last_target_covered = torch.all(states != (self.k_targets + 1), dim=(-1, -2))
        return torch.logical_and(smaller_solved, last_target_covered)

    def render_state(self, state: np.ndarray) -> Figure:
        image = np.zeros((state.shape[0], state.shape[1], 4), dtype=np.uint8)
        for value, color in _VALUE_TO_COLOR.items():
            image[state == value] = color
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image)
        ax.set_aspect('equal')
        ax.axis('off')
        return fig

    # Defaults back to the SlowVectorizedStateSet,
    # as the MazeVectorizedStateSet does not support the colorful maze yet
    def get_state_sets(self, n_sets: int) -> AbstractVectorizedStateSet:
        return SlowVectorizedStateSet(n_sets=n_sets)
