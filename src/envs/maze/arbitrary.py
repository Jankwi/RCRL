import gin
import torch

from src.envs.maze import MaxOneTargetMaze
from src.envs.utils import sample_random_on


@gin.configurable
class ArbitraryPositionMaze(MaxOneTargetMaze):
    """
    A maze env in which the goal is an arbitrary agent position.
    Works with mazes both containing and not containing target markers.
    """

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        end_pos = sample_random_on(states != self.wall_marker, 1)[0].squeeze(1)

        goals = states.clone()
        goals[states == self.pos_marker] = self.floor_marker  # vacate old agent position
        batch_indices = torch.arange(states.shape[0], device=states.device)
        goals[batch_indices, end_pos // self.grid_dim, end_pos % self.grid_dim] = self.pos_marker
        return goals

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return torch.all(states == goals, dim=(1, 2))
