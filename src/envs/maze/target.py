import gin
import torch

from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.maze import AbstractVectorizedMaze


@gin.configurable
class VectorizedTargetMaze(AbstractVectorizedMaze):
    """
    A VectorizedMazeEnvironment in which the goal is a single point.
    """

    def __init__(
        self,
        device: torch.device,
        grid_dim: int = 20,
        target_marker: int = 2,
        pos_marker: int = 3,
        spec_mgr: SpecialStateManager = None,
    ) -> None:
        super().__init__(
            device=device, grid_dim=grid_dim, target_marker=target_marker, pos_marker=pos_marker
        )

        self.spec_mgr = spec_mgr

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        """
        If special manager is configured:
            Returns the special state.
        else:
            Returns the actual goal state.
        """
        if self.spec_mgr is not None:
            return self.spec_mgr.create_many(len(states))

        goals, goals_no_pos = states.clone(), states.clone()
        have_target = torch.any((states == self.target_marker), dim=(-1, -2)).view(-1, 1, 1)
        goals_no_pos[states == self.pos_marker] = 0
        goals_no_pos[states == self.target_marker] = self.pos_marker
        return torch.where(have_target, goals_no_pos, goals)

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return torch.all(states != self.target_marker, dim=(1, 2))
