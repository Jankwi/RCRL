import gin
import torch

from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.maze import MaxOneTargetMaze


@gin.configurable
class VectorizedTargetMaze(MaxOneTargetMaze):
    """
    A VectorizedMazeEnvironment in which the goal is a single point.
    """

    def __init__(
        self,
        device: torch.device,
        grid_dim: int = 20,
        floor_marker: int = 0,
        wall_marker: int = 1,
        pos_marker: int = 2,
        target_marker: int = 3,  # Override target_marker defaulting to None
        spec_mgr: SpecialStateManager = None,
        use_special: bool = True,
    ) -> None:
        super().__init__(
            device=device,
            grid_dim=grid_dim,
            floor_marker=floor_marker,
            wall_marker=wall_marker,
            pos_marker=pos_marker,
            target_marker=target_marker,
            spec_mgr=spec_mgr,
        )
        self.use_special = use_special

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        """
        If special manager is configured:
            Returns the special state.
        else:
            Returns the actual goal state.
        Can be overridden by use_special.
        """
        if self.spec_mgr is not None and self.use_special:
            return self.spec_mgr.create_many(len(states))

        goals, goals_no_pos = states.clone(), states.clone()
        have_target = torch.any((states == self.target_marker), dim=(-1, -2)).view(-1, 1, 1)
        goals_no_pos[states == self.pos_marker] = self.floor_marker
        goals_no_pos[states == self.target_marker] = self.pos_marker
        return torch.where(have_target, goals_no_pos, goals)

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Checks if the agent covers the target"""
        return torch.all(states != self.target_marker, dim=(1, 2))
