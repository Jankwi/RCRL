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
        floor_marker: int = 0,
        wall_marker: int = 1,
        spec_mgr: SpecialStateManager = None,
    ) -> None:
        super().__init__(
            device=device,
            grid_dim=grid_dim,
            pos_marker=pos_marker,
            floor_marker=floor_marker,
            wall_marker=wall_marker,
        )
        self.target_marker: int = target_marker
        self.spec_mgr = spec_mgr

    def get_covered_pos(self, states: torch.Tensor) -> torch.Tensor:
        """Returns target_marker for states where the agent is covering it, else floor_marker."""
        target_visible = torch.any(states == self.target_marker, dim=(-1, -2))
        result = torch.full(
            (states.shape[0],),
            fill_value=self.floor_marker,
            device=states.device,
            dtype=states.dtype,
        )
        result[~target_visible] = self.target_marker
        return result

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
        goals_no_pos[states == self.pos_marker] = self.floor_marker
        goals_no_pos[states == self.target_marker] = self.pos_marker
        return torch.where(have_target, goals_no_pos, goals)

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return torch.all(states != self.target_marker, dim=(1, 2))
