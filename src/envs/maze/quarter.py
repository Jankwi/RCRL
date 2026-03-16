import gin
import torch

from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.maze import AbstractVectorizedMaze
from src.envs.maze.utils import generate_border_positions


@gin.configurable
class VectorizedQuarterMaze(AbstractVectorizedMaze):
    """
    A VectorizedMazeEnvironment in which the goal is reaching the top left quarter of the board.
    """

    def __init__(
        self,
        device: torch.device,
        grid_dim: int = 20,
        pos_marker: int = 2,
        spec_mgr: SpecialStateManager = None,
    ) -> None:
        super().__init__(device=device, grid_dim=grid_dim, pos_marker=pos_marker)
        self.quarter_tensor = torch.sum(self._get_quarter_pos(), dim=0)

        self.spec_mgr = spec_mgr
        if self.spec_mgr is None:
            print(
                'Warning: Maze environment was initialized with spec_mgr=None : '
                'generate_goals() will return all possible states'
            )

    def _get_quarter_pos(self) -> torch.Tensor:
        return generate_border_positions(
            border=int(self.grid_dim / 2), grid_dim=self.grid_dim, device=self.device
        )

    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        """
        If special manager is configured:
            Returns the special state.
        else:
            Returns a tensor containing all valid ending positions, some of them duplicated.
            It is a bit hacky, but it was either this or going down a pure object-oriented road.
        """
        if self.spec_mgr is not None:
            return self.spec_mgr.create_many(len(states))

        goals = states.clone()
        goals[goals == self.pos_marker] = 0
        goals = goals.unsqueeze(1) + self._get_quarter_pos().to(torch.long) * self.pos_marker
        valid = torch.all((goals <= self.pos_marker), dim=(2, 3))
        first_valid_idx = valid.long().argmax(dim=1)
        fill = goals[torch.arange(len(goals), device=goals.device), first_valid_idx]
        goals = torch.where(valid.unsqueeze(-1).unsqueeze(-1), goals, fill.unsqueeze(1))
        return goals

    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        quarter_positions = torch.logical_and(self.quarter_tensor, (states == self.pos_marker))
        return torch.any(quarter_positions, dim=(1, 2))
