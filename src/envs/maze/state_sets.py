import torch

from src.envs.maze.utils import get_pos_indices
from src.envs.state_sets import AbstractVectorizedStateSet


class MazeVectorizedStateSet(AbstractVectorizedStateSet):
    def __init__(self, n_sets: int, grid_dim: int, device: torch.device) -> None:
        self.device, self.grid_dim = device, grid_dim
        self.sets = torch.zeros((n_sets, grid_dim, grid_dim), device=self.device)

    def add_to_sets(self, states: torch.Tensor, idx: torch.Tensor) -> None:
        state_pos = get_pos_indices(states, self.grid_dim)
        self.sets[idx, state_pos[:, 0], state_pos[:, 1]] = 1

    def missing_from_sets(self, candidates: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        cand_pos = get_pos_indices(candidates, self.grid_dim)
        return self.sets[idx, cand_pos[:, 0], cand_pos[:, 1]] == 0
