import torch

from src.envs.state_sets import AbstractVectorizedStateSet


class AlwaysMissingStateSet(AbstractVectorizedStateSet):
    """Useful for Takuzu, where we cannot go back to the same state"""

    def add_to_sets(self, states: torch.Tensor, idx: torch.Tensor) -> None:
        """Performs visited[idx].add(states)"""
        pass

    def missing_from_sets(self, candidates: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Returns true for all candidates"""
        return torch.ones((len(candidates),), dtype=torch.bool, device=candidates.device)
