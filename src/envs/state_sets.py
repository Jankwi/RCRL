from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class AbstractVectorizedStateSet(ABC):
    """Requires the implementation of set-related methods, which depend on the evironment."""

    @abstractmethod
    def add_to_sets(self, states: torch.Tensor, idx: torch.Tensor) -> None:
        """Performs visited[idx].add(states)"""
        pass

    @abstractmethod
    def missing_from_sets(self, candidates: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Returns a torch boolean mask, mask[i] = (candidates[i] is missing from visited[idx])"""
        pass


class SlowVectorizedStateSet(AbstractVectorizedStateSet):
    def __init__(self, n_sets: int) -> None:
        self.sets = [set() for i in range(n_sets)]

    def _idx_to_list(self, idx: torch.Tensor) -> List[Tuple[int, int]]:
        """Returns pairs of (set_index, row_index_in_states_or_candidates)."""
        set_indices = idx.detach().to(device='cpu', dtype=torch.long).tolist()
        return [(set_i, row_i) for row_i, set_i in enumerate(set_indices)]

    @staticmethod
    def _state_to_key(state: torch.Tensor) -> int:
        state_cpu = state.detach().to(device='cpu')
        return hash(
            (
                tuple(state_cpu.shape),
                str(state_cpu.dtype),
                state_cpu.contiguous().numpy().tobytes(),
            )
        )

    def add_to_sets(self, states: torch.Tensor, idx: torch.Tensor) -> None:
        for set_i, row_i in self._idx_to_list(idx):
            self.sets[set_i].add(self._state_to_key(states[row_i]))

    def missing_from_sets(self, candidates: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        idx_pairs = self._idx_to_list(idx)

        result = [
            self._state_to_key(candidates[row_i]) not in self.sets[set_i]
            for set_i, row_i in idx_pairs
        ]
        return torch.tensor(result, dtype=torch.bool, device=candidates.device)
