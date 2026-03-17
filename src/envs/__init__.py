from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from matplotlib.figure import Figure

from envs.state_sets import AbstractVectorizedStateSet, SlowVectorizedStateSet


class VectorizedEnvironment(ABC):
    def __init__(self, action_space_size: int) -> None:
        self.action_space_size = action_space_size

    """Abstract class, detailing the interface that a vectorized environment should have"""

    @abstractmethod
    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def preview_action(
        self, states: torch.Tensor, action_no: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def is_solved(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        pass

    def actions(self) -> range:
        return range(self.action_space_size)

    def get_state_sets(self, n_sets: int) -> AbstractVectorizedStateSet:
        """
        Returns a new instance of a AbstractVectorizedStateSets subclass,
        which is recommended for the environment.

        If not overriden, the default is SlowVectorizedStateSet (works with all environments).
        """
        return SlowVectorizedStateSet(n_sets=n_sets)

    @abstractmethod
    def render_state(self, state: np.ndarray) -> Figure:
        pass


class ValueEstimator(ABC):
    """Layer of abstraction for estimating the value of positions"""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def get_solved_distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        pass

    def get_inf(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Returns a tensor of +inf with the given shape, used to initialise best_val."""
        return torch.full(shape, float('inf'), device=self.device)

    def which_inf(self, val: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask of entries that are still +inf (no valid move found)."""
        return torch.isinf(val)
