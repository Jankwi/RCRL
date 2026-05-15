from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from matplotlib.figure import Figure

from envs.state_sets import AbstractVectorizedStateSet, SlowVectorizedStateSet


class VectorizedEnvironment(ABC):
    def __init__(self, action_tensor: torch.Tensor) -> None:
        self.action_tensor = action_tensor
        self.action_space_size = len(self.action_tensor)

    """Abstract class, detailing the interface that a vectorized environment should have"""

    @abstractmethod
    def generate_goals(self, states: torch.Tensor) -> torch.Tensor:
        pass

    def preview_all_actions(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a (result, legality) tuple for all actions on the given states
            - doesn't corrupt original states.
        """
        n, k = states.shape[0], self.action_space_size
        st_shape, act_shape = states.shape[1:], self.action_tensor.shape[1:]
        expanded_states = states.repeat_interleave(k, dim=0)
        action_tensors = self.action_tensor.repeat(n, *([1] * len(act_shape)))
        new_states, legality = self.preview_actions(expanded_states, action_tensors)
        return new_states.reshape(n, k, *st_shape), legality.reshape(n, k)

    @abstractmethod
    def preview_actions(
        self, states: torch.Tensor, action_tensors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each state action pair, returns a (result, legality) tuple
            - corrupts original states
        """
        pass

    def preview_action(
        self, states: torch.Tensor, action_no: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        act_shape = len(states), *([-1] * len(self.action_tensor.shape[1:]))
        action_tensors = self.action_tensor[action_no].clone().unsqueeze(0).expand(act_shape)
        return self.preview_actions(states, action_tensors)

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
