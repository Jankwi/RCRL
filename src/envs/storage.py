from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import gin
import torch
from tensordict import TensorDict

from src.envs import VectorizedEnvironment


@gin.configurable
class AbstractStateStorage(ABC):
    """
    A layer of abstraction over the current states, for enivornment rollouts.
    Allows the user to operate on local indexing (of active states).
    """

    def __init__(
        self,
        starting_positions: torch.Tensor,
        env: VectorizedEnvironment,
        device: torch.device,
        answers: torch.Tensor = None,
    ) -> None:
        self.device = device
        self.env = env

        self.n_games, self.solved_cnt = starting_positions.shape[0], 0
        self.active_idx = torch.arange(self.n_games, device=self.device)

        self.states = starting_positions.clone()
        self.answers = answers
        if self.env.requires_answers_for_goal_generation():
            self.goals = self.env.generate_goals(self.states, self.answers)
        else:
            self.goals = self.env.generate_goals(self.states)
        self.sets = self.env.get_state_sets(n_sets=len(self.states))

    @abstractmethod
    def new_iteration(self, solved_mask: torch.Tensor) -> None:
        pass

    def active_states(self) -> torch.Tensor:
        """Returns all the states, for which exploration hasn't yet finished."""
        # This is important, as is_solved may crash when getting an empty tensor
        if len(self.active_idx) == 0:
            return torch.empty(
                (0, *self.states.shape[1:]), device=self.device, dtype=self.states.dtype
            )

        if self.env.requires_answers_for_solved_checking():
            solved_mask = self.env.is_solved(
                self.states[self.active_idx],
                self.goals[self.active_idx],
                self.answers[self.active_idx],
            )
        else:
            solved_mask = self.env.is_solved(
                self.states[self.active_idx], self.goals[self.active_idx]
            )
        self.new_iteration(solved_mask)

        self.solved_cnt += solved_mask.sum().item()
        self.active_idx = self.active_idx[~solved_mask]

        chosen = self.states[self.active_idx].clone()
        self.sets.add_to_sets(chosen, self.active_idx)
        return chosen

    def active_goals(self) -> torch.Tensor:
        return self.goals[self.active_idx]

    def process_chosen_moves(self, chosen_moves: torch.Tensor, still_active: torch.Tensor) -> None:
        self.active_idx = self.active_idx[still_active]
        self.states[self.active_idx] = chosen_moves[still_active]

    def wasnt_visited(self, candidates: torch.Tensor) -> torch.Tensor:
        return self.sets.missing_from_sets(candidates, self.active_idx)

    def wasnt_visited_batched(
        self, candidates: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """Batched per-action check: candidates [n, k, *state_shape], validity [n, k] → [n, k] bool.

        Only valid entries are sent to the state set, so corrupted illegal-move boards
        (e.g. maze previews) never reach `missing_from_sets`.
        """
        n, k = candidates.shape[0], candidates.shape[1]
        flat = candidates.reshape(n * k, *candidates.shape[2:])
        flat_valid = validity.reshape(n * k)
        repeated_idx = self.active_idx.repeat_interleave(k)

        result = torch.zeros(n * k, dtype=torch.bool, device=candidates.device)
        if flat_valid.any():
            result[flat_valid] = self.sets.missing_from_sets(
                flat[flat_valid], repeated_idx[flat_valid]
            )
        return result.reshape(n, k)

    def _get_solved_rate(self) -> Dict[str, float]:
        return {
            'solved_rate': self.solved_cnt / self.n_games,
        }

    @abstractmethod
    def continue_condition(self) -> bool:
        pass

    @abstractmethod
    def get_result(self) -> Any:  # noqa: ANN401
        pass


@gin.configurable
class ShallowStateStorage(AbstractStateStorage):
    """
    Shallow state storage: stores only the current state of the rollout.
    Returns just the solved rate.
    """

    def continue_condition(self) -> bool:
        return len(self.active_idx) > 0

    def new_iteration(self, solved_mask: torch.Tensor) -> None:
        return

    def get_result(self) -> Dict[str, float]:
        return self._get_solved_rate()


@gin.configurable
class DeepStateStorage(AbstractStateStorage):
    """Deep state storage - tracks all subsequent positions, lengths and outcomes of rollouts"""

    def __init__(
        self,
        starting_positions: torch.Tensor,
        env: VectorizedEnvironment,
        device: torch.device,
        answers: torch.Tensor = None,
        max_len: int = 128,
    ) -> None:
        super().__init__(
            starting_positions=starting_positions, env=env, device=device, answers=answers
        )
        state_shape = starting_positions.shape[1:]
        self.max_len = max_len
        self.trajectories = torch.zeros(
            (self.n_games, max_len, *state_shape),
            device=self.device,
            dtype=starting_positions.dtype,
        )
        self.lengths = torch.zeros((self.n_games,), device=self.device, dtype=torch.long)
        self.outcome = torch.zeros((self.n_games,), device=self.device, dtype=torch.bool)
        self.iteration = 0

    def new_iteration(self, solved_mask: torch.Tensor) -> None:
        self.trajectories[self.active_idx, self.iteration] = self.states[self.active_idx].clone()
        self.lengths[self.active_idx] += 1
        self.iteration += 1
        self.outcome[self.active_idx[solved_mask]] = True

    def continue_condition(self) -> bool:
        return len(self.active_idx) > 0 and self.iteration < self.max_len

    def get_result(self) -> Tuple[TensorDict, Dict[str, float]]:
        tensors = TensorDict(
            {
                'trajectories': self.trajectories,
                'lengths': self.lengths,
                'outcome': self.outcome,
                'goals': self.goals,
            },
            batch_size=[self.n_games],
        )
        return tensors, self._get_solved_rate()
