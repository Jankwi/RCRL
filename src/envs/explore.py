# ruff: noqa: F401
from typing import Any, Dict, Type

import gin
import torch

from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.storage import AbstractStateStorage, ShallowStateStorage


@gin.configurable
def eval_no_search(
    states: torch.Tensor,
    env: VectorizedEnvironment,
    ve: ValueEstimator,
    storage_class: Type[AbstractStateStorage] = ShallowStateStorage,
    max_len: Dict[str, int] = {},
    answers: torch.Tensor = None,
) -> Any:  # noqa: ANN401
    """Runs a greedy simulation for each state, guided by the value estimator."""
    storage = storage_class(states, env, states.device, answers=answers, **max_len)

    states = storage.active_states()
    while storage.continue_condition():
        n = states.shape[0]
        next_boards, legality = env.preview_all_actions(states)
        k = next_boards.shape[1]

        valid = legality & storage.wasnt_visited_batched(next_boards, legality)
        flat_valid = valid.reshape(n * k)

        flat_next = next_boards.reshape(n * k, *next_boards.shape[2:])
        flat_goals = storage.active_goals().repeat_interleave(k, dim=0)

        flat_vals = ve.get_inf((n * k,))
        if flat_valid.any():
            flat_vals[flat_valid] = ve.get_solved_distance(
                flat_next[flat_valid], flat_goals[flat_valid]
            )

        vals = flat_vals.reshape(n, k)
        best_val, best_idx = vals.min(dim=1)
        best_next = next_boards[torch.arange(n, device=states.device), best_idx]

        storage.process_chosen_moves(best_next, ~ve.which_inf(best_val))
        states = storage.active_states()

    return storage.get_result()
