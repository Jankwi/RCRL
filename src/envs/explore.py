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
) -> Any:  # noqa: ANN401
    """Runs a greedy simulation for each state, guided by the value estimator."""
    storage = storage_class(states, env, states.device, **max_len)

    states = storage.active_states()
    while storage.continue_condition():
        best_next = states.clone()
        best_val = ve.get_inf((states.shape[0],))

        for action in env.actions():
            next_board, legality = env.preview_action(states.clone(), action)

            valid_mask = legality & storage.wasnt_visited(next_board)
            next_val = ve.get_inf((states.shape[0],))
            if valid_mask.any():
                next_val[valid_mask] = ve.get_solved_distance(
                    next_board[valid_mask], storage.active_goals()[valid_mask]
                )

            new_candidate = valid_mask & (next_val < best_val)
            best_next[new_candidate] = next_board[new_candidate]
            best_val[new_candidate] = next_val[new_candidate]

        storage.process_chosen_moves(best_next, ~ve.which_inf(best_val))
        states = storage.active_states()

    return storage.get_result()
