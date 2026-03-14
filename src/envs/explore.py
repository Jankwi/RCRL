from typing import Dict

import torch

from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.storage import StateStorage


def eval_no_search(
    states: torch.Tensor, env: VectorizedEnvironment, ve: ValueEstimator
) -> Dict[str, float]:
    """Runs a greedy simulation for each state, guided by the value estimator."""
    storage = StateStorage(states, env, states.device)

    states = storage.active_states()
    while len(states) > 0:
        best_next = states.clone()
        best_val = ve.get_inf((states.shape[0],))

        for action in env.actions():
            next_board, legality = env.preview_action(states.clone(), action)
            next_val = ve.get_solved_distance(next_board, storage.active_goals())
            new_candidate = (legality & storage.wasnt_visited(next_board)) & (next_val < best_val)
            best_next[new_candidate] = next_board[new_candidate]
            best_val[new_candidate] = next_val[new_candidate]

        storage.process_chosen_moves(best_next, ~ve.which_inf(best_val))
        states = storage.active_states()

    return {'solved_rate': storage.get_solved_rate()}
