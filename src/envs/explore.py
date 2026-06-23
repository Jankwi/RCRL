# ruff: noqa: F401
from typing import Any, Dict, Type

import gin
import torch

from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.storage import AbstractStateStorage, DeepStateStorage, ShallowStateStorage


@gin.configurable
def eval_no_search(
    states: torch.Tensor,
    env: VectorizedEnvironment,
    ve: ValueEstimator,
    storage_class: Type[AbstractStateStorage] = ShallowStateStorage,
    max_len: Dict[str, int] = {},
    answers: torch.Tensor = None,
    prune: bool = False,
) -> Any:  # noqa: ANN401
    """Runs a greedy simulation for each state, guided by the value estimator.

    When prune=True, a chosen board that can no longer reach its answer (env.is_unsolvable) is
    dropped from the rollout early instead of being expanded to completion. This is a speedup: a
    pruned board was bound to fail is_solved anyway. Pruning never filters candidates before the
    value estimator chooses, so the model is still free to pick doomed boards (keeping solved_rate
    honest).
    """
    if prune:
        assert env.supports_pruning(), f'{type(env).__name__} does not support pruning'
        assert answers is not None, 'pruning requires answers to be provided'
        assert not issubclass(storage_class, DeepStateStorage), (
            'pruning is unsafe with DeepStateStorage: it truncates rollouts and would corrupt '
            'the collected trajectories'
        )
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

        still_active = ~ve.which_inf(best_val)
        if prune:
            still_active = still_active & ~env.is_unsolvable(best_next, storage.active_answers())
        storage.process_chosen_moves(best_next, still_active)
        states = storage.active_states()

    return storage.get_result()
