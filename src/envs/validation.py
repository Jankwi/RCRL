from typing import Tuple

import torch

from jobs.utils.anomalies import AnomalyManager
from src.envs import VectorizedEnvironment


def get_neighbours(trajectories: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Returns a (M, 2) shaped tensor of all neighbouring states in the trajectories,
    where M = sum(lengths-1)
    """
    max_T = trajectories.shape[1]
    time_indices = torch.arange(max_T - 1, device=trajectories.device)
    mask = time_indices.unsqueeze(0) < (lengths - 1).unsqueeze(1)  # (N, max_T-1)
    batch_indices, time_idx = torch.where(mask)
    from_states = trajectories[batch_indices, time_idx]
    to_states = trajectories[batch_indices, time_idx + 1]
    return torch.stack([from_states, to_states], dim=1)


def count_valid_transitions(
    trajectories: torch.Tensor,
    lengths: torch.Tensor,
    env: VectorizedEnvironment,
    anm_mgr: AnomalyManager = None,
) -> Tuple[int, int]:
    """Returns a (valid, total) tuple of ints"""
    neighbours = get_neighbours(trajectories, lengths)
    next_state, legality = env.preview_all_actions(neighbours[:, 0])
    is_valid = torch.any(
        torch.logical_and(
            torch.all(
                neighbours[:, 1].unsqueeze(1) == next_state, dim=tuple(range(2, next_state.ndim))
            ),
            legality,
        ),
        dim=1,
    )

    if anm_mgr is not None:
        if anm_mgr.has_budget('invalid_transition') and torch.any(~is_valid).item():
            idx = torch.where(~is_valid)[0][0].item()
            with anm_mgr.open_anomaly_logging('invalid_transition'):
                # render each of the 2 neighbours
                fig = env.render_state(neighbours[idx, 0].cpu().numpy())
                anm_mgr.log_figure(fig, 'neighbour_from')

                fig = env.render_state(neighbours[idx, 1].cpu().numpy())
                anm_mgr.log_figure(fig, 'neighbour_to_invalid')

                # render each of the legal next_state options
                legal_next_states = next_state[idx][legality[idx]]
                for i, st in enumerate(legal_next_states):
                    fig = env.render_state(st.cpu().numpy())
                    anm_mgr.log_figure(fig, f'legal_option_{i}')

    return is_valid.sum().item(), len(is_valid)


def count_solved_trajectories(
    trajectories: torch.Tensor,
    lengths: torch.Tensor,
    env: VectorizedEnvironment,
    anm_mgr: AnomalyManager = None,
) -> Tuple[int, int]:
    """Returns a (solved, total) tuple of ints"""
    batch_idx = torch.arange(len(trajectories), device=trajectories.device)
    goals = env.generate_goals(states=trajectories[batch_idx, 1])
    is_solved = env.is_solved(trajectories[batch_idx, lengths - 1], goals)

    if anm_mgr is not None:
        if anm_mgr.has_budget('unsolved_trajectory') and torch.any(~is_solved).item():
            idx = torch.where(~is_solved)[0][0].item()
            with anm_mgr.open_anomaly_logging('unsolved_trajectory'):
                # render the final state of the unsolved trajectory
                fig = env.render_state(trajectories[idx, lengths[idx] - 1].cpu().numpy())
                anm_mgr.log_figure(fig, 'last_state')

    return is_solved.sum(), len(is_solved)
