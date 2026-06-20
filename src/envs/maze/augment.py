import gin
import torch

from src.envs.augment import AbstractAugmentation, InnerResult
from src.envs.maze.utils import get_pos_indices


@gin.configurable
class RandomDirectionMazeAugmentation(AbstractAugmentation):
    """Works on a batch of mazes with floor, walls and 1 agent position"""

    def __init__(
        self,
        default_temporal_dist: int = None,
        floor_marker: int = 0,
        wall_marker: int = 1,
        pos_marker: int = 2,
    ) -> None:
        super().__init__(default_temporal_dist)
        self.pos_marker = pos_marker
        self.floor_marker = floor_marker
        self.wall_marker = wall_marker

    def _inner_call(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        pivot_idx: torch.Tensor,
        temporal_dist: torch.Tensor,
        batch_idx: torch.Tensor,
        custom_positive: bool = False,
    ) -> InnerResult:
        assert custom_positive is False, "Custom positive doesn't make sense for mazes"

        grid_dim = trajectories.shape[-1]
        board = trajectories[batch_idx, pivot_idx].clone()
        curr_pos = get_pos_indices(board, grid_dim, self.pos_marker)
        # Start with prev pos at the next trajectory state (pivot + 1),
        # so that the agent goes the other way.
        prev_pos = get_pos_indices(
            trajectories[batch_idx, pivot_idx + 1], grid_dim, self.pos_marker
        )
        directions = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]], device=board.device)

        moves_left = temporal_dist.clone()
        while (moves_left > 0).any().item():
            # Avoid moving out of bounds
            candidate_pos = curr_pos.unsqueeze(1) + directions.unsqueeze(0)
            in_bounds_lo = (candidate_pos >= 0).all(dim=2)
            in_bounds_hi = (candidate_pos < board.shape[1]).all(dim=2)
            valid_moves = in_bounds_lo & in_bounds_hi

            # Avoid moving into walls
            candidate_pos = torch.clamp(candidate_pos, 0, board.shape[1] - 1)
            candidate_cells = board[
                batch_idx.unsqueeze(1), candidate_pos[:, :, 0], candidate_pos[:, :, 1]
            ]
            valid_moves &= candidate_cells != self.wall_marker

            # Avoid backtracking to previous position
            is_prev = (candidate_pos == prev_pos.unsqueeze(1)).all(dim=2)
            valid_moves &= ~is_prev

            will_move = (moves_left > 0) & valid_moves.any(dim=1)
            if not will_move.any():
                break

            # Only sample for rows that have at least one valid move to avoid errors.
            chosen_directions = torch.multinomial(valid_moves[will_move].float(), num_samples=1)
            moved_from = curr_pos.clone()
            curr_pos[will_move] += directions[chosen_directions.squeeze(1)]
            prev_pos[will_move] = moved_from[will_move]
            moves_left[will_move] -= 1

        # Update the board with the new agent position
        board[board == self.pos_marker] = self.floor_marker
        board_arange = torch.arange(board.shape[0], device=board.device)
        board[board_arange, curr_pos[:, 0], curr_pos[:, 1]] = self.pos_marker
        return board
