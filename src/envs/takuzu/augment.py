from abc import abstractmethod

import gin
import torch

from src.envs.augment import AbstractAugmentation, InnerResult
from src.envs.utils import indices_to_mask, sample_random_on


class AbstractTakuzuAugmentation(AbstractAugmentation):
    """
    Abstract class for Takuzu-specific augmentations.
    If custom_positive is True,
    creates a positive example with the same pixel placement as the negative.
    """

    def supports_custom_positive(self) -> bool:
        return True

    @abstractmethod
    def _augment(
        self, pivot_state: torch.Tensor, sampled_mask: torch.Tensor, full_board: torch.Tensor
    ) -> torch.Tensor:
        pass

    def _inner_call(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        pivot_idx: torch.Tensor,
        temporal_dist: torch.Tensor,
        batch_idx: torch.Tensor,
        custom_positive: bool = False,
    ) -> InnerResult:
        pivot_state = trajectories[batch_idx, pivot_idx]
        full_board = trajectories[batch_idx, lengths - 1]
        max_t_dist = torch.amax(temporal_dist).item()

        indices, valid = sample_random_on(pivot_state == 0, max_t_dist)
        pos_range = torch.arange(max_t_dist, device=pivot_state.device).unsqueeze(0)
        valid = valid & (pos_range < temporal_dist.unsqueeze(1))
        sampled_mask = indices_to_mask(indices, valid, pivot_state)

        negative = self._augment(pivot_state, sampled_mask, full_board)
        if custom_positive:
            positive = trajectories[batch_idx, pivot_idx].clone()
            positive[sampled_mask] = full_board[sampled_mask]
            return negative, positive
        return negative


@gin.configurable
class AllBadPixelChoice(AbstractTakuzuAugmentation):
    """Fills random pixels with the wrong value."""

    def _augment(
        self, pivot_state: torch.Tensor, sampled_mask: torch.Tensor, full_board: torch.Tensor
    ) -> torch.Tensor:
        negative = pivot_state.clone()
        negative[sampled_mask] = 3 - full_board[sampled_mask]
        return negative


@gin.configurable
class MinOneBadPixelChoice(AbstractTakuzuAugmentation):
    """Fills random pixels randomly, but guarantees that at least one is wrong."""

    def _augment(
        self, pivot_state: torch.Tensor, sampled_mask: torch.Tensor, full_board: torch.Tensor
    ) -> torch.Tensor:
        negative = pivot_state.clone()

        random_full = torch.randint_like(full_board, low=1, high=3)
        negative[sampled_mask] = random_full[sampled_mask]

        # Pick one already-sampled cell per board and force it to the wrong value,
        # so at least one corruption is guaranteed. Boards with no sampled cells
        # (full board or temporal_dist == 0) yield an empty bad_mask and are skipped.
        bad_idx, bad_valid = sample_random_on(sampled_mask, 1)
        bad_mask = indices_to_mask(bad_idx, bad_valid, pivot_state)
        negative[bad_mask] = 3 - full_board[bad_mask]

        return negative


@gin.configurable
class AllRandomPixelChoice(AbstractTakuzuAugmentation):
    """Fills random pixels randomly."""

    def _augment(
        self, pivot_state: torch.Tensor, sampled_mask: torch.Tensor, full_board: torch.Tensor
    ) -> torch.Tensor:
        negative = pivot_state.clone()
        random_full = torch.randint_like(full_board, low=1, high=3)
        negative[sampled_mask] = random_full[sampled_mask]
        return negative
