from abc import ABC, abstractmethod
from typing import Tuple

import gin
import torch


class AbstractTakuzuAugmentation(ABC):
    @abstractmethod
    def __call__(
        self, states: torch.Tensor, full_board: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (positive, negative) Takuzu boards."""
        pass


def sample_random_on(mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Samples k random indices where mask is true, for each batch element.
    Returns a tuple of (indices, valid_mask), both of shape (batch_size, min(k, N)).
    `indices` contains the flattened indices, and `valid_mask` indicates which ones are valid.
    """
    batch_size = mask.shape[0]
    flat_mask = mask.view(batch_size, -1)

    rand_vals = torch.rand_like(flat_mask.float())
    rand_vals[~flat_mask] = float('inf')

    actual_k = min(k, flat_mask.shape[1])
    vals, flat_indices = torch.topk(rand_vals, actual_k, dim=1, largest=False)

    flat_valid = vals < float('inf')

    flat_indices[~flat_valid] = -1
    return flat_indices, flat_valid


@gin.configurable
class BadRandomPixelChoice(AbstractTakuzuAugmentation):
    """Augmentation which makes k wrong pixel choices."""

    def __init__(self, k_flipped: int) -> None:
        self.k_flipped = k_flipped

    def __call__(
        self, states: torch.Tensor, full_board: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        flat_dim = states.numel() // batch_size

        indices, valid = sample_random_on(states == 0, self.k_flipped)

        flip_mask_flat = torch.zeros(
            batch_size, flat_dim + 1, dtype=torch.bool, device=states.device
        )
        safe_indices = torch.where(valid, indices, flat_dim)
        flip_mask_flat.scatter_(1, safe_indices, True)
        flip_mask = flip_mask_flat[:, :-1].view_as(states)

        positive = states.clone()
        positive[flip_mask] = full_board[flip_mask]
        negative = states.clone()
        negative[flip_mask] = 3 - full_board[flip_mask]
        return positive, negative
