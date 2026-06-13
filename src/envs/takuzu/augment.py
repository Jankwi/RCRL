from abc import ABC, abstractmethod
from typing import Tuple

import gin
import torch

from src.envs.utils import sample_random_on


class AbstractTakuzuAugmentation(ABC):
    @abstractmethod
    def __call__(
        self, states: torch.Tensor, full_board: torch.Tensor, specific_k: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (positive, negative) Takuzu boards."""
        pass


@gin.configurable
class BadRandomPixelChoice(AbstractTakuzuAugmentation):
    """Augmentation which makes k wrong pixel choices."""

    def __init__(self, k_flipped: int = None) -> None:
        self.k_flipped = k_flipped

    def __call__(
        self, states: torch.Tensor, full_board: torch.Tensor, specific_k: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        flat_dim = states.numel() // batch_size

        max_k = torch.amax(specific_k).item() if specific_k is not None else self.k_flipped
        assert max_k is not None, 'Either k_flipped or specific_k must be provided.'

        indices, valid = sample_random_on(states == 0, max_k)

        if specific_k is not None:
            pos_range = torch.arange(max_k, device=states.device).unsqueeze(0)
            valid = valid & (pos_range < specific_k.unsqueeze(1))

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
