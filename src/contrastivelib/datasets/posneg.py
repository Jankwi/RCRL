import random
from typing import Dict, Tuple, Type

import gin
import torch

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.takuzu.augment import AbstractTakuzuAugmentation


@gin.configurable
class PositiveNegativeDataset(AbstractContrastiveDataset):
    """
    Yields pairs of positive (solvable) and (negative) Takuzu Boards.
    If split is False, the positive and negative samples are generated from the same state.
    """

    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        aug_class: Type[AbstractTakuzuAugmentation],
        spec_mgr: SpecialStateManager = None,
        gamma: float = 0.9,
        repetition_factor: int = 1,
        split: bool = False,
        same_t0: bool = False,
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=int(batch_size / repetition_factor),
        )
        self.aug = aug_class()
        self.device = device
        self.repetition_factor = repetition_factor
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)
        self.split = split
        self.same_t0 = same_t0

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths.to(torch.long), self.repetition_factor, dim=0)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        if self.same_t0:
            min_length = lengths.min().item()
            index_from_back = random.randint(1, min_length)
            t0 = (lengths - 1) - index_from_back
        else:
            t0 = self.sampler.bounded_uniform(lengths - 1)

        state = trajectories[batch_idx, t0]
        full = trajectories[batch_idx, lengths - 1]
        positive, negative = self.aug(state, full)

        batch_size = len(positive)
        if self.split:
            batch_size //= 2
            positive = positive[:batch_size]
            negative = negative[batch_size:]

        return (
            {'positive': positive, 'negative': negative},
            {},
        )
