from typing import Tuple

import gin
import torch
from tensordict import TensorDict

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler


@gin.configurable
class ContrastiveDatasetNoSpecial(AbstractContrastiveDataset):
    """
    Plain CRTR - doesn't use special states.
    """

    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        gamma: float = 0.9,
        repetition_factor: int = 1,
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=int(batch_size / repetition_factor),
        )
        self.device = device
        self.repetition_factor = repetition_factor
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[TensorDict, TensorDict]:
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths.to(torch.long), self.repetition_factor, dim=0)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        t0 = self.sampler.bounded_uniform(lengths - 1)
        t1 = t0 + self.sampler.clamped_geometric(lengths - 1 - t0)
        return (
            TensorDict({'state': trajectories[batch_idx, t0], 'goal': trajectories[batch_idx, t1]}),
            TensorDict({}),
        )
