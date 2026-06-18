from typing import Dict, Tuple

import gin
import torch

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
        geom_type: str = 'bounded',
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
        self.geom_type = geom_type

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths.to(torch.long), self.repetition_factor, dim=0)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        t0, t1 = self._get_t01(lengths)
        return (
            {'state': trajectories[batch_idx, t0], 'goal': trajectories[batch_idx, t1]},
            {},
        )
