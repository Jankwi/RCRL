from abc import abstractmethod
from typing import Tuple

import gin
import torch
from tensordict import TensorDict

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager


class _AbstractPadded(AbstractContrastiveDataset):
    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        spec_mgr: SpecialStateManager,
        gamma: float = 0.9,
        repetition_factor: int = 1,
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=int(batch_size / repetition_factor),
            spec_mgr=spec_mgr,
        )
        self.device = device
        self.repetition_factor = repetition_factor
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)

    @abstractmethod
    def _delta_from_t0(self, t0: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pass

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[TensorDict, TensorDict]:
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths.to(torch.long), self.repetition_factor, dim=0)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        spec_states = self.spec_mgr.create_many(len(trajectories)).to(trajectories.dtype)
        trajectories = torch.cat((trajectories, spec_states.unsqueeze(1)), dim=1)
        trajectories[batch_idx, lengths] = spec_states

        t0 = self.sampler.bounded_uniform(lengths)
        t1 = t0 + self._delta_from_t0(t0, lengths)
        return (
            TensorDict({'state': trajectories[batch_idx, t0], 'goal': trajectories[batch_idx, t1]}),
            TensorDict({}),
        )


@gin.configurable
class ContrastiveDatasetSpecialPadding(_AbstractPadded):
    """
    Uses CRTR, but instead of rescaling the geometric distribution,
    maps the out of range indices to the special states.

    In other words, pads each trajectory with an infinite number of specialn states.
    """

    def _delta_from_t0(self, t0: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.sampler.clamped_geometric(lengths - t0)


@gin.configurable
class ContrastiveDatasetSingleSpecial(_AbstractPadded):
    """
    Uses CRTR, appending a single special state at the end of every trajectory.
    """

    def _delta_from_t0(self, t0: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.sampler.bounded_geometric(lengths - t0)
