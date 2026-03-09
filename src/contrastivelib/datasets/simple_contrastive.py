import gin
import torch

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager


@gin.configurable
class SimplestContrastiveDataset(AbstractContrastiveDataset):
    """
        Simplest possible contrastive datasets for dealing with special states.
        Uses CRTR, but instead of rescaling the geometric distribution,
        maps the out of range indices to the special states.
    """
    def __init__(self, file_sets_path: str, batch_size: int, device: torch.device,
            spec_manager: SpecialStateManager, gamma: float = 0.9, repetition_factor: int = 1,
        ):
        super().__init__(
            file_sets_path=file_sets_path, tensor_names=['lengths'], device=device,
            chunk_size=int(batch_size/repetition_factor)
        )
        self.device = device
        self.spec_manager = spec_manager
        self.repetition_factor = repetition_factor
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)

    def contrastive_pick(self, trajectories: torch.Tensor, lengths: torch.Tensor):
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths, self.repetition_factor, dim=0)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        spec_states = self.spec_manager.create_many(len(trajectories))
        trajectories = torch.cat((trajectories, spec_states.unsqueeze(1)), dim=1)
        trajectories[batch_idx, lengths] = spec_states

        t0 = self.sampler.bounded_uniform(lengths)
        t1 = t0 + self.sampler.clamped_geometric(lengths - t0)
        return (trajectories[batch_idx, t0], trajectories[batch_idx, t1])
