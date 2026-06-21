from abc import abstractmethod
from typing import Dict, List, Tuple

import gin
import torch

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager


class DatasetWithPadding(AbstractContrastiveDataset):
    def __init__(
        self,
        file_sets_path: str,
        tensor_names: List[str],
        device: torch.device,
        chunk_size: int,
        spec_mgr: SpecialStateManager = None,
        pad_special: bool = False,
        master_tn: str = 'trajectories',
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=tensor_names,
            device=device,
            chunk_size=chunk_size,
            master_tn=master_tn,
            spec_mgr=spec_mgr,
        )
        self.pad_special = pad_special

    def _pad_special(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        which_solved: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends one special state per trajectory when pad_special is True.

        The special token is written at index `lengths` and the length bumped by one, but only for
        rows flagged in `which_solved` (default: all rows). Unflagged rows still get the appended
        column (so the time axis stays uniform) but keep their length, so they never index it.
        """
        if not self.pad_special:
            return trajectories, lengths
        if which_solved is None:
            which_solved = torch.ones(len(trajectories), dtype=torch.bool, device=self.device)
        batch_idx = torch.arange(len(trajectories), device=self.device)
        spec_states = self.spec_mgr.create_many(len(trajectories)).to(trajectories.dtype)
        trajectories = torch.cat((trajectories, spec_states.unsqueeze(1)), dim=1)
        trajectories[batch_idx[which_solved], lengths[which_solved]] = spec_states[which_solved]
        return trajectories, lengths + which_solved.to(lengths.dtype)

    @abstractmethod
    def contrastive_pick(
        self, **kwargs: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pass


@gin.configurable
class PaddedCRTRDataset(DatasetWithPadding):
    """
    CRTR dataset that pads each trajectory with a single special state.

    The geom_type controls how out-of-range geometric samples are handled:
      'clamped'  — samples pile up at the boundary (infinite-padding semantics)
      'bounded'  — samples wrap via modulo (single-special-state semantics)
    """

    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        spec_mgr: SpecialStateManager,
        gamma: float = 0.9,
        repetition_factor: int = 1,
        geom_type: str = 'clamped',
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=int(batch_size / repetition_factor),
            spec_mgr=spec_mgr,
            pad_special=True,
        )
        self.device = device
        self.repetition_factor = repetition_factor
        self.geom_type = geom_type
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths.to(torch.long), self.repetition_factor, dim=0)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        trajectories, sample_lengths = self._pad_special(trajectories, lengths)

        t0, t1 = self._get_t01(sample_lengths)
        return (
            {'state': trajectories[batch_idx, t0], 'goal': trajectories[batch_idx, t1]},
            {},
        )
