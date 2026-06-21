from typing import Dict, Tuple

import gin
import torch

from src.contrastivelib.datasets.special_padding import DatasetWithPadding
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager


@gin.configurable
class MixedDataset(DatasetWithPadding):
    """
    Mixes solved trajectories with trajectories that share the same context but diverge from
    the positive at some point. The bad-trajectory addition replaces the CRTR repetition.

    When a spec_mgr is supplied, the positive (good) trajectories are padded with a single special
    "solved" token, so a positive's goal can land on the learned special embedding. The haywire
    trajectories never solve, so they are left unpadded.

    With different_lengths=True the bad trajectories carry their own `algotobad_lengths` (e.g. maze
    walks that run until they hit a wall); otherwise they share the good trajectories' lengths.
    """

    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        spec_mgr: SpecialStateManager = None,
        gamma: float = 0.9,
        geom_type: str = 'bounded',
        different_lengths: bool = False,
    ) -> None:
        tensor_names = ['lengths', 'algotobad']
        if different_lengths:
            tensor_names.append('algotobad_lengths')
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=tensor_names,
            device=device,
            # The good/bad concat in contrastive_pick doubles the chunk back to batch_size.
            chunk_size=batch_size // 2,
            spec_mgr=spec_mgr,
            pad_special=spec_mgr is not None,
        )
        self.device = device
        self.different_lengths = different_lengths
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)
        self.geom_type = geom_type

    def contrastive_pick(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        algotobad: torch.Tensor,
        algotobad_lengths: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        lengths = lengths.to(torch.long)
        # Bad trajectories use their own lengths when configured; else they share the good lengths.
        bad_lengths = lengths if algotobad_lengths is None else algotobad_lengths.to(torch.long)
        n_good = len(trajectories)

        trajectories = torch.cat([trajectories, algotobad])
        lengths = torch.cat([lengths, bad_lengths])
        # Only the good (first) half are solved trajectories that earn the special "solved" token.
        which_solved = torch.arange(len(trajectories), device=self.device) < n_good

        trajectories, lengths = self._pad_special(trajectories, lengths, which_solved=which_solved)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        t0, t1 = self._get_t01(lengths)
        return (
            {'state': trajectories[batch_idx, t0], 'goal': trajectories[batch_idx, t1]},
            {},
        )
