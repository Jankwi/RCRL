from typing import Dict, Tuple, Type

import gin
import torch

from src.contrastivelib.datasets.special_padding import DatasetWithPadding
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.augment import AbstractAugmentation


@gin.configurable
class RelativeContrastiveDataset(DatasetWithPadding):
    """
    Relative Contrastive learning.

    When pad_special is True, each trajectory is padded with a single special "solved" token
    right after its last real state, so the goal can land on the learned special embedding while
    the positive and negatives always stay on real states.
    """

    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        aug_class: Type[AbstractAugmentation],
        gamma: float = 0.9,
        negs_per_pos: int = 1,
        max_prev: int = 1,
        spec_mgr: SpecialStateManager = None,
        geom_type: str = 'bounded',
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=batch_size,
            spec_mgr=spec_mgr,
            pad_special=spec_mgr is not None,
        )
        self.device = device
        self.k_negatives = negs_per_pos
        self.max_prev = max_prev
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)
        self.aug = aug_class()
        self.geom_type = geom_type

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        lengths = lengths.to(torch.long)
        batch_idx = torch.arange(trajectories.shape[0], device=self.device)

        trajectories, sample_lengths = self._pad_special(trajectories, lengths)

        # Bump the length by one (when padding) only for sampling t0/t1, so the goal can reach the
        # appended special token at index `lengths`. The augmentation below keeps the real
        # `lengths`, since it reads the true solved board at `lengths - 1`; the bumped value would
        # point it at the special token instead.
        t0, t1 = self._get_t01(sample_lengths - 1)
        t0, t1 = t0 + 1, t1 + 1
        pos_state = trajectories[batch_idx, t0]
        goal = trajectories[batch_idx, t1]

        prev_range = torch.minimum(t0, torch.tensor(self.max_prev, device=self.device))
        # Offset is at least 1 so t_prev < t0 and temporal_dist >= 1; a zero offset would make
        # the augmentation corrupt no cells, yielding a negative identical to pos_state.
        t_prev = t0 - (self.sampler.bounded_uniform(prev_range) + 1)

        # Generate k independent negatives per positive, then flatten to (n * k, ...) so the model
        # can embed them as a single batch; the loss reshapes back to (n, k, ...).
        negs = [
            self.aug(
                trajectories=trajectories,
                lengths=lengths,
                pivot_idx=t_prev,
                temporal_dist=t0 - t_prev,
                traj_positive=False,
                custom_positive=False,
            )
            for _ in range(self.k_negatives)
        ]
        neg_states = torch.stack(negs, dim=1).reshape(-1, *trajectories.shape[2:])

        return (
            {'pos_state': pos_state, 'neg_states': neg_states, 'goal': goal},
            {},
        )
