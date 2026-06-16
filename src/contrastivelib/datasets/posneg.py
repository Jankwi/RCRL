import random
from typing import Dict, Tuple, Type

import gin
import torch

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs.augment import AbstractAugmentation


@gin.configurable
class PositiveNegativeDataset(AbstractContrastiveDataset):
    """
    Yields pairs of positive (closer-to-solved) and negative (corrupted) Takuzu Boards,
    both drawn relative to a pivot state along each trajectory.
    """

    def __init__(
        self,
        file_sets_path: str,
        batch_size: int,
        device: torch.device,
        aug_class: Type[AbstractAugmentation],
        default_temporal_dist: int = None,
        spec_mgr: SpecialStateManager = None,
        gamma: float = None,
        repetition_factor: int = 1,
        same_t_pivot: bool = False,
        append_special: bool = True,
        custom_positive: bool = False,
    ) -> None:
        assert default_temporal_dist is not None or gamma is not None, (
            'Set gamma when default_temporal_dist is None (geometric temporal_dist needs it).'
        )
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=int(batch_size / repetition_factor),
        )

        self.default_temporal_dist = default_temporal_dist
        self.aug = aug_class(default_temporal_dist=self.default_temporal_dist)

        self.device = device
        self.repetition_factor = repetition_factor
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)

        self.same_t_pivot = same_t_pivot
        self.append_special = append_special and spec_mgr is not None
        self.spec_mgr = spec_mgr

        self.custom_positive = custom_positive
        self.traj_positive = not custom_positive

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        trajectories = torch.repeat_interleave(trajectories, self.repetition_factor, dim=0)
        lengths = torch.repeat_interleave(lengths.to(torch.long), self.repetition_factor, dim=0)

        if self.same_t_pivot:
            min_length = lengths.min().item()
            index_from_back = random.randint(1, min_length)
            t_pivot = (lengths - 1) - index_from_back
        else:
            t_pivot = self.sampler.bounded_uniform(lengths - 1)

        if self.default_temporal_dist is None:
            temporal_dist = self.sampler.bounded_geometric(lengths - 1 - t_pivot)
        else:
            temporal_dist = torch.full_like(t_pivot, self.default_temporal_dist)
            temporal_dist = torch.minimum(temporal_dist, lengths - 1 - t_pivot)

        negative, positive = self.aug(
            trajectories=trajectories,
            lengths=lengths,
            pivot_idx=t_pivot,
            temporal_dist=temporal_dist,
            traj_positive=self.traj_positive,
            custom_positive=self.custom_positive,
        )

        ret_states = {'positive': positive, 'negative': negative}
        if self.append_special:
            special = self.spec_mgr.create_many(1)
            ret_states['special'] = special

        return (
            ret_states,
            {},
        )
