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
        negs_per_pos: int = 1,
        independent_negatives: bool = False,
        same_t_pivot: bool = False,
        append_special: bool = True,
        custom_positive: bool = False,
    ) -> None:
        assert default_temporal_dist is not None or gamma is not None, (
            'Set gamma when default_temporal_dist is None (geometric temporal_dist needs it).'
        )
        assert negs_per_pos >= 1, 'negs_per_pos must be at least 1 (negatives per positive).'
        # batch_size is the number of positives (independent of negs_per_pos); the batch then
        # holds batch_size positives and batch_size * negs_per_pos negatives.
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=batch_size,
        )

        self.default_temporal_dist = default_temporal_dist
        self.aug = aug_class(default_temporal_dist=self.default_temporal_dist)

        self.device = device
        self.k_negatives = negs_per_pos
        self.independent_negatives = independent_negatives
        self.sampler = DistributionSampler(device=self.device, gamma=gamma)

        self.same_t_pivot = same_t_pivot
        self.append_special = append_special and spec_mgr is not None
        self.spec_mgr = spec_mgr

        self.custom_positive = custom_positive
        self.traj_positive = not custom_positive

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        n = trajectories.shape[0]
        # We over-generate matched (positive, negative) pairs by tiling the state, then keep
        # `n` positives and `n * k_negatives` negatives. With independent (custom) negatives the
        # extra tile makes the kept negatives disjoint from the kept positives; otherwise the kept
        # negatives include tile 0, which is matched to the positives.
        expand = self.k_negatives + int(self.independent_negatives and self.custom_positive)
        tile_shape = (expand,) + (1,) * (trajectories.dim() - 1)
        trajectories = trajectories.tile(tile_shape)
        lengths = lengths.to(torch.long).tile(expand)

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

        # Keep `n` positives (first tile) and the last `n * k_negatives` negatives. The negatives
        # are laid out as `k_negatives` tiles of `n`, so the loss reads them as (k_negatives, n).
        positive = positive[:n]
        negative = negative[-n * self.k_negatives :]

        ret_states = {'positive': positive, 'negative': negative}
        if self.append_special:
            special = self.spec_mgr.create_many(1)
            ret_states['special'] = special

        return (
            ret_states,
            {},
        )
