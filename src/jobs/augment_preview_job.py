import os
from typing import Dict, Tuple, Type

import gin
import matplotlib.pyplot as plt
import torch

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import VectorizedEnvironment
from src.envs.augment import AbstractAugmentation
from src.envs.takuzu import augment as _takuzu_augments  # noqa: F401  registers gin augments
from src.jobs import AbstractJob


class _AugmentPreviewDataset(AbstractContrastiveDataset):
    """Pulls a pivot state and its solved board from each trajectory and produces a positive
    (temporal_dist steps closer to solved) and a negative (corrupted) via the augmentation."""

    def __init__(
        self,
        file_sets_path: str,
        device: torch.device,
        chunk_size: int,
        aug: AbstractAugmentation,
        gamma: float = 0.9,
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=chunk_size,
        )
        self.aug = aug
        self.sampler = DistributionSampler(device=device, gamma=gamma)

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        lengths = lengths.to(torch.long)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        t_pivot = self.sampler.bounded_uniform(lengths - 1)
        if self.aug.default_temporal_dist is None:
            temporal_dist = self.sampler.bounded_geometric(lengths - 1 - t_pivot)
        else:
            temporal_dist = torch.full_like(t_pivot, self.aug.default_temporal_dist)
            temporal_dist = torch.minimum(temporal_dist, lengths - 1 - t_pivot)

        state = trajectories[batch_idx, t_pivot]
        full = trajectories[batch_idx, lengths - 1]

        append_custom_positive = self.aug.supports_custom_positive()
        ret_states = self.aug(
            trajectories=trajectories,
            lengths=lengths,
            pivot_idx=t_pivot,
            temporal_dist=temporal_dist,
            traj_positive=True,
            custom_positive=append_custom_positive,
        )

        # When the augmentation supports it we request both positives, so __call__
        # returns the nested ((negative, custom_positive), traj_positive).
        custom = {}
        if append_custom_positive:
            (negative, custom_positive), traj_positive = ret_states
            custom['custom_positive'] = custom_positive
        else:
            negative, traj_positive = ret_states

        return (
            {'state': state, 'traj_positive': traj_positive, 'negative': negative, 'full': full}
            | custom,
            {},
        )


@gin.configurable
class AugmentPreviewJob(AbstractJob):
    """One-shot job that renders augmentation samples to disk for visual inspection."""

    def __init__(
        self,
        dataset_path: str,
        env_class: Type[VectorizedEnvironment],
        aug: AbstractAugmentation,
        n_renders: int,
        save_dir: str = '/augment-previews',
        gamma: float = 0.9,
        spec_mgr_param: Dict[str, SpecialStateManager] = {},
        gin_config: Dict[str, str] = None,
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_renders = n_renders
        self.save_dir = dataset_path + save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.env = env_class(device=self.device, **spec_mgr_param)
        self.dataset = _AugmentPreviewDataset(
            file_sets_path=dataset_path,
            device=self.device,
            chunk_size=n_renders,
            aug=aug,
            gamma=gamma,
        )

        print('-----Initalized AugmentPreviewJob------')
        for name, value in (gin_config or {}).items():
            print(f'  {name}: {value}')
        print('---------------------------------------')

    def execute(self) -> None:
        states_td, _ = next(iter(self.dataset))
        for i in range(self.n_renders):
            sample_dir = os.path.join(self.save_dir, f'sample-{i}')
            os.makedirs(sample_dir, exist_ok=True)
            for key in states_td.keys():
                fig = self.env.render_state(states_td[key][i].cpu().numpy())
                fig.savefig(
                    os.path.join(sample_dir, f'{key}.png'),
                    bbox_inches='tight',
                    pad_inches=0.1,
                )
                plt.close(fig)
        print(f'Rendered {self.n_renders} augmentation samples to {self.save_dir}')
