import os
from typing import Dict, Tuple, Type

import gin
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.utils.distributions import DistributionSampler
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import VectorizedEnvironment
from src.envs.takuzu.augment import AbstractTakuzuAugmentation
from src.jobs import AbstractJob


class _AugmentPreviewDataset(AbstractContrastiveDataset):
    """Dataset that pulls (state, positive_future, full_board) triples and produces a negative
    by applying the given augmentation."""

    def __init__(
        self,
        file_sets_path: str,
        device: torch.device,
        chunk_size: int,
        aug: AbstractTakuzuAugmentation,
    ) -> None:
        super().__init__(
            file_sets_path=file_sets_path,
            tensor_names=['lengths'],
            device=device,
            chunk_size=chunk_size,
        )
        self.aug = aug
        self.sampler = DistributionSampler(device=device)

    def contrastive_pick(
        self, trajectories: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[TensorDict, TensorDict]:
        lengths = lengths.to(torch.long)
        batch_idx = torch.arange(len(trajectories), device=self.device)

        t0 = self.sampler.bounded_uniform(lengths - 1)
        t_full = lengths - 1

        state = trajectories[batch_idx, t0]
        full = trajectories[batch_idx, t_full]
        positive, negative = self.aug(state, full)

        batch_size = state.shape[0]
        return (
            TensorDict(
                {'state': state, 'positive': positive, 'negative': negative, 'full': full},
                batch_size=batch_size,
            ),
            TensorDict({}, batch_size=batch_size),
        )


@gin.configurable
class AugmentPreviewJob(AbstractJob):
    """One-shot job that renders augmentation samples to disk for visual inspection."""

    def __init__(
        self,
        dataset_path: str,
        env_class: Type[VectorizedEnvironment],
        aug: AbstractTakuzuAugmentation,
        n_renders: int,
        save_dir: str = '/augment-previews',
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
            for key in ('state', 'positive', 'negative', 'full'):
                fig = self.env.render_state(states_td[key][i].cpu().numpy())
                fig.savefig(
                    os.path.join(sample_dir, f'{key}.png'),
                    bbox_inches='tight',
                    pad_inches=0.1,
                )
                plt.close(fig)
        print(f'Rendered {self.n_renders} augmentation samples to {self.save_dir}')
