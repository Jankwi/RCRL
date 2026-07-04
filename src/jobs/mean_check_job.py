# ruff: noqa: F401
from typing import Dict, Type

import gin
import torch
import torch.nn as nn

from src.contrastivelib.datasets import (
    AbstractContrastiveDataset,
    ContrastiveDatasetNoSpecial,
)
from src.contrastivelib.metrics.estimation import SingleGoalValueEstimator
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import Embedding2D, LNConvNet
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.maze.center import VectorizedCenterMaze
from src.jobs import AbstractJob
from src.jobs.solved_rate_job import SolvedRateJob

_TRAIN_STR, _VALID_STR = 'train', 'valid'


@gin.configurable
class MeanCompassCheckJob(AbstractJob):
    """
    Heuristic baseline: turn an explicitly-trained model into a special-token model
    without any special training. Loads a base-model checkpoint into a
    SpecialStateModelWrapper, sets the special embedding to the mean embedding of a
    batch of solved states ("answers") from the train set, then measures the solved
    rate on the valid set.
    """

    def __init__(
        self,
        embed_dim: int,
        metric_name: str,
        normalize: bool,
        base_model_class: Type[nn.Module],
        dataset_class: Type[AbstractContrastiveDataset],
        spec_mgr_class: Type[SpecialStateManager],
        env_class: Type[VectorizedEnvironment],
        ve_class: Type[ValueEstimator],
        base_checkpoint_path: str,
        dataset_dir: str,
        n_answers: int = 256,
        spec_model_wrapper_class: Type[nn.Module] = SpecialStateModelWrapper,
        solved_rate_job_class: Type[SolvedRateJob] = SolvedRateJob,
        envs_common: Dict = {},
        gin_config: Dict[str, str] = None,
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sim_fn = get_similarity_measure(measure=metric_name, normalize=normalize)
        spec_mgr = spec_mgr_class(device=self.device)

        base_model = base_model_class(embed_dim=embed_dim)
        self.model = spec_model_wrapper_class(
            base_model=base_model,
            spec_mgr=spec_mgr,
            embed_dim=embed_dim,
            base_checkpoint_path=base_checkpoint_path,
        ).to(self.device)
        self.model.eval()

        self.trainset = dataset_class(
            file_sets_path=f'{dataset_dir}/{_TRAIN_STR}', device=self.device
        )
        self.validset = dataset_class(
            file_sets_path=f'{dataset_dir}/{_VALID_STR}', device=self.device
        )

        ve = ve_class(model=self.model, sim_fn=sim_fn, device=self.device)
        env = env_class(device=self.device, spec_mgr=spec_mgr, **envs_common)
        self.solved_rate_job = solved_rate_job_class(
            device=self.device, env=env, ve=ve, dataset=self.validset, print_stats=False
        )

        self.n_answers = n_answers

    def execute(self) -> None:
        """Sets the special embedding to the mean answer embedding, then evaluates."""
        with torch.no_grad():
            with self.trainset.preload_file_set():
                _, answers = self.trainset.get_random_starting_positions(
                    k=self.n_answers, append_answers=True
                )
            mean_embed = self.model.base_model(answers).mean(dim=0)
            self.model.special_embed.copy_(mean_embed)

            results = self.solved_rate_job.execute()

        print('Solved rates with mean compass:')
        for k, v in results.items():
            print(f'{k}: {v:.4f}')
