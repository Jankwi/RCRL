# ruff: noqa: F401
from time import perf_counter
from typing import Dict, Type

import gin
import torch
import torch.nn as nn

from src.contrastivelib.datasets import AbstractContrastiveDataset, PassthroughDataset
from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.metrics.losses import AbstractContrastiveLoss
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import BroNet, ConvNet, Embedding2D, LNConvNet, LNDenseNet
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.maze.colorful import VectorizedColorfulMaze
from src.envs.maze.quarter import VectorizedQuarterMaze
from src.envs.maze.target import VectorizedTargetMaze
from src.jobs import AbstractJob
from src.jobs.solved_rate_job import SolvedRateJob


@gin.configurable
class TimingJob(AbstractJob):
    """
    A job for timing evaluation time of a given (random) model on a vectorized environment.
    """

    def __init__(
        self,
        embed_dim: int,
        metric_name: str,
        normalize: bool,
        dataset_path: str,
        n_starting: int,
        env_class: Type[VectorizedEnvironment],
        base_model_class: Type[nn.Module],
        dataset_class: Type[AbstractContrastiveDataset] = PassthroughDataset,
        spec_mgr_class: Type[SpecialStateManager] = None,
        spec_model_wrapper_class: Type[nn.Module] = SpecialStateModelWrapper,
        ve_class: Type[ValueEstimator] = ValueEstimator,
        repeats: int = 5,
        gin_config: Dict[str, str] = None,
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sim_fn = get_similarity_measure(measure=metric_name, normalize=normalize)
        spec_mgr = spec_mgr_class(device=self.device) if spec_mgr_class is not None else None

        base_model = base_model_class(embed_dim=embed_dim)
        self.model = (
            spec_model_wrapper_class(base_model=base_model, spec_mgr=spec_mgr, embed_dim=embed_dim)
            if spec_mgr is not None
            else base_model
        ).to(self.device)

        self.dataset = dataset_class(
            file_sets_path=dataset_path, device=self.device, spec_mgr=spec_mgr
        )

        self.env = env_class(device=self.device, spec_mgr=spec_mgr)
        self.ve = ve_class(model=self.model, sim_fn=sim_fn, device=self.device)
        self.srj = SolvedRateJob(
            n_boards=n_starting,
            env=self.env,
            ve=self.ve,
            dataset=self.dataset,
            print_solved_rates=False,
            gin_config=None,
        )
        self.repeats = repeats

        print('-----Initalized TimingJob-----------')
        for name, value in gin_config.items():
            print(f'  {name}: {value}')
        print('---------------------------------------')

    def execute(self) -> None:
        """Evaluates the model on the environment"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.eval()
        n_starting = self.srj.n_boards
        start_time = perf_counter()
        with torch.inference_mode():
            for _ in range(self.repeats):
                self.srj.execute()
        end_time = perf_counter()
        print(f'Evaluation of {self.repeats} runs for {n_starting} states finished')
        print(f'Total eval time: {end_time - start_time:.4f} seconds')
        print(
            f'Avg eval time for {n_starting} states: {(end_time - start_time) / self.repeats:.4f} s'
        )
        print(
            f'Avg eval time per state: '
            f'{(end_time - start_time) / (self.repeats * n_starting):.6f} s'
        )
