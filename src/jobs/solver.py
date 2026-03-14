# ruff: noqa: F401
from typing import Dict, Type

import gin
import torch

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.maze.quarter import VectorizedQuarterMaze


@gin.configurable
class Solver:
    def __init__(
        self,
        games: int,
        dataset: AbstractContrastiveDataset,
        device: torch.device,
        ve: ValueEstimator,
        env_class: Type[VectorizedEnvironment],
        spec_mgr: SpecialStateManager = None,
    ) -> None:
        self.games = games
        self.dataset = dataset
        self.spec_mgr = spec_mgr
        self.device = device

        self.env = env_class(device=device, spec_mgr=spec_mgr)
        self.ve = ve

    def solve(self) -> Dict[str, float]:
        starting = self.dataset.get_random_starting_positions(k=self.games)
        return eval_no_search(starting, self.env, self.ve)
