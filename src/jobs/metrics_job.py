# ruff: noqa: F401
import os
from typing import Dict, Type

import gin
import torch
import torch.nn as nn

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.metrics.losses import AbstractContrastiveLoss
from src.contrastivelib.metrics.sims import SimilarityMeasure
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.maze.quarter import VectorizedQuarterMaze
from src.envs.visual import get_trajectory_gifs
from src.jobs.utils.logging import AbstractLogger, MultipleLoggers, StdOutLogger, WandbLogger


@gin.configurable
class MetricsJob:
    """Job for evaluating the model, and saving checkpoints and logging"""

    def __init__(
        self,
        epochs: int,
        model: nn.Module,
        optimizer: nn.Module,
        validset_path: str,
        validset_class: Type[AbstractContrastiveDataset],
        loss_fn: AbstractContrastiveLoss,
        device: torch.device,
        ve_class: Type[ValueEstimator],
        env_class: Type[VectorizedEnvironment],
        logger_class: Type[AbstractLogger],
        sim_fn: SimilarityMeasure,
        spec_mgr_param: Dict[str, SpecialStateManager] = {},
        gin_config: Dict[str, str] = None,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_models: str = 'dont', # one of 'dont', 'local', 'log'
        solve_attempts: int = 1024,
        n_gifs: int = 0,
        gif_max_len: int = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = logger_class(epochs=epochs, run_params=gin_config)

        self.model, self.optimizer = model, optimizer
        self.device = device

        self.loss_fn = loss_fn
        self.validset = validset_class(
            file_sets_path=validset_path, device=self.device, **spec_mgr_param
        )

        self.env = env_class(device=device, **spec_mgr_param)
        self.ve = ve_class(model=self.model, sim_fn=sim_fn, device=self.device)

        self.solve_attempts = solve_attempts
        self.checkpoint_models = checkpoint_models
        self.n_gifs, self.gif_max_len = n_gifs, gif_max_len

    def measure_valid_loss(self) -> Dict[str, float]:
        """Completes a full sweep of the validation data set, and computes the loss function"""
        valid_loss = torch.tensor([0.0], device=self.device)

        for states, extras in self.validset:
            states_repr = states.apply(self.model)
            loss = self.loss_fn(**states_repr, **extras)
            valid_loss += loss

        return {'valid_loss': valid_loss.cpu().item() / self.validset.get_batches_per_epoch()}

    def measure_solved_rate(self) -> Dict[str, float]:
        """Simulates solving attempts (no search) with the current model. Returns the solved rate"""
        starting = self.validset.get_random_starting_positions(k=self.solve_attempts)
        return eval_no_search(starting, self.env, self.ve)

    def save_checkpoint(self, epoch: int, dir: str) -> None:
        """Saves model & optimizer states locally, and then passes them to the logger"""
        if self.checkpoint_models == 'local' or self.checkpoint_models == 'log':
            model_path, optim_path = f'{dir}/model.pt', f'{dir}/optimizer.pt'
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            if self.checkpoint_models == 'log':
                self.logger.log_state_dicts(epoch, {'model': model_path, 'optimizer': optim_path})

    def create_trajectory_gifs(self, epoch: int, dir: str) -> None:
        """Creates gifs, with subsequent frames depicting subsequent trajectory states"""
        if self.n_gifs == 0:
            return

        starting = self.validset.get_random_starting_positions(k=self.n_gifs)
        materials = get_trajectory_gifs(starting, self.env, self.ve, self.gif_max_len)
        dir = f'{dir}/gifs'
        os.makedirs(dir, exist_ok=True)

        gifs_dict: Dict[str, str] = {}
        for i, frames in enumerate(materials):
            filename = f'{dir}/trajectory-{i}.gif'
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
            gifs_dict[f'trajectory-{i}-gif'] = filename

    def execute(self, epoch: int, avg_train_loss: float) -> None:
        if not self.logger.is_log_epoch(epoch):
            return

        self.model.eval()

        epoch_dir = f'{self.checkpoint_dir}/{epoch}/'
        os.makedirs(epoch_dir, exist_ok=True)
        self.save_checkpoint(epoch, dir=epoch_dir)
        self.create_trajectory_gifs(epoch, dir=epoch_dir)

        self.logger.log_metrics(
            epoch,
            (
                {'train_loss': avg_train_loss}
                | self.measure_valid_loss()
                | self.measure_solved_rate()
            ),
        )

        self.model.train()

    def finish_run(self) -> None:
        self.logger.finish_run()
