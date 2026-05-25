# ruff: noqa: F401
import os
from typing import Dict, Type

import gin
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.metrics.losses import AbstractContrastiveLoss, MetricAccumulator
from src.contrastivelib.metrics.sims import SimilarityMeasure
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.maze.colorful import VectorizedColorfulMaze
from src.envs.maze.quarter import VectorizedQuarterMaze
from src.envs.maze.target import VectorizedTargetMaze
from src.envs.visual import get_embed_plots, get_trajectory_gifs
from src.jobs.solved_rate_job import SolvedRateJob
from src.jobs.utils.logging import (
    AbstractLogger,
    BestValOsLogger,
    MultipleLoggers,
    StdOutLogger,
    WandbLogger,
    metric_action,
)


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
        solved_rate_job_class: Type[SolvedRateJob] = SolvedRateJob,
        gin_config: Dict[str, str] = None,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_models: str = 'dont',  # one of 'dont', 'local', 'log'
        n_gifs: int = 0,
        gif_max_len: int = None,
        n_embed_trajs: int = 0,
        embed_plot_max_len: int = 20,
        append_loss_metrics: bool = True,
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

        self.checkpoint_models = checkpoint_models
        self.n_gifs, self.gif_max_len = n_gifs, gif_max_len
        self.n_embed_trajs, self.embed_plot_max_len = n_embed_trajs, embed_plot_max_len
        self.append_loss_metrics = append_loss_metrics
        self.epoch_dir = ''

        self.solved_rate_job = solved_rate_job_class(
            device=self.device,
            env=self.env,
            ve=self.ve,
            print_solved_rates=False,
            dataset=self.validset,
        )

    def _get_epoch_dir(self, epoch: int) -> str:
        epoch_dir = f'{self.checkpoint_dir}/{epoch}/'
        if self.epoch_dir != epoch_dir:
            os.makedirs(epoch_dir, exist_ok=True)
        return epoch_dir

    @metric_action('valid_loss')
    def measure_valid_loss(self, epoch: int) -> Dict[str, float]:
        """Completes a full sweep of the validation data set, and computes the loss function"""
        acc = MetricAccumulator()

        for states, extras in self.validset:
            states_repr = states.apply(self.model)
            loss, metrics = self.loss_fn(
                **states_repr,
                **extras,
                append_metrics=self.append_loss_metrics,
                append_empty=True,
            )
            acc.update({'loss': loss.item()} | metrics)

        return acc.average(prefix='valid')

    @metric_action('solved_rate')
    def measure_solved_rate(self, epoch: int) -> Dict[str, float]:
        """
        Simulates solving attempts (no search) with the current model.
        Returns the solved rates for starting configurations,
        specified when constructing the SolvedRateJob instance.
        """
        return self.solved_rate_job.execute()

    @metric_action('state_dict', measure_time=False)
    def save_checkpoint(self, epoch: int) -> None:
        """Saves model & optimizer states locally, and then passes them to the logger"""
        dir = self._get_epoch_dir(epoch)
        if self.checkpoint_models == 'local' or self.checkpoint_models == 'log':
            model_path, optim_path = f'{dir}/model.pt', f'{dir}/optimizer.pt'
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            if self.checkpoint_models == 'log':
                self.logger.log_state_dicts(epoch, {'model': model_path, 'optimizer': optim_path})

    @metric_action('gif')
    def create_trajectory_gifs(self, epoch: int) -> None:
        """Creates gifs, with subsequent frames depicting subsequent trajectory states"""
        if self.n_gifs == 0:
            return

        starting = self.validset.get_random_starting_positions(k=self.n_gifs)
        materials = get_trajectory_gifs(starting, self.env, self.ve, self.gif_max_len)
        dir = f'{self._get_epoch_dir(epoch)}/gifs'
        os.makedirs(dir, exist_ok=True)

        gifs_dict: Dict[str, str] = {}
        for i, frames in enumerate(materials):
            filename = f'{dir}/trajectory-{i}.gif'
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
            gifs_dict[f'trajectory-{i}-gif'] = filename
        self.logger.log_gifs(epoch, gifs_dict)

    @metric_action('plot_embeds')
    def plot_trajectory_embeddings(self, epoch: int) -> None:
        """Creates and stores a t-SNE projection plot for trajectory-end embeddings."""
        if self.n_embed_trajs == 0:
            return

        tensors, labels = self.validset.get_random_trajectory_ends(
            self.n_embed_trajs,
            self.embed_plot_max_len,
        )
        embeds = self.model(tensors)

        fig = get_embed_plots(embeds, labels)
        filename = f'{self._get_epoch_dir(epoch)}/trajectory-embeddings.png'
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        self.logger.log_image(epoch, {'trajectory-embeddings': filename}, format='png')

    def execute(self, epoch: int, train_metrics: Dict[str, float] = {}) -> None:
        if not self.logger.is_log_epoch(epoch):
            return

        self.model.eval()
        self.save_checkpoint(epoch)

        with self.validset.preload_file_set():
            self.logger.log_metrics(
                epoch,
                (
                    train_metrics
                    | self.create_trajectory_gifs(epoch)
                    | self.plot_trajectory_embeddings(epoch)
                    | self.measure_valid_loss(epoch)
                    | self.measure_solved_rate(epoch)
                ),
            )

        self.model.train()

    def finish_run(self) -> None:
        self.logger.finish_run()
