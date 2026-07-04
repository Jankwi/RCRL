# ruff: noqa: F401
import os
from typing import Dict, List, Type

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from src.contrastivelib.datasets import AbstractContrastiveDataset
from src.contrastivelib.metrics.estimation import ManyGoalsValueEstimator, SingleGoalValueEstimator
from src.contrastivelib.metrics.losses import AbstractContrastiveLoss, MetricAccumulator
from src.contrastivelib.metrics.sims import SimilarityMeasure
from src.contrastivelib.utils.special import SpecialStateManager
from src.envs import ValueEstimator, VectorizedEnvironment
from src.envs.explore import eval_no_search
from src.envs.maze.arbitrary import ArbitraryPositionMaze
from src.envs.maze.center import VectorizedCenterMaze
from src.envs.maze.explicit import ExplicitGoalMaze
from src.envs.maze.target import VectorizedTargetMaze
from src.envs.takuzu import VectorizedTakuzu
from src.envs.takuzu.arbitrary import ArbitraryGoalTakuzu
from src.envs.takuzu.explicit import ExplicitGoalTakuzu
from src.envs.visual import get_embed_plots
from src.jobs.solved_rate_job import SolvedRateJob
from src.jobs.traj_vis_job import TrajectoryVisualizationJob
from src.jobs.utils import switch_path
from src.jobs.utils.logging import (
    AbstractLogger,
    BestValOsLogger,
    MultipleLoggers,
    StdOutLogger,
    WandbLogger,
    metric_action,
)

_VALID_STR = 'valid'
_TRAIN_STR = 'train'


@gin.configurable
class MetricsJob:
    """Job for evaluating the model, and saving checkpoints and logging"""

    def __init__(
        self,
        epochs: int,
        model: nn.Module,
        optimizer: nn.Module,
        validset_path: str,
        trainset: AbstractContrastiveDataset,
        loss_fn: AbstractContrastiveLoss,
        device: torch.device,
        ve_class: Type[ValueEstimator],
        env_class: Type[VectorizedEnvironment],
        logger_class: Type[AbstractLogger],
        sim_fn: SimilarityMeasure,
        validset_class: Type[AbstractContrastiveDataset] = None,
        default_validset_class: Type[AbstractContrastiveDataset] = None,
        spec_mgr_param: Dict[str, SpecialStateManager] = {},
        solved_rate_job_class: Type[SolvedRateJob] = SolvedRateJob,
        traj_vis_job_class: Type[TrajectoryVisualizationJob] = TrajectoryVisualizationJob,
        gin_config: Dict[str, str] = None,
        starting_epoch: int = 0,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_models: str = 'dont',  # one of 'dont', 'local', 'log'
        n_gifs: int = 0,
        gif_max_len: int = None,
        n_embed_trajs: int = 0,
        embed_plot_max_len: int = 20,
        n_corr_trajs: int = 128,
        append_loss_metrics: bool = True,
        metrics_on_trainset: bool = True,
        envs_common: Dict = {},
        extra_solved_rate_envs: List[Dict] = [],
        primary_sr_params: Dict = {},
        multi_env_aggregation_type: str = None,
        multi_env_aggregation_name: str = None,
    ) -> None:
        validset_path = switch_path(validset_path)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = logger_class(
            epochs=epochs, starting_epoch=starting_epoch, run_params=gin_config
        )

        self.model, self.optimizer = model, optimizer
        self.device = device

        self.metrics_on_trainset = metrics_on_trainset
        if metrics_on_trainset:
            assert trainset is not None, 'trainset must be provided if metrics_on_trainset is True'

        self.loss_fn = loss_fn
        self.datasets: Dict[str, AbstractContrastiveDataset] = {}

        validset_class = validset_class or default_validset_class
        assert validset_class is not None, 'validset class not provided'
        self.datasets[_VALID_STR] = validset_class(
            file_sets_path=validset_path, device=self.device, **spec_mgr_param
        )
        if metrics_on_trainset:
            self.datasets[_TRAIN_STR] = trainset

        self.ve = ve_class(model=self.model, sim_fn=sim_fn, device=self.device)

        self.checkpoint_models = checkpoint_models
        self.n_embed_trajs, self.embed_plot_max_len = n_embed_trajs, embed_plot_max_len
        self.n_corr_trajs = n_corr_trajs
        self.append_loss_metrics = append_loss_metrics
        self.epoch_dir = ''

        self.multi_env_aggregation_type = multi_env_aggregation_type
        self.multi_env_aggregation_name = multi_env_aggregation_name

        # The primary env (also used for rendering/gifs/embeds) first, then the
        # solved-rate-only extra envs. Each env gets its own solved rate job per split.
        splits = [_VALID_STR] + ([_TRAIN_STR] if self.metrics_on_trainset else [])
        env_specs = [{'env_class': env_class, 'sr_params': primary_sr_params}]
        env_specs += extra_solved_rate_envs

        self.envs: List[VectorizedEnvironment] = []
        self.solved_rate_jobs: List[Dict[str, SolvedRateJob]] = []
        self.traj_vis_jobs: List[TrajectoryVisualizationJob] = []
        for spec in env_specs:
            spec = {'env_params': {}, 'sr_params': {}, 'vis_params': {}, **spec}
            env = spec['env_class'](
                device=device, **envs_common, **spec['env_params'], **spec_mgr_param
            )
            self.envs.append(env)
            self.solved_rate_jobs.append(
                {
                    split: solved_rate_job_class(
                        device=self.device,
                        env=env,
                        ve=self.ve,
                        print_stats=False,
                        dataset=self.datasets[split],
                        **spec['sr_params'],
                    )
                    for split in splits
                }
            )
            # Every env gets gifs equally: n_gifs is the fallback default, and each env
            # can override count/length via its own vis_params.
            vis_params = {'gif_max_len': gif_max_len, **spec['vis_params']}
            self.traj_vis_jobs.append(
                traj_vis_job_class(
                    env=env,
                    ve=self.ve,
                    device=self.device,
                    spec_mgr=spec_mgr_param.get('spec_mgr'),
                    logger=self.logger,
                    default_n_gifs=n_gifs,
                    **vis_params,
                )
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

        for states, extras in self.datasets[_VALID_STR]:
            states_repr = {k: self.model(v) for k, v in states.items()}
            loss, metrics = self.loss_fn(
                **states_repr,
                **extras,
                append_metrics=self.append_loss_metrics,
                append_empty=True,
            )
            acc.update({'loss': loss.item()} | metrics)

        return acc.average(prefix=_VALID_STR)

    @metric_action('correlation')
    def measure_monotonicity_correlation(self, epoch: int, split: str) -> Dict[str, float]:
        """
        Draws random trajectories and measures the Spearman rank correlation
        between predicted distances-to-goal and temporal order.
        A high correlation indicates the embedding orders states monotonically.
        """
        if self.n_corr_trajs == 0:
            return {}

        data = self.datasets[split].get_random_trajectory_data(k=self.n_corr_trajs)
        trajectories = data[self.datasets[split].master_tn]  # (n, T, state)
        lengths = data['lengths'].long()  # (n,)

        correlations = []
        for traj, length in zip(trajectories, lengths):
            length = length.item()
            if length < 3:  # need >= 2 points for a rank correlation
                continue
            traj = traj[:length]  # (L, state)
            goal = traj[-1].unsqueeze(0).expand_as(traj[:-1])  # (L-1, *state)
            dist = self.ve.get_solved_distance(traj[:-1], goal).cpu().numpy()  # (L-1,)
            correlations.append(-spearmanr(dist, np.arange(length - 1)).statistic)

        return {f'{split}_correlation': float(np.mean(correlations))}

    @metric_action('solved_rate')
    def measure_solved_rate(self, epoch: int, split: str, env_idx: int) -> Dict[str, float]:
        """
        Simulates solving attempts (no search) with the current model.
        Returns the solved rates for starting configurations,
        specified when constructing the SolvedRateJob instance.
        """
        res = self.solved_rate_jobs[env_idx][split].execute()
        return {f'{self.envs[env_idx].env_string()}_{split}_{k}': v for k, v in res.items()}

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
    def create_trajectory_gifs(self, epoch: int, split: str) -> Dict[str, float]:
        """
        Creates per-env trajectory visualizations (gifs, goal renders, board logs).
        Each env writes into its own subdir of the epoch's gifs directory.
        """
        out_dir = f'{self._get_epoch_dir(epoch)}/gifs'
        for job in self.traj_vis_jobs:
            job.execute(self.datasets[split], out_dir, split, epoch)
        return {}

    @metric_action('plot_embeds')
    def plot_trajectory_embeddings(self, epoch: int, split: str) -> None:
        """Creates and stores a t-SNE projection plot for trajectory-end embeddings."""
        if self.n_embed_trajs == 0:
            return

        tensors, labels = self.datasets[split].get_random_trajectory_ends(
            self.n_embed_trajs,
            self.embed_plot_max_len,
        )
        embeds = self.model(tensors)

        fig = get_embed_plots(embeds, labels)
        filename = f'{self._get_epoch_dir(epoch)}/trajectory-embeddings-{split}.png'
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        self.logger.log_image(epoch, {f'{split}_trajectory-embeddings': filename}, format='png')

    def aggregate(
        self, metrics: Dict[str, float], prefs: List[str], reject_suff: str
    ) -> Dict[str, float]:
        if self.multi_env_aggregation_type is None:
            return {}

        elements: Dict[str, List[float]] = {}

        def _add(body: str, value: float) -> None:
            if body not in elements:
                elements[body] = []
            elements[body].append(value)

        for k, v in metrics.items():
            if k.endswith(reject_suff):
                continue
            for pref in prefs:
                if k.startswith(pref):
                    body = k[len(pref) :]
                    _add(body, v)

        match self.multi_env_aggregation_type:
            case 'min':
                return {
                    f'{self.multi_env_aggregation_name}_{body}': min(vals)
                    for body, vals in elements.items()
                }
            case _:
                return {}

    def execute(self, epoch: int, train_metrics: Dict[str, float] = {}) -> None:
        if not self.logger.is_log_epoch(epoch):
            return

        self.model.eval()
        self.save_checkpoint(epoch)

        metrics = train_metrics.copy()

        if self.metrics_on_trainset:
            with self.datasets[_TRAIN_STR].preload_file_set():
                for i in range(len(self.envs)):
                    metrics |= self.measure_solved_rate(epoch, split=_TRAIN_STR, env_idx=i)
                metrics |= self.measure_monotonicity_correlation(epoch, split=_TRAIN_STR)

        with self.datasets[_VALID_STR].preload_file_set():
            for i in range(len(self.envs)):
                metrics |= self.measure_solved_rate(epoch, split=_VALID_STR, env_idx=i)
            metrics |= self.create_trajectory_gifs(epoch, split=_VALID_STR)
            metrics |= self.plot_trajectory_embeddings(epoch, split=_VALID_STR)
            metrics |= self.measure_monotonicity_correlation(epoch, split=_VALID_STR)
            metrics |= self.measure_valid_loss(epoch)

        prefs: List[str] = [f'{env.env_string()}_' for env in self.envs]
        metrics |= self.aggregate(metrics, prefs, reject_suff='_execution_time_seconds')

        self.logger.log_metrics(epoch, metrics)

        self.model.train()

    def finish_run(self) -> None:
        self.logger.finish_run()
