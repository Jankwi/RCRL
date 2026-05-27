# ruff: noqa: F401
import math
from typing import Dict, Type

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.contrastivelib.datasets import (
    AbstractContrastiveDataset,
    ContrastiveDatasetNoSpecial,
    ContrastiveDatasetSingleSpecial,
    ContrastiveDatasetSpecialPadding,
    ContrastiveDatasetWeightedSpecial,
)
from src.contrastivelib.metrics.losses import AbstractContrastiveLoss, MetricAccumulator
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import BroNet, ConvNet, Embedding2D, LNConvNet, LNDenseNet
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.jobs import AbstractJob
from src.jobs.metrics_job import MetricsJob


@gin.configurable
class TrainJob(AbstractJob):
    """
    A Contrastive Train Job for training models on goal-reaching tasks.
    If a special_manager is configured (using special tokens to represent abstract states),
    then it is passed to all relevant classes.
    """

    def __init__(
        self,
        embed_dim: int,
        metric_name: str,
        normalize: bool,
        trainset_path: str,
        base_model_class: Type[nn.Module],
        loss_class: Type[AbstractContrastiveLoss],
        dataset_class: Type[AbstractContrastiveDataset],
        metrics_job_class: Type[MetricsJob],
        spec_mgr_class: Type[SpecialStateManager] = None,
        spec_model_wrapper_class: Type[nn.Module] = SpecialStateModelWrapper,
        gin_config: Dict[str, str] = None,
        epochs: int = None,
        lr: float = 0.001,
        update_steps: int = None,
        append_loss_metrics: bool = True,
    ) -> None:
        if not ((epochs is None) ^ (update_steps is None)):
            raise ValueError(f'Exactly one of {epochs=} and {update_steps=} has to be set')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sim_fn = get_similarity_measure(measure=metric_name, normalize=normalize)
        spec_mgr = spec_mgr_class(device=self.device) if spec_mgr_class is not None else None
        spec_mgr_param = {'spec_mgr': spec_mgr} if spec_mgr is not None else {}

        base_model = base_model_class(embed_dim=embed_dim)
        self.model = (
            spec_model_wrapper_class(base_model=base_model, spec_mgr=spec_mgr, embed_dim=embed_dim)
            if spec_mgr is not None
            else base_model
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.loss_fn = loss_class(embed_dim=embed_dim, sim_fn=sim_fn, device=self.device)
        self.trainset = dataset_class(
            file_sets_path=trainset_path, device=self.device, **spec_mgr_param
        )
        self.epochs = (
            epochs
            if (epochs is not None)
            else math.ceil(update_steps / self.trainset.get_batches_per_epoch())
        )

        self.append_loss_metrics = append_loss_metrics
        self.metrics_job = metrics_job_class(
            epochs=epochs,
            device=self.device,
            model=self.model,
            optimizer=self.optimizer,
            trainset=self.trainset,
            sim_fn=sim_fn,
            loss_fn=self.loss_fn,
            gin_config=gin_config,
            spec_mgr_param=spec_mgr_param,
        )

    def execute(self) -> None:
        """Trains the model, and logs metrics"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            train_acc = MetricAccumulator()

            for states, extras in self.trainset:
                self.optimizer.zero_grad()

                states_repr = states.apply(self.model)
                loss, metrics = self.loss_fn(
                    **states_repr,
                    **extras,
                    append_metrics=self.append_loss_metrics,
                    append_empty=True,
                )

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    train_acc.update({'loss': loss.item()} | metrics)

            with torch.inference_mode():
                self.metrics_job.execute(epoch, train_acc.average(prefix='train'))

        self.metrics_job.finish_run()
