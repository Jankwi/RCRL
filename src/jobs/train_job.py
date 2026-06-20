# ruff: noqa: F401
import math
from typing import Dict, Type

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tqdm import tqdm

from src.contrastivelib.datasets import (
    AbstractContrastiveDataset,
    ContrastiveDatasetNoSpecial,
    PaddedCRTRDataset,
    PositiveNegativeDataset,
)
from src.contrastivelib.metrics.losses import (
    AbstractContrastiveLoss,
    BaselineContrastiveLoss,
    ElementwiseRelativeLoss,
    ElementwiseSupConLoss,
    MetricAccumulator,
    SparseSupConLoss,
    SupConLoss,
)
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import BroNet, ConvNet, Embedding2D, LNConvNet, LNDenseNet
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.jobs import AbstractJob
from src.jobs.metrics_job import MetricsJob
from src.jobs.utils import switch_path

_TRAIN_STR, _VALID_STR = 'train', 'valid'


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
        base_model_class: Type[nn.Module],
        loss_class: Type[AbstractContrastiveLoss],
        dataset_class: Type[AbstractContrastiveDataset],
        metrics_job_class: Type[MetricsJob],
        spec_mgr_class: Type[SpecialStateManager] = None,
        spec_model_wrapper_class: Type[nn.Module] = SpecialStateModelWrapper,
        trainset_path: str = None,
        dataset_dir: str = None,  # Used for deriving trainset/validset paths
        gin_config: Dict[str, str] = None,
        epochs: int = None,
        lr: float = 0.001,
        update_steps: int = None,
        append_loss_metrics: bool = True,
        cat_train_input: bool = True,
    ) -> None:
        trainset_path = switch_path(trainset_path)
        dataset_dir = switch_path(dataset_dir)
        if not ((trainset_path is None) ^ (dataset_dir is None)):
            raise ValueError(f'Exactly one of {trainset_path=} and {dataset_dir=} has to be set')
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
        self.cat_train_input = cat_train_input

        self.trainset_path = (
            trainset_path if trainset_path is not None else f'{dataset_dir}/{_TRAIN_STR}'
        )
        self.validset_path_param = (
            {'validset_path': f'{dataset_dir}/{_VALID_STR}'} if dataset_dir is not None else {}
        )
        self.trainset = dataset_class(
            file_sets_path=self.trainset_path, device=self.device, **spec_mgr_param
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
            default_validset_class=dataset_class,
            loss_fn=self.loss_fn,
            gin_config=gin_config,
            spec_mgr_param=spec_mgr_param,
            **self.validset_path_param,
        )

    def _train_forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forwards the input through the model, applying special token handling if configured."""
        if self.cat_train_input:
            labels = list(input.keys())
            sizes = [input[k].shape[0] for k in labels]
            cat_input = torch.cat([input[k] for k in labels], dim=0)
            cat_output = self.model(cat_input)
            split_output = torch.split(cat_output, sizes, dim=0)
            return {k: split_output[i] for i, k in enumerate(labels)}
        return {k: self.model(v) for k, v in input.items()}

    def execute(self) -> None:
        """Trains the model, and logs metrics"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            train_acc = MetricAccumulator()

            for states, extras in self.trainset:
                self.optimizer.zero_grad()

                states_repr = self._train_forward(states)
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
