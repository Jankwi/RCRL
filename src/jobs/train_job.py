# ruff: noqa: F401
import math
import os
from typing import Dict, Type

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.contrastivelib.datasets import (
    AbstractContrastiveDataset,
    ContrastiveDatasetNoSpecial,
    ContrastiveDatasetSpecialPadding,
    ContrastiveDatasetWeightedSpecial,
)
from src.contrastivelib.metrics.losses import AbstractContrastiveLoss
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import BroNet
from src.contrastivelib.utils import fix_seed
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.jobs import AbstractJob
from src.jobs.solver import Solver, ValueEstimator
from src.jobs.utils.logging import AbstractLogger, StdOutLogger, WandbLogger


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
        seed: int,
        trainset_path: str,
        validset_path: str,
        base_model_class: Type[nn.Module],
        loss_class: Type[AbstractContrastiveLoss],
        logger_class: Type[AbstractLogger],
        dataset_class: Type[AbstractContrastiveDataset],
        solver_class: Type[Solver],
        ve_class: Type[ValueEstimator],
        spec_mgr_class: Type[SpecialStateManager] = None,
        gin_config: Dict[str, str] = None,
        epochs: int = None,
        lr: float = 0.001,
        update_steps: int = None,
        checkpoint_dir: str = 'checkpoints/',
    ) -> None:
        if not ((epochs is None) ^ (update_steps is None)):
            raise ValueError(f'Exactly one of {epochs=} and {update_steps=} has to be set')

        fix_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        sim_fn = get_similarity_measure(measure=metric_name, normalize=normalize)
        spec_mgr = spec_mgr_class(device=self.device) if spec_mgr_class is not None else None
        spec_mgr_param = {'spec_mgr': spec_mgr} if spec_mgr is not None else {}

        base_model = base_model_class(embed_dim=embed_dim)
        self.model = (
            SpecialStateModelWrapper(base_model=base_model, spec_mgr=spec_mgr, embed_dim=embed_dim)
            if spec_mgr is not None
            else base_model
        ).to(self.device)

        self.loss_fn = loss_class(embed_dim=embed_dim, sim_fn=sim_fn, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.trainset = dataset_class(
            file_sets_path=trainset_path, device=self.device, **spec_mgr_param
        )
        self.validset = dataset_class(
            file_sets_path=validset_path, device=self.device, **spec_mgr_param
        )

        self.epochs = (
            epochs
            if (epochs is not None)
            else math.ceil(update_steps / self.trainset.get_batches_per_epoch())
        )
        self.logger = logger_class(epochs=self.epochs, run_params=gin_config)

        self.solver = solver_class(
            dataset=self.validset,
            device=self.device,
            **spec_mgr_param,
            ve=ve_class(model=self.model, sim_fn=sim_fn, device=self.device),
        )

    def measure_valid_loss(self) -> float:
        """Compute the validation loss."""
        valid_loss = torch.tensor([0.0], device=self.device)

        for states, extras in self.validset:
            states_repr = states.apply(self.model)
            loss = self.loss_fn(**states_repr, **extras)
            valid_loss += loss

        return valid_loss.cpu().item() / self.validset.get_batches_per_epoch()

    def save_checkpoint(self, epoch: int) -> None:
        """Saves model & optimizer states locally, and then passes them to the logger"""
        epoch_path = f'{self.checkpoint_dir}/{epoch}/'
        os.makedirs(epoch_path, exist_ok=True)
        model_path = f'{epoch_path}/model.pt'
        optimizer_path = f'{epoch_path}/optimizer.pt'
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        self.logger.log_state_dicts(epoch, {'model': model_path, 'optimizer': optimizer_path})

    @torch.no_grad()
    def log_everything(self, epoch: int, avg_train_loss: float) -> None:
        """Calculates metrics, and then logs metrics & checpoints"""
        self.model.eval()
        self.logger.log_metrics(
            epoch,
            (
                {'train_loss': avg_train_loss}
                | {'valid_loss': self.measure_valid_loss()}
                | self.solver.solve()
            ),
        )
        self.save_checkpoint(epoch)
        self.model.train()

    def execute(self) -> None:
        """Trains the model, and logs metrics"""
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            train_loss_sum = torch.tensor([0.0], device=self.device)

            for states, extras in self.trainset:
                self.optimizer.zero_grad()

                states_repr = states.apply(self.model)
                loss = self.loss_fn(**states_repr, **extras)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    train_loss_sum += loss

            if self.logger.is_log_epoch(epoch):
                self.log_everything(
                    epoch, train_loss_sum.cpu().item() / self.trainset.get_batches_per_epoch()
                )

        self.logger.finish_run()
