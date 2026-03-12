# ruff: noqa: F401
import math
from typing import Dict, Type

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.contrastivelib.datasets import AbstractContrastiveDataset, SimplestContrastiveDataset
from src.contrastivelib.metrics.losses import ContrastiveLoss
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import BroNet
from src.contrastivelib.utils import fix_seed
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.jobs import AbstractJob
from src.jobs.utils.logging import AbstractLogger, StdOutLogger, WandbLogger


@gin.configurable
class TrainJob(AbstractJob):
    def __init__(
        self,
        embed_dim: int,
        metric_name: str,
        normalize: bool,
        seed: int,
        trainset_path: str,
        validset_path: str,
        spec_mgr_class: Type[SpecialStateManager],
        base_model_class: Type[nn.Module],
        logger_class: Type[AbstractLogger],
        dataset_class: Type[AbstractContrastiveDataset],
        gin_config: Dict[str, str] = None,
        epochs: int = None,
        update_steps: int = None,
    ) -> None:
        if not ((epochs is None) ^ (update_steps is None)):
            raise ValueError(f'Exactly one of {epochs=} and {update_steps=} has to be set')

        fix_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sim_fn = get_similarity_measure(measure=metric_name, normalize=normalize)
        spec_mgr = spec_mgr_class(device=self.device)

        self.model = SpecialStateModelWrapper(
            base_model=base_model_class(embed_dim=embed_dim), spec_mgr=spec_mgr, embed_dim=embed_dim
        ).to(self.device)
        self.loss_fn = ContrastiveLoss(embed_dim=embed_dim, sim_fn=sim_fn, device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)

        self.trainset = dataset_class(
            file_sets_path=trainset_path, device=self.device, spec_mgr=spec_mgr
        )
        self.validset = dataset_class(
            file_sets_path=validset_path, device=self.device, spec_mgr=spec_mgr
        )
        self.epochs = (
            epochs
            if (epochs is not None)
            else math.ceil(update_steps / self.trainset.get_batches_per_epoch())
        )

        self.logger = logger_class(epochs=self.epochs, run_params=gin_config)

    @torch.no_grad()
    def measure_valid_loss(self) -> float:
        """Compute the validation loss."""
        valid_loss = torch.tensor([0.0], device=self.device)

        for states, extras in self.validset:
            states_repr = states.apply(self.model)
            loss = self.loss_fn(**states_repr, **extras)
            valid_loss += loss

        return valid_loss.cpu().item() / self.validset.get_batches_per_epoch()

    def execute(self) -> None:
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            train_loss = torch.tensor([0.0], device=self.device)

            for states, extras in self.trainset:
                self.optimizer.zero_grad()

                states_repr = states.apply(self.model)
                loss = self.loss_fn(**states_repr, **extras)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    train_loss += loss

            if self.logger.is_log_epoch(epoch):
                avg_train_loss = train_loss.cpu().item() / self.trainset.get_batches_per_epoch()
                avg_valid_loss = self.measure_valid_loss()
                self.logger.log_metrics(
                    epoch, {'train_loss': avg_train_loss, 'valid_loss': avg_valid_loss}
                )

        self.logger.finish_run()
