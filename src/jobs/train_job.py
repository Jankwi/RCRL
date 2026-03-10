# ruff: noqa: F401
from typing import Dict, Type

import gin
import torch
import torch.nn as nn
import torch.optim as optim

from src.contrastivelib.datasets import AbstractContrastiveDataset, SimplestContrastiveDataset
from src.contrastivelib.metrics.losses import ContrastiveLoss
from src.contrastivelib.metrics.sims import get_similarity_measure
from src.contrastivelib.networks import BroNet
from src.contrastivelib.utils import fix_seed
from src.contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper
from src.jobs import AbstractJob
from src.jobs.utils.logging import AbstractLogger, MultipleLoggers, StdOutLogger, WandbLogger


@gin.configurable
class TrainJob(AbstractJob):
    def __init__(self,
        embed_dim: int, metric_name: str, normalize: bool, seed: int,
        train_dataset_path: str, epochs: int,
        spec_mgr_class: Type[SpecialStateManager], base_model_class: Type[nn.Module],
        logger_class: Type[AbstractLogger], dataset_class: Type[AbstractContrastiveDataset],
        gin_config: Dict[str, str] = None
    ):
        fix_seed(seed)
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sim_fn = get_similarity_measure(measure=metric_name, normalize=normalize)

        spec_mgr = spec_mgr_class(device=self.device)
        self.model = SpecialStateModelWrapper(
            base_model=base_model_class(embed_dim=embed_dim),
            spec_mgr=spec_mgr,
            embed_dim=embed_dim
        ).to(self.device)

        self.loss_fn = ContrastiveLoss(embed_dim=embed_dim, sim_fn=sim_fn, device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)

        self.train_dataset = dataset_class(
            file_sets_path=train_dataset_path,
            device=self.device,
            spec_mgr=spec_mgr
        )

        self.logger = logger_class(
            gin_config if gin_config is not None else ({
            'TrainJob.embed_dim': embed_dim,
            'TrainJob.metric_name': metric_name,
            'TrainJob.normalize': normalize,
            'TrainJob.seed': seed,
            'TrainJob.train_dataset': train_dataset_path,
            'TrainJob.epochs': self.epochs
        }))

    def execute(self):
        self.model.train()

        for epoch in range(self.epochs):
            train_loss = torch.tensor([0.], device = self.device)

            for state, goal in self.train_dataset:
                self.optimizer.zero_grad()

                psi_0 = self.model(state)
                psi_T = self.model(goal)
                loss = self.loss_fn(psi_0, psi_T)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    train_loss += loss

            self.logger.log_metrics(epoch, {
                'train_loss': train_loss.cpu().item()
            })

        self.logger.finish_run()
