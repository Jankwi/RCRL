import math
from abc import ABC
from typing import Dict

import gin
import torch
from torch.nn import BCEWithLogitsLoss

from src.contrastivelib.metrics.sims import SimilarityMeasure


class MetricAccumulator:
    """Accumulates metric dicts across batches and returns per-batch averages."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._count: int = 0

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
        self._count += 1

    def average(self, prefix: str = '') -> Dict[str, float]:
        if self._count == 0:
            return {}
        sep = '_' if prefix else ''
        return {f'{prefix}{sep}{k}': v / self._count for k, v in self._sums.items()}


@gin.configurable
class AbstractContrastiveLoss(ABC):
    def __init__(
        self, embed_dim: int, sim_fn: SimilarityMeasure, device: torch.device, tau: float = None
    ) -> None:
        self.tau = tau if tau is not None else 1 / math.sqrt(embed_dim)
        self.sim_fn = sim_fn
        self.device = device

    def cross_dist_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.sim_fn(a, b, 'cross') / self.tau


@gin.configurable
class BaselineContrastiveLoss(AbstractContrastiveLoss):
    """Applies symmetrized InfoNCE to states and goals"""

    def __init__(
        self,
        embed_dim: int,
        sim_fn: SimilarityMeasure,
        device: torch.device,
        loss_type: str = 'backward',
    ) -> None:
        super().__init__(embed_dim=embed_dim, sim_fn=sim_fn, device=device)
        self.loss_type = loss_type

    def info_nce(
        self,
        psi_0: torch.Tensor,
        psi_T: torch.Tensor,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        dists = self.cross_dist_logits(psi_0, psi_T)
        loss = (dists.diagonal() + torch.logsumexp(-dists, dim=1)).mean()

        if not (append_metrics or append_empty):
            return loss
        metrics = {}
        if append_metrics:
            with torch.no_grad():
                n = dists.shape[0]
                correct = (
                    (dists.argmin(dim=1) == torch.arange(n, device=dists.device)).float().mean()
                )
            metrics = {'contrastive_accuracy': correct.item()}
        return loss, metrics

    def __call__(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        match self.loss_type:
            case 'symmetric':
                fwd_loss, fwd_m = self.info_nce(state, goal, append_metrics, append_empty=True)
                bwd_loss, bwd_m = self.info_nce(goal, state, append_metrics, append_empty=True)
                loss = (fwd_loss + bwd_loss) / 2
                metrics = {f'forward_{k}': v for k, v in fwd_m.items()} | {
                    f'backward_{k}': v for k, v in bwd_m.items()
                }
            case 'forward':
                loss, m = self.info_nce(state, goal, append_metrics, append_empty=True)
                metrics = {f'forward_{k}': v for k, v in m.items()}
            case 'backward':
                loss, m = self.info_nce(goal, state, append_metrics, append_empty=True)
                metrics = {f'backward_{k}': v for k, v in m.items()}
            case _:
                raise ValueError()
        return (loss, metrics) if (append_metrics or append_empty) else loss


@gin.configurable
class CrossEntropyClassificationLoss(AbstractContrastiveLoss):
    def __init__(self, embed_dim: int, sim_fn: SimilarityMeasure, device: torch.device) -> None:
        super().__init__(embed_dim=embed_dim, sim_fn=sim_fn, device=device)
        self.criterion = BCEWithLogitsLoss()

    def __call__(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        positive_target = torch.ones((len(positive), 1), device=positive.device, dtype=torch.float)
        negative_target = torch.zeros((len(negative), 1), device=negative.device, dtype=torch.float)
        loss = self.criterion(positive, positive_target) + self.criterion(negative, negative_target)

        metrics = {}
        if append_metrics:
            with torch.no_grad():
                metrics['positive_accuracy'] = (
                    (positive > 0).float().mean().item() if len(positive) > 0 else 0.0
                )
                metrics['negative_accuracy'] = (
                    (negative < 0).float().mean().item() if len(negative) > 0 else 0.0
                )

        return (loss, metrics) if (append_metrics or append_empty) else loss
