from abc import ABC
from typing import Dict

import gin
import torch

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
    def __init__(self, embed_dim: int, sim_fn: SimilarityMeasure, device: torch.device) -> None:
        self.tau = -torch.sqrt(torch.tensor([embed_dim], device=device))
        self.sim_fn = sim_fn
        self.device = device

    def _get_exp_sim(self, psi_0: torch.Tensor, psi_T: torch.Tensor, mode: str) -> torch.Tensor:
        sims = self.sim_fn(psi_0, psi_T, mode)
        return torch.exp(sims / self.tau)


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
        # Compute raw similarities without `.exp()` for numerical stability
        sims = self.sim_fn(psi_0, psi_T, 'cross') / self.tau
        # log-sum-exp trick avoids NaN from overflow/underflow
        loss = (-sims.diagonal() + torch.logsumexp(sims, dim=1)).mean()

        if not (append_metrics or append_empty):
            return loss
        metrics = {}
        if append_metrics:
            with torch.no_grad():
                n = sims.shape[0]
                correct = (sims.argmax(dim=1) == torch.arange(n, device=sims.device)).float().mean()
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
