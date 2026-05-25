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
        exp_sims = self._get_exp_sim(psi_0, psi_T, 'cross')
        loss = -torch.log(exp_sims.diagonal() / exp_sims.sum(dim=1)).mean()
        if not (append_metrics or append_empty):
            return loss
        metrics = {}
        if append_metrics:
            with torch.no_grad():
                n = exp_sims.shape[0]
                correct = (
                    (exp_sims.argmax(dim=1) == torch.arange(n, device=exp_sims.device))
                    .float()
                    .mean()
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
class WeightedSpecialContrastive(AbstractContrastiveLoss):
    """Factors the special state with a weight into the loss"""

    def __init__(self, embed_dim: int, sim_fn: SimilarityMeasure, device: torch.device) -> None:
        super().__init__(embed_dim=embed_dim, sim_fn=sim_fn, device=device)

    def __call__(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        spec_state: torch.Tensor,
        spec_weight: torch.Tensor,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        exp_sims = self._get_exp_sim(state, goal, 'cross')
        exp_special_sims = self._get_exp_sim(state, spec_state, 'elementwise')

        new_diag = exp_sims.diagonal() * (1 - spec_weight) + spec_weight * exp_special_sims
        exp_sims_sum = exp_sims.sum(dim=1) - exp_sims.diagonal() + new_diag

        loss = -torch.log(new_diag / exp_sims_sum).mean()
        return (loss, {}) if (append_metrics or append_empty) else loss


# class FullyWeightedSpecialContrastive(AbstractContrastiveLoss):
#     """Factors everything with an appropriate weight"""

#     def __init__(self, embed_dim: int, sim_fn: SimilarityMeasure, device: torch.device) -> None:
#         super().__init__(embed_dim=embed_dim, sim_fn=sim_fn, device=device)

#     def __call__(
#         self,
#         state: torch.Tensor,
#         goal: torch.Tensor,
#         spec_state: torch.Tensor,
#         spec_weight: torch.Tensor,
#     ) -> torch.Tensor:
#         exp_sims = self._get_exp_sim(state, goal, 'cross')
#         exp_special_sims = self._get_exp_sim(state, spec_state, 'elementwise')

#         new_diag = exp_sims.diagonal() * (1 - spec_weight) + spec_weight * exp_special_sims
#         exp_sims_sum = exp_sims.sum(dim=1) - exp_sims.diagonal() + new_diag

#         loss = -torch.log(new_diag / exp_sims_sum)
#         return loss.mean()
