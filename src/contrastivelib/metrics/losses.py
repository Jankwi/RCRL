import math
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
    def __init__(
        self,
        embed_dim: int,
        sim_fn: SimilarityMeasure,
        device: torch.device,
        tau: float = None,
        default_normalized_tau: float = 0.07,
    ) -> None:
        self.sim_fn = sim_fn
        self.device = device
        self._set_temperature(tau, embed_dim, default_normalized_tau)

    def _set_temperature(self, tau: float, embed_dim: int, default_normalized_tau: float) -> None:
        if tau is not None:
            self.tau = tau
        elif self.sim_fn.normalize:
            self.tau = default_normalized_tau
        else:
            self.tau = math.sqrt(embed_dim)

    def cross_dist_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.sim_fn(a, b, 'cross') / self.tau

    def elem_dist_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.sim_fn(a, b, 'elementwise') / self.tau


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
class ElementwiseRelativeLoss(AbstractContrastiveLoss):
    """
    Optimizes per-positive fractions
    exp(sim(pos_i, goal_i))/(exp(sim(pos_i, goal_i)) + sum_j exp(sim(neg_ij, goal_i)))
    where neg_ij are the k_negatives negatives paired with positive i.
    """

    def __call__(
        self,
        pos_state: torch.Tensor,
        neg_states: torch.Tensor,
        goal: torch.Tensor,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        n = pos_state.shape[0]
        # neg_states is (n * k_negatives, ...); reshape to (n, k_negatives, ...) so each
        # positive is compared against its own negatives.
        neg_grouped = neg_states.reshape(n, -1, *neg_states.shape[1:])
        # Elementwise (per-sample goal), negated so smaller distance is a larger similarity logit.
        pos_goal = -self.elem_dist_logits(pos_state, goal)  # (n,)
        neg_goal = -self.elem_dist_logits(neg_grouped, goal.unsqueeze(1))  # (n, k_negatives)

        # neg_goal is (n, k_negatives); concatenate with pos_goal to get (n, 1 + k_negatives)
        pos_neg = torch.cat([pos_goal.unsqueeze(1), neg_goal], dim=1)
        denominator = torch.logsumexp(pos_neg, dim=1)
        fraction = pos_goal - denominator
        loss = -fraction.mean()

        metrics = {}
        if append_metrics:
            with torch.no_grad():
                metrics['elementwise_accuracy'] = (
                    (pos_goal.unsqueeze(1) > neg_goal).float().mean().item()
                )
        return (loss, metrics) if (append_metrics or append_empty) else loss
