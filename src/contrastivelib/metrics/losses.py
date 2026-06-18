import math
from abc import ABC, abstractmethod
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


@gin.configurable
class SupConLoss(AbstractContrastiveLoss):
    """
    Applies 2-class supervised contrastive loss to cluster positives and negatives
    https://arxiv.org/abs/2004.11362
    """

    def _get_class_term(
        self, in_class_sims: torch.Tensor, cross_class_sims: torch.Tensor
    ) -> torch.Tensor:
        N = in_class_sims.shape[0]
        assert N > 1, 'SupConLoss requires at least 2 samples per class'
        no_diag_mask = ~torch.eye(N, dtype=bool, device=self.device)
        non_self_in_class_sims = in_class_sims[no_diag_mask].view(N, N - 1)

        Pi_sims = non_self_in_class_sims
        Ai_sims = torch.cat([non_self_in_class_sims, cross_class_sims], dim=1)

        denominator = torch.logsumexp(Ai_sims, dim=1)
        fraction = Pi_sims - denominator.unsqueeze(1)

        return -fraction.mean(dim=1).mean()

    def __call__(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        special: torch.Tensor = None,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        if special is not None:
            positive = torch.cat([positive, special], dim=0)
        pos_sims = -self.cross_dist_logits(positive, positive)
        neg_sims = -self.cross_dist_logits(negative, negative)
        pos_neg_sims = -self.cross_dist_logits(positive, negative)

        pos_loss = self._get_class_term(pos_sims, pos_neg_sims)
        neg_loss = self._get_class_term(neg_sims, pos_neg_sims.t())
        loss = (pos_loss + neg_loss) / 2

        metrics = {}
        if append_metrics:
            metrics['pos_loss'] = pos_loss.item()
            metrics['neg_loss'] = neg_loss.item()
        return (loss, metrics) if (append_metrics or append_empty) else loss


@gin.configurable
class AbstractLinearSupConLoss(AbstractContrastiveLoss):
    """
    Base class for supervised contrastive losses that take into account, only the distance between
    positives/negatives and the positive special state
    """

    @abstractmethod
    def _calc_loss(self, pos_sims: torch.Tensor, neg_sims: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        special: torch.Tensor,
        append_metrics: bool = False,
        append_empty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        pos_sims = -self.cross_dist_logits(positive, special).squeeze(1)
        neg_sims = -self.cross_dist_logits(negative, special).squeeze(1)

        loss = self._calc_loss(pos_sims, neg_sims)

        metrics = {}
        if append_metrics:
            with torch.no_grad():
                # neg_sims is (n_pos * k_negatives,); reshape to (k_negatives, n_pos) so each
                # positive is compared against its own negatives.
                neg_grouped = neg_sims.reshape(-1, pos_sims.shape[0])
                metrics['elementwise_accuracy'] = (
                    (pos_sims.unsqueeze(0) > neg_grouped).float().mean().item()
                )
        return (loss, metrics) if (append_metrics or append_empty) else loss


@gin.configurable
class ElementwiseSupConLoss(AbstractLinearSupConLoss):
    """
    Optimizes per-positive fractions
    exp(sim(pos_i, special))/(exp(sim(pos_i, special)) + sum_j exp(sim(neg_ij, special)))
    where neg_ij are the k_negatives negatives paired with positive i.
    """

    def _calc_loss(self, pos_sims: torch.Tensor, neg_sims: torch.Tensor) -> torch.Tensor:
        n = pos_sims.shape[0]
        # neg_sims is (n * k_negatives,), laid out as k_negatives tiles of n.
        neg_grouped = neg_sims.reshape(-1, n)  # (k_negatives, n)
        pos_neg = torch.cat([pos_sims.unsqueeze(0), neg_grouped], dim=0)  # (1 + k_negatives, n)
        denominator = torch.logsumexp(pos_neg, dim=0)  # (n,)
        fraction = pos_sims - denominator
        return -fraction.mean()


@gin.configurable
class SparseSupConLoss(AbstractLinearSupConLoss):
    """
    Optimizes fractions of the form
    exp(sim(pos_i, special))/(exp(sim(pos_i, special)) + sum_{j} (exp(sim(neg_j, special))))
    so that each positive is closer to the special state, than all the negatives
    """

    def _calc_loss(self, pos_sims: torch.Tensor, neg_sims: torch.Tensor) -> torch.Tensor:
        log_sum_neg = torch.logsumexp(neg_sims, dim=0)
        denom_terms = torch.stack([pos_sims, log_sum_neg.expand_as(pos_sims)], dim=1)
        denominator = torch.logsumexp(denom_terms, dim=1)
        fraction = pos_sims - denominator
        return -fraction.mean()


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
