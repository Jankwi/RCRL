from abc import ABC

import gin
import torch

from src.contrastivelib.metrics.sims import SimilarityMeasure


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

    def info_nce(self, psi_0: torch.Tensor, psi_T: torch.Tensor) -> torch.Tensor:
        exp_sims = self._get_exp_sim(psi_0, psi_T, 'cross')
        loss = -torch.log(exp_sims.diagonal() / exp_sims.sum(dim=1))
        return loss.mean()

    def __call__(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        match self.loss_type:
            case 'symmetric':
                return (self.info_nce(state, goal) + self.info_nce(goal, state)) / 2
            case 'forward':
                return self.info_nce(state, goal)
            case 'backward':
                return self.info_nce(goal, state)
            case _:
                raise ValueError()


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
    ) -> torch.Tensor:
        exp_sims = self._get_exp_sim(state, goal, 'cross')
        exp_special_sims = self._get_exp_sim(state, spec_state, 'elementwise')

        new_diag = exp_sims.diagonal() * (1 - spec_weight) + spec_weight * exp_special_sims
        exp_sims_sum = exp_sims.sum(dim=1) - exp_sims.diagonal() + new_diag

        loss = -torch.log(new_diag / exp_sims_sum)
        return loss.mean()


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
