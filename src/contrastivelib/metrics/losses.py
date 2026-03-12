import torch

from src.contrastivelib.metrics.sims import SimilarityMeasure


class ContrastiveLoss:
    def __init__(self, embed_dim: int, sim_fn: SimilarityMeasure, device: torch.device) -> None:
        self.tau = -torch.sqrt(torch.tensor([embed_dim], device=device))
        self.sim_fn = sim_fn

    def info_nce(self, psi_0: torch.Tensor, psi_T: torch.Tensor) -> torch.Tensor:
        sims = self.sim_fn(psi_0, psi_T, 'cross')
        exp_sims = torch.exp(sims / self.tau)
        loss = -torch.log(exp_sims.diagonal() / exp_sims.sum(dim=1))
        return loss.mean()

    def __call__(self, state: torch.Tensor, goal: torch.Tensor, **kwargs) -> torch.Tensor:
        return (self.info_nce(state, goal) + self.info_nce(goal, state)) / 2
