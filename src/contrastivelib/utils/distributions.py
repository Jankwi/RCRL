import torch
from torch.distributions import Geometric


class DistributionSampler():
    def __init__(self, device: torch.device, gamma: float = None):
        self.device = device
        if gamma is not None:
            # [Pytorch docs]: (k+1)-th trial is the first success hence draws samples in {0,1,…}.
            self.geom = Geometric(probs=torch.tensor((1-gamma), device = self.device))

    def bounded_uniform(self, exclusive_high: torch.Tensor):
        """Samples numbers from [0, exclusive_high) from a uniform distribution"""
        exclusive_high = exclusive_high.to(torch.long)
        u = torch.rand(exclusive_high.shape, device=exclusive_high.device)
        return (u * exclusive_high).to(torch.long)

    def bounded_geometric(self, inclusive_high: torch.Tensor):
        """Samples numbers from [1, inclusive_high], from a rescaled geometric distribution"""
        return (torch.remainder(
            self.geom.sample(inclusive_high.shape).to(torch.long),
            inclusive_high.to(torch.long)) + 1
        )
