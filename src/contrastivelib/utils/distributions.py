import torch
from torch.distributions import Geometric


class DistributionSampler:
    def __init__(self, device: torch.device, gamma: float = None) -> None:
        self.device = device
        if gamma is not None:
            # [Pytorch docs]: (k+1)-th trial is the first success hence draws samples in {0,1,…}.
            self.failure_prob = torch.tensor(gamma, device=device)
            self.geom = Geometric(probs=(1 - self.failure_prob))

    def bounded_uniform(self, exclusive_high: torch.Tensor) -> torch.Tensor:
        """Samples numbers from [0, exclusive_high) from a uniform distribution"""
        exclusive_high = exclusive_high.to(torch.long)
        u = torch.rand(exclusive_high.shape, device=exclusive_high.device)
        return (u * exclusive_high).to(torch.long)

    def bounded_geometric(self, inclusive_high: torch.Tensor) -> torch.Tensor:
        """Samples numbers from [1, inclusive_high], from a rescaled geometric distribution"""
        return (
            torch.remainder(
                self.geom.sample(inclusive_high.shape).to(torch.long), inclusive_high.to(torch.long)
            )
            + 1
        )

    def clamped_geometric(self, inclusive_high: torch.Tensor) -> torch.Tensor:
        """
        Samples numbers from [1, inclusive_high] from a geometric distribution,
        by clamping samples drawn above this range
        """
        return torch.clamp(
            self.geom.sample(inclusive_high.shape).to(torch.long) + 1, max=inclusive_high
        )

    def geometric_over_low(self, inclusive_low: torch.Tensor) -> torch.Tensor:
        """
        For each k := inclusive_low[i],
        returns the probability of the unbounded geometric distribution being at least k.
        """
        return self.failure_prob ** (inclusive_low - 1)
