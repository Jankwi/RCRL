import gin
import torch
import torch.nn as nn
import torch.nn.functional as f


@gin.configurable
class BroBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


@gin.configurable
class BroNet(nn.Module):
    """BRO nn.Module based on the BRO paper (https://arxiv.org/pdf/2405.16158)"""

    def __init__(
        self,
        input_dim: int = 400,
        embed_dim: int = 32,
        normalize_output: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            BroBlock(embed_dim, dropout),
            BroBlock(embed_dim, dropout),
        )
        self.normalize_output = normalize_output
        self.repr_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x.to(torch.float))

        if self.normalize_output:
            x = f.normalize(x, p=2, dim=-1)

        return x
