import gin
import torch
import torch.nn as nn


@gin.configurable
class Embedding2D(nn.Module):
    def __init__(self, n_states: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_states, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x.long()).permute(0, 3, 1, 2)  # [M, embed_dim, grid_dim, grid_dim]
