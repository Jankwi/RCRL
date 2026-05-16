import gin
import torch
import torch.nn as nn


@gin.configurable
class Embedding2D(nn.Module):
    def __init__(self, n_states: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_states, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.long()
        return self.embed(x).permute(0, 3, 1, 2)  # [M, embed_dim, grid_dim, grid_dim]


@gin.configurable
class ConvNet(nn.Module):
    def __init__(self, grid_dim: int, embed_dim: int, n_states: int = 4, emb2d: int = 8) -> None:
        super().__init__()
        assert grid_dim == 21
        self.emb2d = emb2d
        self.conv = nn.Sequential(
            Embedding2D(n_states, emb2d),  # (M, E, 21, 21)
            nn.Conv2d(
                in_channels=emb2d, out_channels=(2 * emb2d), kernel_size=4
            ),  # (M, 2E, 18, 18)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3),  # (M, 2E, 6, 6)
        )
        self.final = nn.Linear(6 * 6, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).permute(0, 2, 3, 1)  # [M, 6, 6, 2*emb2d]
        x = x.mean(dim=-1)  # Average over channels -> [M, 6, 6]
        x = x.flatten(1)  # [M, 36]
        return self.final(x)
