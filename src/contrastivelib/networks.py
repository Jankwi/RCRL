from typing import Type

import gin
import torch
import torch.nn as nn
import torch.nn.functional as f


def small_init(layer: nn.Module, sd: float) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=sd)
        nn.init.zeros_(layer.bias)


@gin.configurable
class LNConvBlock(nn.Module):
    """A single block of LNConvNet (see below)."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


@gin.configurable
class Embedding2D(nn.Module):
    def __init__(self, n_states: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_states, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x.long()).permute(0, 3, 1, 2)  # [M, embed_dim, grid_dim, grid_dim]


@gin.configurable
class LNConvNet(nn.Module):
    """
    Architecture adapted from the official code for the CRTR paper
    (https://github.com/Princeton-RL/CRTR/blob/main/networks.py).

    Expects input_network_class to produce spatial output (M, hidden_size, H, W).
    """

    def __init__(
        self,
        input_network_class: Type[nn.Module],
        hidden_size: int = 128,
        embed_dim: int = 64,
        block_depth: int = 4,
        dropout: float = 0.0,
        last_sd: float = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.input_network = input_network_class(embed_dim=hidden_size)
        self.first_bn = nn.BatchNorm2d(hidden_size)
        self.blocks = nn.Sequential(
            *[LNConvBlock(hidden_size, dropout=dropout) for _ in range(block_depth)]
        )
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, embed_dim),
        )

        if last_sd is not None:
            small_init(self.output_layer[-1], last_sd)
        for m in self.parameters():
            self._initialize_weights(m)

    def _initialize_weights(self, m: nn.Parameter) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.first_bn(self.input_network(x)))
        x = self.blocks(x)
        psi = self.output_layer(x)
        return psi
