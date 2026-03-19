from typing import Type

import gin
import torch
import torch.nn as nn
import torch.nn.functional as f

from src.contrastivelib.networks.bro import BroConvBlock
from src.contrastivelib.networks.utils import small_init


@gin.configurable
class LNConvNet(nn.Module):
    """
    Convolutional analogue of LNDenseNet.
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
            *[BroConvBlock(hidden_size, dropout=dropout) for _ in range(block_depth)]
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
