from typing import Type

import gin
import torch
import torch.nn as nn
import torch.nn.functional as f

from src.contrastivelib.networks.bro import BroBlock
from src.contrastivelib.networks.conv import ConvNet  # noqa: F401
from src.contrastivelib.networks.utils import small_init


@gin.configurable
class LNDenseNet(nn.Module):
    """
    Adapted from https://github.com/Princeton-RL/CRTR
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
        self.first_ln = nn.LayerNorm(hidden_size)
        self.blocks = nn.Sequential(
            *[BroBlock(hidden_size, dropout=dropout) for i in range(block_depth)]
        )
        self.output_layer = nn.Linear(hidden_size, embed_dim)

        if last_sd is not None:
            small_init(self.output_layer, last_sd)
        for m in self.parameters():
            self.initialize_weights(m)

    def initialize_weights(self, m: nn.Parameter) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = f.relu(self.first_ln(self.input_network(x)))
        x = self.blocks(x)
        psi = self.output_layer(x)
        return psi
