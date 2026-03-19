import torch.nn as nn


def small_init(layer: nn.Module, sd: float) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=sd)
        nn.init.zeros_(layer.bias)
