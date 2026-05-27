"""Module containing all the different datasets"""

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.datasets.no_special import ContrastiveDatasetNoSpecial
from src.contrastivelib.datasets.passthrough import PassthroughDataset
from src.contrastivelib.datasets.special_padding import (
    ContrastiveDatasetSingleSpecial,
    ContrastiveDatasetSpecialPadding,
)
from src.contrastivelib.datasets.weighted_special import ContrastiveDatasetWeightedSpecial

__all__ = [
    'AbstractContrastiveDataset',
    'ContrastiveDatasetNoSpecial',
    'PassthroughDataset',
    'ContrastiveDatasetSingleSpecial',
    'ContrastiveDatasetSpecialPadding',
    'ContrastiveDatasetWeightedSpecial',
]
