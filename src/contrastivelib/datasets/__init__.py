"""Module containing all the different datasets"""

from contrastivelib.datasets.special_padding import ContrastiveDatasetSpecialPadding
from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.datasets.no_special import ContrastiveDatasetNoSpecial
from src.contrastivelib.datasets.weighted_special import ContrastiveDatasetWeightedSpecial

__all__ = [
    'AbstractContrastiveDataset',
    'ContrastiveDatasetNoSpecial',
    'ContrastiveDatasetSpecialPadding',
    'ContrastiveDatasetWeightedSpecial',
]
