"""Module containing all the different datasets"""

from src.contrastivelib.datasets.abstract_contrastive import AbstractContrastiveDataset
from src.contrastivelib.datasets.no_special import ContrastiveDatasetNoSpecial
from src.contrastivelib.datasets.simple_contrastive import SimplestContrastiveDataset

__all__ = [
    'AbstractContrastiveDataset',
    'ContrastiveDatasetNoSpecial',
    'SimplestContrastiveDataset',
]
