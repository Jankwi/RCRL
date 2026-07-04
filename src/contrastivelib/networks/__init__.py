"""Module containing all the different neural networks"""

from src.contrastivelib.networks.conv import Embedding2D
from src.contrastivelib.networks.ln_conv import LNConvNet

__all__ = ['Embedding2D', 'LNConvNet']
