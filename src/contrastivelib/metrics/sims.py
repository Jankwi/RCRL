from abc import abstractmethod

import torch
import torch.nn.functional as F


class SimilarityMeasure:
    """Abstract class for similarity measures (lower is closer)"""

    def __init__(self, normalize: bool, p: int = None) -> None:
        self.normalize, self.p = normalize, p

    def _norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, p=self.p, dim=-1) if self.normalize else tensor

    @abstractmethod
    def _measure(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, a: torch.Tensor, b: torch.Tensor, mode: str = 'elementwise') -> torch.Tensor:
        """ "
        Calculate the similarities between two tensors.
        Assumes that a and b are 2D tensors of the same shape (batch_size, latent_dim).

        With mode = 'elementwise',
            returns a 1D tensor c, where c[i] = sim(a[i], b[i])
        With mode = 'cross'
            returnsa a 2D tensor c, where c[i][j] = sim(a[i], b[j])
        """
        a, b = self._norm(a), self._norm(b)
        match mode:
            case 'elementwise':
                return self._measure(a=a, b=b)
            case 'cross':
                return self._measure(
                    torch.repeat_interleave(a.unsqueeze(1), b.shape[0], 1),
                    torch.repeat_interleave(b.unsqueeze(0), a.shape[0], 0),
                )
            case _:
                raise ValueError(f'Invalid similarity measure mode {mode}')


class L2SimilarityMeasure(SimilarityMeasure):
    """SimilarityMeasure class for Euclidean distance"""

    def __init__(self, normalize: bool) -> None:
        super().__init__(normalize=normalize, p=2)

    def _measure(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


class DotProductSimilarityMeasure(SimilarityMeasure):
    """SimilarityMeasure class for the negative dot product"""

    def __init__(self, normalize: bool) -> None:
        super().__init__(normalize=normalize, p=2)

    def _measure(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return -torch.sum(a * b, dim=-1)


class L1SimilarityMeasure(SimilarityMeasure):
    """SimilarityMeasure class for Manhattan distance"""

    def __init__(self, normalize: bool) -> None:
        super().__init__(normalize=normalize, p=1)

    def _measure(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(a - b), dim=-1)


class L2SquaredSimilarityMeasure(SimilarityMeasure):
    """SimilarityMeasure class for squared Euclidean distance"""

    def __init__(self, normalize: bool) -> None:
        super().__init__(normalize=normalize, p=2)

    def _measure(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sum((a - b) ** 2, dim=-1)


def get_similarity_measure(measure: str, normalize: bool = False) -> SimilarityMeasure:
    match measure:
        case 'L2' | 'euclidean':
            return L2SimilarityMeasure(normalize=normalize)
        case 'dot':
            return DotProductSimilarityMeasure(normalize=normalize)
        case 'L1' | 'manhattan':
            return L1SimilarityMeasure(normalize=normalize)
        case 'L22':
            return L2SquaredSimilarityMeasure(normalize=normalize)
        case _:
            raise ValueError(f'Unknown similarity measure name {measure}')
