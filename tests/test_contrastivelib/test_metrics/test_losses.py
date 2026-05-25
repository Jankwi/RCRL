import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.metrics.losses import BaselineContrastiveLoss, MetricAccumulator
from src.contrastivelib.metrics.sims import L2SimilarityMeasure

EMBED_DIM = 4
BATCH_SIZE = 4


@pytest.fixture(params=['forward', 'backward', 'symmetric'])
def loss_type(request):
    return request.param


@pytest.fixture
def loss_fn(device_fx, loss_type):
    sim_fn = L2SimilarityMeasure(normalize=False)
    return BaselineContrastiveLoss(
        embed_dim=EMBED_DIM, sim_fn=sim_fn, device=device_fx, loss_type=loss_type
    )


def _make_embeddings(device, pattern='perfect'):
    """
    Returns (psi_0, psi_T) pairs for deterministic accuracy tests.

    perfect: psi_0[i] == psi_T[i], all rows distinct → accuracy = 1.0
    zero:    psi_0[i] == psi_T[(i+1) % N] → accuracy = 0.0
    half:    rows 0,1 correct; rows 2,3 swapped → accuracy = 0.5
    """
    base = torch.eye(BATCH_SIZE, EMBED_DIM, device=device) * 10.0

    if pattern == 'perfect':
        return base.clone(), base.clone()

    if pattern == 'zero':
        psi_T = torch.roll(base, shifts=1, dims=0)
        return base.clone(), psi_T

    if pattern == 'half':
        psi_T = base.clone()
        psi_T[2], psi_T[3] = base[3].clone(), base[2].clone()
        return base.clone(), psi_T

    raise ValueError(pattern)


@pytest.mark.parametrize(
    'pattern,expected_acc',
    [
        ('perfect', 1.0),
        ('zero', 0.0),
        ('half', 0.5),
    ],
)
def test_contrastive_accuracy(loss_fn, device_fx, loss_type, pattern, expected_acc):
    psi_0, psi_T = _make_embeddings(device_fx, pattern)

    loss, metrics = loss_fn(psi_0, psi_T, append_metrics=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    if loss_type == 'symmetric':
        assert 'forward_contrastive_accuracy' in metrics
        assert 'backward_contrastive_accuracy' in metrics
        # For symmetric loss with symmetric embeddings (psi_0 and psi_T have same structure),
        # both directions should give the same accuracy
        assert_close(
            torch.tensor(metrics['forward_contrastive_accuracy']),
            torch.tensor(expected_acc),
        )
        assert_close(
            torch.tensor(metrics['backward_contrastive_accuracy']),
            torch.tensor(expected_acc),
        )
    elif loss_type == 'forward':
        assert 'forward_contrastive_accuracy' in metrics
        assert_close(
            torch.tensor(metrics['forward_contrastive_accuracy']),
            torch.tensor(expected_acc),
        )
    elif loss_type == 'backward':
        assert 'backward_contrastive_accuracy' in metrics
        assert_close(
            torch.tensor(metrics['backward_contrastive_accuracy']),
            torch.tensor(expected_acc),
        )


def test_no_metrics_by_default(loss_fn, device_fx):
    psi_0, psi_T = _make_embeddings(device_fx, 'perfect')
    result = loss_fn(psi_0, psi_T)
    assert isinstance(result, torch.Tensor)


def test_accuracy_is_no_grad(loss_fn, device_fx):
    psi_0, psi_T = _make_embeddings(device_fx, 'perfect')
    _, metrics = loss_fn(psi_0, psi_T, append_metrics=True)
    for v in metrics.values():
        assert isinstance(v, float)


@pytest.mark.parametrize('n_updates', [1, 3])
def test_metric_accumulator(n_updates):
    torch.manual_seed(0)
    vals = torch.rand(n_updates, 2)

    acc = MetricAccumulator()
    for i in range(n_updates):
        acc.update({'metric_0': vals[i][0].item(), 'metric_1': vals[i][1].item()})

    expected = vals.mean(dim=0)
    result = acc.average()
    assert_close(torch.tensor(result['metric_0']), expected[0])
    assert_close(torch.tensor(result['metric_1']), expected[1])

    prefixed = acc.average(prefix='train')
    assert set(prefixed.keys()) == {'train_metric_0', 'train_metric_1'}
    assert_close(torch.tensor(prefixed['train_metric_0']), expected[0])
    assert_close(torch.tensor(prefixed['train_metric_1']), expected[1])
