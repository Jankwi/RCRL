import pytest
import torch
from torch.testing import assert_close

from src.contrastivelib.metrics.losses import (
    BaselineContrastiveLoss,
    ElementwiseRelativeLoss,
    ElementwiseSupConLoss,
    MetricAccumulator,
)
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


def _elementwise_loss(device):
    sim_fn = L2SimilarityMeasure(normalize=False)
    return ElementwiseSupConLoss(embed_dim=EMBED_DIM, sim_fn=sim_fn, device=device)


def _brute_force_elementwise(pos_sims, neg_sims, k):
    """Naive per-positive -log(exp(pos)/(exp(pos) + sum_j exp(neg_j)))."""
    n = pos_sims.shape[0]
    total = pos_sims.new_zeros(())
    for j in range(n):
        negs_j = [neg_sims[t * n + j] for t in range(k)]
        denom = torch.exp(pos_sims[j]) + sum(torch.exp(ng) for ng in negs_j)
        total += -(pos_sims[j] - torch.log(denom))
    return total / n


@pytest.mark.parametrize('k', [1, 2, 5])
def test_elementwise_supcon_matches_brute_force(device_fx, k):
    torch.manual_seed(0)
    n = 6
    pos_sims = torch.randn(n, device=device_fx)
    neg_sims = torch.randn(n * k, device=device_fx)

    loss = _elementwise_loss(device_fx)._calc_loss(pos_sims, neg_sims)
    expected = _brute_force_elementwise(pos_sims, neg_sims, k)
    assert_close(loss, expected)


def test_elementwise_supcon_k1_matches_legacy_stack(device_fx):
    torch.manual_seed(1)
    n = 8
    pos_sims = torch.randn(n, device=device_fx)
    neg_sims = torch.randn(n, device=device_fx)

    loss = _elementwise_loss(device_fx)._calc_loss(pos_sims, neg_sims)
    legacy = -(pos_sims - torch.logsumexp(torch.stack([pos_sims, neg_sims], dim=1), dim=1)).mean()
    assert_close(loss, legacy)


def _relative_loss(device):
    sim_fn = L2SimilarityMeasure(normalize=False)
    return ElementwiseRelativeLoss(embed_dim=EMBED_DIM, sim_fn=sim_fn, device=device)


def _brute_force_relative(loss_fn, pos_state, neg_states, goal):
    """
    Naive per-positive InfoNCE against a per-sample goal, mirroring the dataset's
    (n, k) -> (n * k) negative layout (neg of positive b, slot i is row b * k + i).
    Distances are negated to similarity logits so smaller distance scores higher.
    """
    sim_fn, tau = loss_fn.sim_fn, loss_fn.tau
    n = pos_state.shape[0]
    k = neg_states.shape[0] // n
    losses, correct = [], []
    for b in range(n):
        pg = -sim_fn(pos_state[b : b + 1], goal[b : b + 1], 'elementwise')[0] / tau
        ngs = []
        for i in range(k):
            neg = neg_states[b * k + i].unsqueeze(0)
            ng = -sim_fn(neg, goal[b : b + 1], 'elementwise')[0] / tau
            ngs.append(ng)
            correct.append(float(pg > ng))
        denom = torch.logsumexp(torch.stack([pg, *ngs]), dim=0)
        losses.append(-(pg - denom))
    return torch.stack(losses).mean(), sum(correct) / len(correct)


@pytest.mark.parametrize('k', [1, 2, 5])
def test_relative_matches_brute_force(device_fx, k):
    torch.manual_seed(0)
    n = 6
    pos_state = torch.randn(n, EMBED_DIM, device=device_fx)
    neg_states = torch.randn(n * k, EMBED_DIM, device=device_fx)
    goal = torch.randn(n, EMBED_DIM, device=device_fx)

    loss_fn = _relative_loss(device_fx)
    loss, metrics = loss_fn(pos_state, neg_states, goal, append_metrics=True)
    expected_loss, expected_acc = _brute_force_relative(loss_fn, pos_state, neg_states, goal)

    assert loss.shape == torch.Size([])
    assert_close(loss, expected_loss)
    assert_close(torch.tensor(metrics['elementwise_accuracy']), torch.tensor(expected_acc))


def test_relative_uses_per_sample_goal(device_fx):
    """Each positive must be scored against its OWN goal (elementwise), not all-pairs (cross).

    With pos_state == goal (distinct rows) and far negatives, every positive sits exactly on
    its goal, so accuracy is 1.0 — a cross-product implementation would mismatch rows.
    """
    torch.manual_seed(2)
    n, k = 5, 3
    goal = torch.randn(n, EMBED_DIM, device=device_fx) * 5.0
    pos_state = goal.clone()
    neg_states = goal.repeat_interleave(k, dim=0) + 100.0  # far from every goal

    loss_fn = _relative_loss(device_fx)
    _, metrics = loss_fn(pos_state, neg_states, goal, append_metrics=True)
    assert_close(torch.tensor(metrics['elementwise_accuracy']), torch.tensor(1.0))


def test_relative_pulls_positive_toward_goal(device_fx):
    """Positive close to its goal must yield a lower loss than a positive far from it.

    This pins the sign: distances are negated into similarity logits, so the softmax
    rewards (not penalises) proximity to the goal.
    """
    torch.manual_seed(3)
    n, k = 4, 2
    goal = torch.randn(n, EMBED_DIM, device=device_fx)
    neg_states = torch.randn(n * k, EMBED_DIM, device=device_fx) + 10.0

    loss_fn = _relative_loss(device_fx)
    close_loss = loss_fn(goal.clone() + 0.01, neg_states, goal)
    far_loss = loss_fn(goal.clone() + 10.0, neg_states, goal)
    assert close_loss < far_loss


def test_relative_no_metrics_by_default(device_fx):
    torch.manual_seed(4)
    n, k = 4, 2
    pos_state = torch.randn(n, EMBED_DIM, device=device_fx)
    neg_states = torch.randn(n * k, EMBED_DIM, device=device_fx)
    goal = torch.randn(n, EMBED_DIM, device=device_fx)

    result = _relative_loss(device_fx)(pos_state, neg_states, goal)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])


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
