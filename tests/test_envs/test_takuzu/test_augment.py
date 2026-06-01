import pytest
import torch
from torch.testing import assert_close

from src.common import fix_seed
from src.envs.takuzu.augment import BadRandomPixelChoice, sample_random_on


def random_non_full_mask(m_boards, n, device):
    mask = torch.randint(0, 2, (m_boards, n, n), dtype=torch.bool, device=device)
    empty_idx = torch.randint(0, n * n, (m_boards,), device=device)
    for i in range(m_boards):
        mask.view(m_boards, -1)[i, empty_idx[i]] = False
    return mask


class TestSampleRandomOn:
    """Tests for sample_random_on function"""

    def setup_method(self, method):
        self.seed = 42
        self.atol = 0.02
        fix_seed(self.seed)

    @pytest.mark.parametrize('m_boards, samples', [(2, 10_000)])
    @pytest.mark.parametrize('n', [2, 4])
    @pytest.mark.parametrize('k', [1, 3])
    def test_sample_random_on(self, m_boards, n, samples, k, device_fx):
        mask = random_non_full_mask(m_boards, n, device_fx)
        pick_ratio = (k / mask.sum(dim=(1, 2)).float()).clamp(max=1.0).view(-1, 1, 1)
        expected_distrib = pick_ratio * mask.float()

        mask_rep = mask.repeat_interleave(samples, dim=0)
        sampled, valid = sample_random_on(mask_rep, k)
        sampled_grouped = sampled.view(m_boards, samples, -1)

        distrib = torch.zeros_like(expected_distrib)
        for mask_idx in range(m_boards):
            for i in range(n):
                for j in range(n):
                    idx = i * n + j
                    valid_samples = (sampled_grouped[mask_idx] == idx).sum(dim=-1)
                    distrib[mask_idx, i, j] = valid_samples.float().mean()

        assert_close(distrib, expected_distrib, atol=self.atol, rtol=0.0)


class TestBadRandomPixelChoice:
    """Tests for BadRandomPixelChoice augmentation"""

    def setup_method(self, method):
        self.seed = 42
        self.atol = 0.02
        fix_seed(self.seed)

    @pytest.mark.parametrize('n', [4])
    @pytest.mark.parametrize('k_flipped', [1, 3])
    @pytest.mark.parametrize('m_boards', [1000])
    def test_bad_random_pixel_choice(self, n, k_flipped, m_boards, device_fx):
        augment = BadRandomPixelChoice(k_flipped=k_flipped)

        full_board = torch.randint(1, 3, (m_boards, n, n), dtype=torch.uint8, device=device_fx)
        not_filled_mask = random_non_full_mask(m_boards, n, device_fx).to(torch.bool)
        states = full_board.clone()
        states[not_filled_mask] = 0

        positive, negative = augment(states, full_board)

        flip_mask = positive != states
        assert_close(positive[flip_mask], full_board[flip_mask])
        assert_close(negative[flip_mask], 3 - full_board[flip_mask])

        expected_flipped = not_filled_mask.sum(dim=(1, 2)).clamp(max=k_flipped)
        assert (expected_flipped == flip_mask.sum(dim=(1, 2))).all().item()
