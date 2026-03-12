import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.testing import assert_close

from contrastivelib.utils.special import SpecialStateManager, SpecialStateModelWrapper


class TestSpecialStateManager:
    """Tests for the SpecialStateManager class"""

    def setup_method(self, method):
        self.seed = 42
        torch.manual_seed(self.seed)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('val', [-1, -2])
    @pytest.mark.parametrize(
        'shape,n_states,special_pos',
        [
            ((10, 10), 5, [0, 2, 4]),
            ((10, 10), 3, [1]),
            ((5,), 4, [0, 3]),
            ((5,), 2, []),
        ],
    )
    def test_detect_special_states(self, val, shape, n_states, special_pos, device):
        """Tests SpecialStateManager's detect_special_states method"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        expected_mask = torch.zeros(n_states, dtype=torch.bool, device=device)
        expected_mask[special_pos] = True

        states = torch.rand(n_states, *shape, device=device) + 1.0
        for pos in special_pos:
            states[pos] = val

        # Initialize manager and test
        spec_mgr = SpecialStateManager(val=val, device=device, shape=shape)
        result_mask = spec_mgr.detect_special_states(states)

        assert_close(result_mask, expected_mask)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('val', [-1, -2])
    @pytest.mark.parametrize(
        'shape,n_states',
        [
            ((10, 10), 3),
            ((10, 10), 1),
            ((5,), 5),
            ((5,), 2),
        ],
    )
    def test_create_many(self, val, shape, n_states, device):
        """Tests SpecialStateManager's create_many method"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        # Create expected result BEFORE calling the function
        expected_shape = (n_states, *shape)
        expected_tensor = torch.full(expected_shape, val, device=device)

        # Initialize manager and create states
        spec_mgr = SpecialStateManager(val=val, device=device, shape=shape)
        result = spec_mgr.create_many(n_states)

        # Verify shape
        assert result.shape == expected_shape, (
            f'Expected shape {expected_shape}, got {result.shape}'
        )

        # Verify all values are val
        assert_close(result, expected_tensor)


class TestSpecialStateModelWrapper:
    """Tests for the SpecialStateModelWrapper class"""

    def setup_method(self, method):
        self.atol = 0.02
        self.val = -1
        self.seed = 42
        torch.manual_seed(self.seed)

    def fit_model(
        self,
        model: nn.Module,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        num_epochs: int = 2000,
        lr: float = 0.05,
    ):
        optimizer, loss_fn = Adam(model.parameters(), lr=lr), nn.MSELoss()
        for _ in range(num_epochs):
            optimizer.zero_grad()
            loss = loss_fn(train_y, model(train_X))
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize(
        'shape,train_sz,test_sz,spec_train_pos,spec_test_pos',
        [
            ((1, 2), 17, 10, [2, 4, 10], [3, 7]),
            ((3, 3), 20, 40, [1], []),
        ],
    )
    def test_wrapped_model(self, shape, train_sz, test_sz, spec_train_pos, spec_test_pos, device):
        """Tests SpecialStateModelWrapper's forward pass with special states"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        spec_mgr = SpecialStateManager(val=self.val, shape=shape, device=device)
        data_dim, embed_dim = shape[0] * shape[1], shape[0]
        base_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(data_dim, embed_dim),
        ).to(device)
        model = SpecialStateModelWrapper(base_model, spec_mgr, embed_dim).to(device)

        train_X = torch.rand(train_sz, *shape, device=device)
        test_X = torch.rand(test_sz, *shape, device=device)
        magic_embed = torch.randn(embed_dim, device=device)
        transform_matrix = torch.randn(data_dim, embed_dim, device=device)
        with torch.no_grad():
            train_y = train_X.flatten(1) @ transform_matrix
            test_y = test_X.flatten(1) @ transform_matrix

        for i in spec_train_pos:
            train_X[i] = self.val
            train_y[i] = magic_embed
        for i in spec_test_pos:
            test_X[i] = self.val
            test_y[i] = magic_embed

        self.fit_model(model, train_X, train_y)
        with torch.no_grad():
            test_y_hat = model(test_X)
            assert_close(test_y_hat, test_y, atol=self.atol, rtol=0.0)
