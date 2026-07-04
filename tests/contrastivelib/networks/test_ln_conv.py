from functools import partial

import torch

from src.contrastivelib.networks import Embedding2D, LNConvNet


class TestLNConvNet:
    GRID_DIM = 21
    N_STATES = 4

    def test_output_format(self, device_fx, batch_size, embed_dim):
        model = LNConvNet(
            input_network_class=partial(Embedding2D, n_states=self.N_STATES),
            embed_dim=embed_dim,
            block_depth=2,
            hidden_size=8,
        ).to(device_fx)
        x = torch.randint(
            0, self.N_STATES, (batch_size, self.GRID_DIM, self.GRID_DIM), device=device_fx
        )
        output = model(x)
        assert output.shape == (batch_size, embed_dim)
        assert output.dtype == torch.float32
