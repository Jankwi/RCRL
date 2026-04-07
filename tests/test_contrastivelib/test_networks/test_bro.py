import torch

from src.contrastivelib.networks import BroNet


class TestBroNet:
    GRID_DIM = 21
    N_STATES = 4

    def test_output_shape(self, device_fx, batch_size, embed_dim):
        model = BroNet(input_dim=self.GRID_DIM**2, embed_dim=embed_dim).to(device_fx)
        x = torch.randint(
            0, self.N_STATES, (batch_size, self.GRID_DIM, self.GRID_DIM), device=device_fx
        )
        output = model(x)
        assert output.shape == (batch_size, embed_dim)
        assert output.dtype == torch.float32
