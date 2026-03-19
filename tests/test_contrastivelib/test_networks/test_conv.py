import torch

from src.contrastivelib.networks import ConvNet


class TestConvNet:
    GRID_DIM = 21
    N_STATES = 4

    def test_output_shape(self, device, batch_size, embed_dim):
        model = ConvNet(grid_dim=self.GRID_DIM, embed_dim=embed_dim).to(device)
        x = torch.randint(
            0, self.N_STATES, (batch_size, self.GRID_DIM, self.GRID_DIM), device=device
        )
        output = model(x)
        assert output.shape == (batch_size, embed_dim)
        assert output.dtype == torch.float32
