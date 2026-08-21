import torch
from torch import nn

class DeterministicForecaster(nn.Module):

    def __init__(
            self,
            d_latent: int = 256,
            hidden: int = 256,
            layers: int = 2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size = d_latent + 1,
            hidden_size = hidden,
            num_layers = layers,
            batch_first = True,
            dropout = 0.10 if layers > 1 else 0.0
        )

        self.out = nn.Linear(hidden, d_latent)

    def forward(
            self,
            history: torch.Tensor,
            delta_hours: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([history, delta_hours.unsqueeze(-1)], dim = -1)
        h, _ = self.gru(x)
        prediction = self.out(h[:, -1])

        assert torch.isfinite(prediction).all()
        return prediction