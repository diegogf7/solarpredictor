import torch
from torch import nn

class RolloutAwareFlareHead(nn.Module):
    def __init__(self, d_latent: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4 * d_latent, 256), 
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).squeeze(-1)
        assert torch.isfinite(logits).all()

        return logits

    