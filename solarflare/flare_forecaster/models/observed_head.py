import torch
from torch import nn

class ObservedHistoryHead(nn.Module):

    def __init__(self, d_fused: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_fused, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1)
        )

    def forward(
            self,
            history: torch.Tensor,
            valid: torch.Tensor,
    ) -> torch.Tensor:
        weight = valid.unsqueeze(-1).to(history.dtype)
        pooled = (history * weight).sum(1) / weight.sum(1).clamp_min(1)
        return self.net(pooled).squeeze(-1)
