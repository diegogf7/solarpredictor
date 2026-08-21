import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class PhysicsEncoder(nn.Module):
    def __init__(self, in_ch: int = 8, d_out: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_ch, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        assert torch.isfinite(x).all()
        h = self.features(x)
        h = self.pool(h).flatten(1)
        z = self.proj(h)
        assert z.ndim == 2
        assert torch.isfinite(z).all()
        return z
