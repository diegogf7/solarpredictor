"""Temporary decoder for Stage A. Discarded at inference."""

import torch
from torch import nn


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class PhysicsDecoder(nn.Module):
    def __init__(self, d_in: int = 256, out_ch: int = 8, base: int = 8):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(d_in, 256 * base * base)
        self.net = nn.Sequential(
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 32),
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(-1, 256, self.base, self.base)
        x = self.net(h)
        assert torch.isfinite(x).all()
        return x


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Invalid pixels are never reconstructed against."""
    per_pixel = torch.nn.functional.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=1.0,
    )
    weight = valid.to(per_pixel.dtype)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)
