"""Synthetic shape + gradient-boundary check. No real data, no metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.config import Config, resolve_surya_dim
from flare_forecaster.checkpoints import freeze, assert_no_grads
from flare_forecaster.models.fusion import FusionMLP
from flare_forecaster.models.observed_head import ObservedHistoryHead
from flare_forecaster.models.deterministic import DeterministicForecaster
from flare_forecaster.models.flow_transformer import FlowMatchingTransformer
from flare_forecaster.models.flare_head import RolloutAwareFlareHead
from flare_forecaster.models.rollout import rollout, summarize_rollouts
from flare_forecaster.utils.seed import set_seed


class MockSurya(nn.Module):
    """Stands in for the frozen encoder so this runs with no weights."""

    def __init__(self, d_surya: int):
        super().__init__()
        self.d_surya = d_surya

    def forward(self, batch_size: int, length: int) -> torch.Tensor:
        return torch.randn(batch_size, length, self.d_surya)


def main() -> int:
    set_seed(0)
    cfg = resolve_surya_dim(Config(), 1280)
    B, L, H = 2, cfg.history_steps, cfg.forecast_steps
    K = 3  # small ensemble; rollout_samples_train is for real runs

    surya = MockSurya(cfg.d_surya)
    fusion = FusionMLP(cfg.d_surya, cfg.d_physics, cfg.fusion_hidden,
                       cfg.d_fused, cfg.fusion_dropout)
    obs_head = ObservedHistoryHead(cfg.d_fused)
    determ = DeterministicForecaster(cfg.d_fused, cfg.deterministic_hidden,
                                     cfg.deterministic_layers)
    flow = FlowMatchingTransformer(cfg.d_fused, cfg.flow_width, cfg.flow_layers,
                                   cfg.flow_heads, cfg.flow_ff, cfg.flow_dropout,
                                   max_history=L)
    flare_head = RolloutAwareFlareHead(cfg.d_fused)

    z_surya = surya(B, L)
    z_phys = torch.randn(B, L, cfg.d_physics)
    delta = torch.ones(B, L)
    valid = torch.ones(B, L, dtype=torch.bool)
    residual_scale = torch.ones(cfg.d_fused)

    def check(name, tensor, expected):
        got = tuple(tensor.shape)
        assert got == expected, f"{name}: {got} != {expected}"
        assert torch.isfinite(tensor).all(), f"{name}: non-finite"
        print(f"  ok  {name:<22} {got}")

    print("shapes")
    check("mock surya", z_surya, (B, L, cfg.d_surya))
    fused = fusion(z_surya, z_phys)
    check("fusion", fused, (B, L, cfg.d_fused))
    check("observed logit", obs_head(fused, valid), (B,))
    mu = determ(fused, delta)
    check("deterministic", mu, (B, cfg.d_fused))

    s = torch.rand(B)
    v = flow(mu, s, fused, delta, valid)
    check("flow velocity", v, (B, cfg.d_fused))

    ens = rollout(determ, flow, fused, delta, valid, residual_scale,
                  future_steps=H, samples=K)
    check("rollout", ens, (K, B, H, cfg.d_fused))
    summary = summarize_rollouts(fused, valid, ens)
    check("summary", summary, (B, 4 * cfg.d_fused))
    check("flare logit", flare_head(summary), (B,))

    print("\ngradient boundaries")
    # Stage B: fusion + obs head train, nothing upstream
    fusion.zero_grad(); obs_head.zero_grad()
    obs_head(fusion(z_surya, z_phys), valid).sum().backward()
    assert any(p.grad is not None for p in fusion.parameters())
    print("  ok  stage B reaches fusion")

    # Stage C: deterministic only, fusion frozen
    freeze(fusion)
    determ.zero_grad()
    determ(fusion(z_surya, z_phys).detach(), delta).sum().backward()
    assert_no_grads(fusion, "fusion")
    print("  ok  stage C leaves fusion untouched")

    # Stage D: flow only, deterministic frozen
    freeze(determ)
    flow.zero_grad()
    with torch.no_grad():
        mu = determ(fused, delta)
    x0 = mu + cfg.base_noise_scale * residual_scale * torch.randn_like(mu)
    x1 = torch.randn_like(mu)
    sm = torch.rand(B)
    xs = (1 - sm[:, None]) * x0 + sm[:, None] * x1
    loss = torch.nn.functional.mse_loss(
        flow(xs, sm, fused, delta, valid), x1 - x0)
    loss.backward()
    assert_no_grads(determ, "deterministic")
    assert any(p.grad is not None for p in flow.parameters())
    print("  ok  stage D leaves deterministic untouched")

    print("\nALL SHAPE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
