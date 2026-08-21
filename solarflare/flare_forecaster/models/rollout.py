import torch
from torch import nn


@torch.no_grad()
def sample_next(
    deterministic: nn.Module,
    flow: nn.Module,
    history: torch.Tensor,
    delta_hours: torch.Tensor,
    history_valid: torch.Tensor,
    residual_scale: torch.Tensor,
    base_noise_scale: float = 1.0,
    n_flow_steps: int = 16,
) -> torch.Tensor:
    mu = deterministic(history, delta_hours)
    x = mu + base_noise_scale * residual_scale * torch.randn_like(mu)
    ds = 1.0 / n_flow_steps

    for index in range(n_flow_steps):
        s0 = torch.full(
            (x.shape[0],),
            index / n_flow_steps,
            device=x.device,
        )
        v0 = flow(x, s0, history, delta_hours, history_valid)
        x_euler = x + ds * v0

        s1 = torch.full(
            (x.shape[0],),
            (index + 1) / n_flow_steps,
            device=x.device,
        )
        v1 = flow(x_euler, s1, history, delta_hours, history_valid)
        x = x + 0.5 * ds * (v0 + v1)

    return x


@torch.no_grad()
def rollout(
    deterministic: nn.Module,
    flow: nn.Module,
    initial_history: torch.Tensor,
    initial_delta_hours: torch.Tensor,
    initial_valid: torch.Tensor,
    residual_scale: torch.Tensor,
    future_steps: int = 24,
    samples: int = 16,
    base_noise_scale: float = 1.0,
    n_flow_steps: int = 16,
) -> torch.Tensor:
    """Return [K,B,H,D]."""
    ensemble = []
    for _ in range(samples):
        history = initial_history.clone()
        delta = initial_delta_hours.clone()
        valid = initial_valid.clone()
        generated = []

        for _ in range(future_steps):
            next_z = sample_next(
                deterministic,
                flow,
                history,
                delta,
                valid,
                residual_scale,
                base_noise_scale=base_noise_scale,
                n_flow_steps=n_flow_steps,
            )
            generated.append(next_z)

            history = torch.cat(
                [history[:, 1:], next_z.unsqueeze(1)],
                dim=1,
            )
            next_delta = torch.ones_like(delta[:, :1])
            delta = torch.cat([delta[:, 1:], next_delta], dim=1)
            valid = torch.cat(
                [
                    valid[:, 1:],
                    torch.ones_like(valid[:, :1]),
                ],
                dim=1,
            )

        ensemble.append(torch.stack(generated, dim=1))

    result = torch.stack(ensemble, dim=0)
    assert result.ndim == 4
    assert torch.isfinite(result).all()
    return result


def summarize_rollouts(
    history: torch.Tensor,       # [B,L,D]
    history_valid: torch.Tensor, # [B,L]
    rollouts: torch.Tensor,      # [K,B,H,D]
) -> torch.Tensor:
    weight = history_valid.unsqueeze(-1).to(history.dtype)
    history_mean = (
        (history * weight).sum(1)
        / weight.sum(1).clamp_min(1)
    )

    trajectory_mean = rollouts.mean(dim=0)       # [B,H,D]
    trajectory_std = rollouts.std(dim=0)         # [B,H,D]
    future_mean = trajectory_mean.mean(dim=1)    # [B,D]
    future_std = trajectory_std.mean(dim=1)      # [B,D]
    final_mean = trajectory_mean[:, -1]          # [B,D]

    return torch.cat(
        [history_mean, future_mean, future_std, final_mean],
        dim=-1,
    )
