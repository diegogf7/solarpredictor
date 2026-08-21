"""The eight-channel analytic readout stack (spec sec 9).

Everything is expressed on the footpoint plane at 128x128, because twist is a
property of a field line and its natural index is where the line meets the
photosphere.

Channels, in PHYSICS_CHANNELS order:
    alpha              (curl B . B) / |B|^2          pointwise, autodiff
    curvature          |(b_hat . grad) b_hat|        pointwise, autodiff
    twist              (1/4pi) integral alpha dl     requires tracing
    int_jz             height-integrated |J_z|       quadrature along z
    b_mag              |B| at the footpoint          free
    closure_error      |Tw(up) - Tw(down)|           free byproduct of tracing
    trace_valid        binary                        explicit, never implied
    strong_field_mask  binary                        explicit, never implied

Masked pixels carry a sentinel plus their mask channel. Nothing is silently
zero-filled: the encoder is shown what was invalid rather than being handed a
zero it cannot distinguish from a real measurement.
"""

from __future__ import annotations

import torch

from ..contracts import N_PHYSICS_CHANNELS, PHYSICS_CHANNELS
from ..reconstruction.field import curl
from .tracing import trace_field_lines

SENTINEL = 0.0


def _grid(resolution: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, resolution, device=device),
        torch.linspace(0, 1, resolution, device=device),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)


def _b_and_curl(model, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = points.clone().requires_grad_(True)
    b = model(points)
    c = curl(b, points)
    return b, c


def alpha_at_points(model, points: torch.Tensor) -> torch.Tensor:
    b, c = _b_and_curl(model, points)
    numerator = (c * b).sum(dim=-1)
    denominator = (b * b).sum(dim=-1).clamp_min(1e-8)
    return (numerator / denominator).detach()


def curvature_at_points(model, points: torch.Tensor) -> torch.Tensor:
    """|(b_hat . grad) b_hat| by autodiff through the unit vector field."""
    points = points.clone().requires_grad_(True)
    b = model(points)
    unit = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    directional = torch.zeros_like(unit)
    for component in range(3):
        grad = torch.autograd.grad(
            unit[:, component].sum(), points, create_graph=False, retain_graph=True
        )[0]
        directional[:, component] = (unit * grad).sum(dim=-1)
    return directional.norm(dim=-1).detach()


def integrated_jz(model, xy: torch.Tensor, n_z: int, z_max: float) -> torch.Tensor:
    """Trapezoidal integral of |J_z| along z above each footpoint."""
    heights = torch.linspace(0.0, z_max, n_z, device=xy.device)
    total = torch.zeros(xy.shape[0], device=xy.device)
    previous = None
    for index, z in enumerate(heights):
        points = torch.cat([xy, z.expand(xy.shape[0], 1)], dim=1)
        _, c = _b_and_curl(model, points)
        jz = c[:, 2].abs().detach()
        if previous is not None:
            total = total + 0.5 * (jz + previous) * (heights[index] - heights[index - 1])
        previous = jz
    return total


@torch.no_grad()
def _noop() -> None:
    return None


def build_readout(
    model,
    resolution: int = 128,
    n_z: int = 16,
    z_max: float = 1.0,
    b_floor_fraction: float = 0.1,
    trace_kwargs: dict | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict]:
    """-> (readout [C,H,W], diagnostics).

    `model` is the trained PINN, queried as model(points[N,3]) -> B[N,3] in
    normalized box coordinates.
    """
    device = device or next(model.parameters()).device
    trace_kwargs = dict(trace_kwargs or {})

    xy = _grid(resolution, device)
    footpoints = torch.cat([xy, torch.zeros(xy.shape[0], 1, device=device)], dim=1)

    b_foot, curl_foot = _b_and_curl(model, footpoints)
    b_magnitude = b_foot.norm(dim=-1).detach()
    alpha = (
        (curl_foot * b_foot).sum(dim=-1) / (b_foot * b_foot).sum(dim=-1).clamp_min(1e-8)
    ).detach()
    curvature = curvature_at_points(model, footpoints)
    int_jz = integrated_jz(model, xy, n_z=n_z, z_max=z_max)

    strong = b_magnitude > b_floor_fraction * b_magnitude.max()

    def field(points: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return model(points).detach()

    def alpha_fn(points: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return alpha_at_points(model, points)

    seeds = footpoints + torch.tensor([0.0, 0.0, 1e-3], device=device)
    up = trace_field_lines(field, alpha_fn, seeds, direction=+1.0, **trace_kwargs)
    down = trace_field_lines(field, alpha_fn, seeds, direction=-1.0, **trace_kwargs)

    twist_up = up.alpha_integral / (4.0 * torch.pi)
    twist_down = down.alpha_integral / (4.0 * torch.pi)
    twist = 0.5 * (twist_up + twist_down)
    # Tracing from both ends and differencing gives a per-pixel error
    # estimate for free. The encoder sees it; it also drives the mask.
    closure_error = (twist_up - twist_down).abs()
    trace_valid = up.valid & down.valid

    usable = strong & trace_valid
    channels = {
        "alpha": alpha,
        "curvature": curvature,
        "twist": twist,
        "int_jz": int_jz,
        "b_mag": b_magnitude,
        "closure_error": closure_error,
        "trace_valid": trace_valid.float(),
        "strong_field_mask": strong.float(),
    }
    for name in ("alpha", "curvature", "twist", "closure_error"):
        channels[name] = torch.where(
            usable, channels[name], torch.full_like(channels[name], SENTINEL)
        )
    for name in ("int_jz", "b_mag"):
        channels[name] = torch.where(
            strong, channels[name], torch.full_like(channels[name], SENTINEL)
        )

    readout = torch.stack(
        [channels[name].reshape(resolution, resolution) for name in PHYSICS_CHANNELS]
    )
    assert readout.shape == (N_PHYSICS_CHANNELS, resolution, resolution)

    diagnostics = {
        "strong_fraction": float(strong.float().mean()),
        "trace_valid_fraction": float(trace_valid.float().mean()),
        "usable_fraction": float(usable.float().mean()),
        "closure_error_median": float(closure_error[usable].median()) if bool(usable.any()) else float("nan"),
        "mean_arc_length": float(up.arc_length.mean()),
        "exit_domain": int((up.exit_reason == 1).sum()),
        "exit_weak": int((up.exit_reason == 2).sum()),
        "exit_maxlen": int((up.exit_reason == 3).sum()),
        "nonfinite": int((~torch.isfinite(readout)).sum()),
    }
    return readout, diagnostics


def robust_scale(
    readout: torch.Tensor,
    median: torch.Tensor,
    iqr: torch.Tensor,
    clip: float = 10.0,
) -> torch.Tensor:
    """Per-channel median/IQR scaling. Mask channels pass through untouched."""
    scaled = (readout - median[:, None, None]) / iqr[:, None, None].clamp_min(1e-9)
    scaled = scaled.clamp(-clip, clip)
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=clip, neginf=-clip)
    for name in ("trace_valid", "strong_field_mask"):
        index = PHYSICS_CHANNELS.index(name)
        scaled[index] = readout[index]
    return scaled
