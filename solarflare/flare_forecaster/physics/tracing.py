"""Adaptive-step field-line tracing.

Dormand-Prince 5(4) with per-line step control, written out rather than
pulled from torchdiffeq so the termination logic (domain exit, weak field,
max arc length) can act per-line inside the integrator instead of after it.

Fixed-step tracing is deliberately not offered: step error accumulates along
the line and shows up as a bias in the twist integral, which is exactly the
quantity the readout stack exists to measure.
"""

from __future__ import annotations

from typing import Callable

import torch

# Dormand-Prince 5(4) tableau
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B4 = (
    5179 / 57600,
    0.0,
    7571 / 16695,
    393 / 640,
    -92097 / 339200,
    187 / 2100,
    1 / 40,
)


class TraceResult:
    """Per-line outcome of one trace."""

    def __init__(
        self,
        end_point: torch.Tensor,
        arc_length: torch.Tensor,
        alpha_integral: torch.Tensor,
        valid: torch.Tensor,
        exit_reason: torch.Tensor,
    ) -> None:
        self.end_point = end_point
        self.arc_length = arc_length
        self.alpha_integral = alpha_integral
        self.valid = valid
        self.exit_reason = exit_reason  # 0 running, 1 domain, 2 weak, 3 maxlen


def trace_field_lines(
    b_field: Callable[[torch.Tensor], torch.Tensor],
    alpha_at: Callable[[torch.Tensor], torch.Tensor] | None,
    seeds: torch.Tensor,
    direction: float = 1.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    initial_step: float = 1e-2,
    max_step: float = 5e-2,
    max_arc_length: float = 4.0,
    max_iterations: int = 2000,
    b_floor: float = 1e-6,
    domain_min: float = 0.0,
    domain_max: float = 1.0,
) -> TraceResult:
    """Integrate dr/dl = +/- B/|B| from `seeds` in normalized box coordinates.

    Accumulates the twist integral along the way when `alpha_at` is given, so
    alpha is sampled on the adaptive grid the integrator actually visits
    rather than re-interpolated afterwards.

    seeds: [N,3]. Returns per-line end points, arc lengths, integral of alpha
    dl, a validity mask, and why each line stopped.
    """
    device = seeds.device
    n_lines = seeds.shape[0]

    position = seeds.clone()
    arc_length = torch.zeros(n_lines, device=device)
    integral = torch.zeros(n_lines, device=device)
    step = torch.full((n_lines,), initial_step, device=device)
    active = torch.ones(n_lines, dtype=torch.bool, device=device)
    exit_reason = torch.zeros(n_lines, dtype=torch.int8, device=device)

    def tangent(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = b_field(points)
        magnitude = b.norm(dim=-1, keepdim=True)
        unit = direction * b / magnitude.clamp_min(b_floor)
        return unit, magnitude.squeeze(-1)

    for _ in range(max_iterations):
        if not bool(active.any()):
            break

        idx = active.nonzero(as_tuple=True)[0]
        p = position[idx]
        h = step[idx].unsqueeze(-1)

        stages: list[torch.Tensor] = []
        alpha_stages: list[torch.Tensor] = []
        for stage in range(7):
            offset = torch.zeros_like(p)
            for j, coefficient in enumerate(_A[stage]):
                if coefficient != 0.0:
                    offset = offset + coefficient * stages[j]
            k, _ = tangent(p + h * offset)
            stages.append(k)
            if alpha_at is not None and stage in (0, 6):
                alpha_stages.append(alpha_at(p + h * offset))

        high = torch.zeros_like(p)
        low = torch.zeros_like(p)
        for j in range(7):
            if _B5[j] != 0.0:
                high = high + _B5[j] * stages[j]
            if _B4[j] != 0.0:
                low = low + _B4[j] * stages[j]

        candidate = p + h * high
        error = (h * (high - low)).norm(dim=-1)
        tolerance = atol + rtol * candidate.norm(dim=-1)
        accept = error <= tolerance

        # PI-free step control; 0.9 safety, clamped growth
        scale = 0.9 * (tolerance / error.clamp_min(1e-30)).pow(0.2)
        scale = scale.clamp(0.2, 5.0)
        new_step = (step[idx] * scale).clamp(max=max_step)

        accepted = idx[accept]
        if accepted.numel():
            taken = step[accepted]
            position[accepted] = candidate[accept]
            arc_length[accepted] = arc_length[accepted] + taken
            if alpha_at is not None:
                mean_alpha = 0.5 * (alpha_stages[0][accept] + alpha_stages[1][accept])
                integral[accepted] = integral[accepted] + mean_alpha * taken
        step[idx] = new_step

        # termination, evaluated only on lines that moved
        if accepted.numel():
            q = position[accepted]
            outside = (
                (q < domain_min).any(dim=-1) | (q > domain_max).any(dim=-1)
            )
            _, magnitude = tangent(q)
            weak = magnitude < b_floor
            too_long = arc_length[accepted] >= max_arc_length

            exit_reason[accepted[outside]] = 1
            exit_reason[accepted[weak & ~outside]] = 2
            exit_reason[accepted[too_long & ~outside & ~weak]] = 3
            active[accepted[outside | weak | too_long]] = False

    # A line still active at the iteration cap never closed
    exit_reason[active] = 3
    valid = exit_reason == 1  # reached the boundary: a genuinely closed trace

    return TraceResult(position, arc_length, integral, valid, exit_reason)
