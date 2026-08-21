"""Measure real PINN reconstruction cost, then extrapolate the campaign.

Replaces the estimate borrowed from NF2 (~1.2 GPU-min/frame) with a number
measured on this hardware. Times one cold start plus N warm-started frames
and reports what a full campaign would cost.

    srun -p mi3001x -N 1 -n 1 -t 00:40:00 --pty \
        python scripts/05_bench_pinn.py --device cuda

Caveat worth remembering when reading the output: reconstruction.pinn.train
builds a fresh AdamW on every call, so a warm-started frame still takes an
optimizer kick. v1 traced α drift to exactly this. The timing is valid; the
physics needs the single-run + lr-decay fix before any real campaign.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.reconstruction.pinn import train  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402


def boundary_from_cube(cube: np.ndarray, index: int, stride: int) -> np.ndarray:
    """(n_frames, 3, H, W) with (Br, Bt, Bp) -> (h, w, 3) as (Bx, By, Bz).

    CEA -> Cartesian: Bx = Bp, By = -Bt, Bz = Br. The minus sign is not
    optional; v1 baked this convention into 01_pinn_smoke.py.
    """
    br, bt, bp = cube[index, 0], cube[index, 1], cube[index, 2]
    br, bt, bp = br[::stride, ::stride], bt[::stride, ::stride], bp[::stride, ::stride]
    return np.stack([bp, -bt, br], axis=-1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", default="flare_forecaster/cache/harp3894_cube.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps-cold", type=int, default=6000)
    parser.add_argument("--steps-warm", type=int, default=1500)
    parser.add_argument("--n-warm", type=int, default=6, help="warm frames to time")
    parser.add_argument("--stride", type=int, default=4, help="boundary downsample")
    parser.add_argument("--n-col", type=int, default=2048)
    # campaign shape
    parser.add_argument("--n-ars", type=int, default=400)
    parser.add_argument("--frames-per-ar", type=int, default=120)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    cube = np.load(args.cube)["cube"]
    n_frames = cube.shape[0]
    sample = boundary_from_cube(cube, 0, args.stride)
    print(
        f"device={device} | cube {cube.shape} -> boundary {sample.shape} "
        f"(stride {args.stride}) | n_col={args.n_col}"
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- cold start ------------------------------------------------------
    print(f"\ncold start ({args.steps_cold} steps)")
    sync()
    t0 = time.perf_counter()
    model, _ = train(
        sample,
        n_steps=args.steps_cold,
        n_col=args.n_col,
        device=str(device),
        model=None,
    )
    sync()
    cold_seconds = time.perf_counter() - t0
    print(f"  cold: {cold_seconds:.1f} s  ({cold_seconds/args.steps_cold*1000:.2f} ms/step)")

    # ---- warm frames -----------------------------------------------------
    print(f"\nwarm frames ({args.steps_warm} steps each)")
    warm_times: list[float] = []
    for i in range(args.n_warm):
        boundary = boundary_from_cube(cube, (i + 1) % n_frames, args.stride)
        sync()
        t0 = time.perf_counter()
        model, _ = train(
            boundary,
            n_steps=args.steps_warm,
            n_col=args.n_col,
            device=str(device),
            model=model,
        )
        sync()
        elapsed = time.perf_counter() - t0
        warm_times.append(elapsed)
        print(f"  frame {i+1}: {elapsed:.1f} s")

    warm_mean = float(np.mean(warm_times))
    warm_std = float(np.std(warm_times))
    print(f"\nwarm mean {warm_mean:.1f} s +/- {warm_std:.1f}  "
          f"({warm_mean/args.steps_warm*1000:.2f} ms/step)")

    # ---- campaign extrapolation -----------------------------------------
    per_ar_seconds = cold_seconds + (args.frames_per_ar - 1) * warm_mean
    total_hours = args.n_ars * per_ar_seconds / 3600.0
    print(
        f"\ncampaign: {args.n_ars} ARs x {args.frames_per_ar} frames"
        f"\n  per AR      {per_ar_seconds/60:.1f} min"
        f"\n  TOTAL       {total_hours:.0f} GPU-hours"
    )
    for frames in (48, 120):
        hours = args.n_ars * (cold_seconds + (frames - 1) * warm_mean) / 3600.0
        print(f"  ({frames:>3} frames/AR -> {hours:.0f} GPU-hours)")
    print(
        "\nNOTE: at stride "
        f"{args.stride} this is a downsampled boundary. Full resolution costs "
        f"more per step; scale accordingly before committing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
