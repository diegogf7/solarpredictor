"""PINN reconstruction + 8-channel readout for every campaign frame.

Warm-started along each AR's time series with a PERSISTENT optimizer: the
model and the AdamW state both carry from frame to frame, and the learning
rate decays within each frame. Rebuilding the optimizer per frame is what
produced v1's alpha drift and the pilot's rising loss.

Every N frames a cold-start reconstruction runs at the same timestamp and the
divergence is logged. If cold-vs-warm divergence trends upward across an AR's
lifetime, the features carry a slow systematic the dynamics model would
happily latch onto. The v2 spec calls this a required artifact, not optional QA.

Resumable per AR. Readouts are stored float16 (~256 KB/frame).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.contracts import PHYSICS_CHANNELS  # noqa: E402
from flare_forecaster.physics.readout import build_readout  # noqa: E402
from flare_forecaster.reconstruction.pinn import train  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402


def boundary_from_cube(cube: np.ndarray, index: int) -> np.ndarray:
    """(n,3,H,W) as (Br,Bt,Bp) -> (H,W,3) as (Bx,By,Bz), NaN zeroed.

    Bx = Bp, By = -Bt, Bz = Br. The minus sign is the CEA->Cartesian
    convention and is not optional.
    """
    br, bt, bp = cube[index, 0], cube[index, 1], cube[index, 2]
    boundary = np.stack([bp, -bt, br], axis=-1).astype(np.float32)
    return np.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps-cold", type=int, default=4000)
    parser.add_argument("--steps-warm", type=int, default=1200)
    parser.add_argument("--n-col", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--cold-check-every", type=int, default=8)
    parser.add_argument("--time-budget-s", type=float, default=27000)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    cubes = Path(args.cubes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((cubes / "manifest.json").read_text())
    drift_log_path = out / "cold_warm_divergence.json"
    drift_log = json.loads(drift_log_path.read_text()) if drift_log_path.exists() else {}

    started = time.perf_counter()
    todo = [n for n in sorted(manifest) if not (out / f"noaa{n}.pt").exists()]
    print(f"{len(manifest)} ARs total | {len(todo)} remaining | device {device}")

    for index, noaa in enumerate(todo):
        if time.perf_counter() - started > args.time_budget_s:
            print("\nTIME BUDGET reached; stopping cleanly")
            break

        data = np.load(cubes / f"noaa{noaa}.npz")
        cube, timestamps = data["cube"], data["timestamps"]
        label = int(data["label"])
        ar_started = time.perf_counter()

        model = optimizer = None
        readouts, kept, diagnostics = [], [], []
        drift = []

        for frame in range(len(timestamps)):
            boundary = boundary_from_cube(cube, frame)
            if float(np.abs(boundary).max()) == 0.0:
                continue

            n_steps = args.steps_cold if model is None else args.steps_warm
            try:
                model, b_scale, optimizer = train(
                    boundary,
                    n_steps=n_steps,
                    n_col=args.n_col,
                    device=str(device),
                    model=model,
                    optimizer=optimizer,
                    return_optimizer=True,
                    verbose=False,
                )
            except (FloatingPointError, ValueError) as error:
                print(f"    frame {frame} failed: {error}")
                model = optimizer = None
                continue

            readout, diag = build_readout(
                model,
                resolution=args.resolution,
                n_z=16,
                trace_kwargs=dict(max_iterations=400, max_arc_length=3.0),
                device=device,
            )
            readouts.append(readout.cpu().to(torch.float16))
            kept.append(str(timestamps[frame]))
            diagnostics.append(diag)

            # cold-start control at the same timestamp
            if frame > 0 and frame % args.cold_check_every == 0:
                cold_model, _ = train(
                    boundary, n_steps=args.steps_cold, n_col=args.n_col,
                    device=str(device), model=None, verbose=False,
                )
                cold_readout, _ = build_readout(
                    cold_model, resolution=args.resolution, n_z=16,
                    trace_kwargs=dict(max_iterations=400, max_arc_length=3.0),
                    device=device,
                )
                alpha_index = PHYSICS_CHANNELS.index("alpha")
                strong_index = PHYSICS_CHANNELS.index("strong_field_mask")
                # Restrict to pixels usable in BOTH reconstructions. Over the
                # full map the median is dominated by sentinel zeros and comes
                # back 0.0 no matter how far the two actually diverged.
                usable = (readout[strong_index] > 0) & (cold_readout[strong_index] > 0)
                if bool(usable.any()):
                    delta = (readout[alpha_index] - cold_readout[alpha_index]).abs()[usable]
                    scale = readout[alpha_index][usable].abs().median().clamp_min(1e-8)
                    divergence = float(delta.median())
                    relative = float(delta.median() / scale)
                else:
                    divergence = relative = float("nan")
                drift.append(
                    {
                        "frame": frame,
                        "alpha_divergence": divergence,
                        "alpha_divergence_relative": relative,
                        "usable_pixels": int(usable.sum()),
                    }
                )
                del cold_model, cold_readout

        if not readouts:
            print(f"[{index+1}/{len(todo)}] {noaa}: no usable frames")
            continue

        torch.save(
            {
                "readout": torch.stack(readouts),
                "timestamps": kept,
                "label": label,
                "harp": manifest[noaa]["harp"],
                "diagnostics": diagnostics,
            },
            out / f"noaa{noaa}.pt",
        )
        if drift:
            drift_log[noaa] = drift
            drift_log_path.write_text(json.dumps(drift_log, indent=2))

        strong = float(np.mean([d["strong_fraction"] for d in diagnostics]))
        valid = float(np.mean([d["trace_valid_fraction"] for d in diagnostics]))
        elapsed = time.perf_counter() - ar_started
        rate = (time.perf_counter() - started) / (index + 1)
        print(
            f"[{index+1}/{len(todo)}] {noaa} label {label} | {len(readouts)} frames "
            f"| strong {strong:.2f} valid {valid:.2f} | {elapsed:.0f}s "
            f"| ETA {(len(todo)-index-1)*rate/3600:.1f}h"
            + (f" | drift {drift[-1]['alpha_divergence']:.4f}" if drift else "")
        )

    produced = sorted(out.glob("noaa*.pt"))
    frames = 0
    for path in produced:
        frames += len(torch.load(path, map_location="cpu", weights_only=False)["timestamps"])
    print(f"\n{len(produced)} ARs, {frames} readout frames, "
          f"{sum(p.stat().st_size for p in produced)/1e9:.2f} GB")
    print(f"{(time.perf_counter()-started)/3600:.2f}h elapsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
