"""PINN reconstruction + 8-channel readout + physics latent, for the pilot ARs.

One reconstruction per AR at the final history time, per the pilot spec.
Stage A has never run on real data and two maps cannot train an encoder, so
the autoencoder here is fit on the pilot's own readouts and is IN-SAMPLE by
construction. It exists to show the encoder produces finite, non-constant
latents -- not to represent a trained physics representation.
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
from flare_forecaster.encoders.physics_decoder import PhysicsDecoder, masked_huber  # noqa: E402
from flare_forecaster.encoders.physics_encoder import PhysicsEncoder  # noqa: E402
from flare_forecaster.physics.readout import build_readout, robust_scale  # noqa: E402
from flare_forecaster.reconstruction.pinn import train  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402


def boundary_from_cube(cube: np.ndarray, index: int, stride: int) -> np.ndarray:
    """(n,3,H,W) as (Br,Bt,Bp) -> (h,w,3) as (Bx,By,Bz). Bx=Bp, By=-Bt, Bz=Br.

    SHARP CEA cutouts carry NaN outside the patch. Left in, a single NaN makes
    B_scale NaN and the whole reconstruction collapses -- silently, since the
    loss just prints nan. Off-patch pixels become zero field.
    """
    br, bt, bp = cube[index, 0], cube[index, 1], cube[index, 2]
    br, bt, bp = br[::stride, ::stride], bt[::stride, ::stride], bp[::stride, ::stride]
    boundary = np.stack([bp, -bt, br], axis=-1).astype(np.float32)
    return np.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0)


def pick_frame(cube: np.ndarray, max_nan_fraction: float = 0.25) -> int:
    """Latest frame whose NaN fraction is tolerable, else the cleanest one."""
    fractions = np.isnan(cube).mean(axis=(1, 2, 3))
    usable = np.nonzero(fractions <= max_nan_fraction)[0]
    if usable.size:
        return int(usable[-1])
    return int(fractions.argmin())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes", default="flare_forecaster/cache/pilot")
    parser.add_argument("--out", default="artifacts_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--n-col", type=int, default=2048)
    parser.add_argument("--ae-steps", type=int, default=300)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    cubes = Path(args.cubes)
    out = Path(args.out)
    (out / "caches").mkdir(parents=True, exist_ok=True)

    manifest = json.loads((cubes / "pilot_manifest.json").read_text())
    timings: dict[str, float] = {}
    readouts: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict] = {}

    for noaa, meta in manifest.items():
        data = np.load(cubes / f"noaa{noaa}_cube.npz")
        cube, timestamps = data["cube"], data["timestamps"]
        final = pick_frame(cube)
        nan_fraction = float(np.isnan(cube[final]).mean())
        boundary = boundary_from_cube(cube, final, args.stride)
        print(
            f"\n=== NOAA {noaa} ({meta['role']}) === frame {final}/{len(timestamps)-1}"
            f" {timestamps[final]}  boundary {boundary.shape}"
            f"  NaN {100*nan_fraction:.1f}%"
        )
        if not np.isfinite(boundary).all() or float(np.abs(boundary).max()) == 0.0:
            raise RuntimeError(f"NOAA {noaa}: boundary is unusable after sanitization")

        started = time.perf_counter()
        model, b_scale = train(
            boundary,
            n_steps=args.steps,
            n_col=args.n_col,
            device=str(device),
            model=None,
        )
        timings[f"pinn_{noaa}"] = time.perf_counter() - started
        print(f"  PINN {timings[f'pinn_{noaa}']:.0f}s  B_scale={b_scale:.0f} G")

        started = time.perf_counter()
        readout, diag = build_readout(
            model,
            resolution=args.resolution,
            n_z=16,
            trace_kwargs=dict(max_iterations=400, max_arc_length=3.0),
            device=device,
        )
        timings[f"readout_{noaa}"] = time.perf_counter() - started
        readouts[noaa] = readout.cpu()
        diagnostics[noaa] = diag
        print(f"  readout {tuple(readout.shape)} in {timings[f'readout_{noaa}']:.0f}s")
        for key, value in diag.items():
            print(f"    {key:<24} {value}")
        for i, name in enumerate(PHYSICS_CHANNELS):
            ch = readout[i]
            print(
                f"    {name:<18} min {float(ch.min()):>8.3f} max {float(ch.max()):>8.3f} "
                f"mean {float(ch.mean()):>8.3f}"
            )

    # --- shared robust scaler (pilot-only; real runs fit on train ARs) ----
    stacked = torch.stack(list(readouts.values()))
    flat = stacked.permute(1, 0, 2, 3).reshape(len(PHYSICS_CHANNELS), -1)
    median = flat.median(dim=1).values
    iqr = (flat.quantile(0.75, dim=1) - flat.quantile(0.25, dim=1)).clamp_min(1e-9)
    scaled = torch.stack([robust_scale(r, median, iqr) for r in readouts.values()])
    print(f"\nscaled stack {tuple(scaled.shape)}  finite={bool(torch.isfinite(scaled).all())}")

    # --- in-sample autoencoder fit ---------------------------------------
    encoder = PhysicsEncoder(len(PHYSICS_CHANNELS), 256).to(device)
    decoder = PhysicsDecoder(256, len(PHYSICS_CHANNELS)).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4
    )
    valid_index = PHYSICS_CHANNELS.index("trace_valid")
    x = scaled.to(device)
    valid = (x[:, valid_index : valid_index + 1] > 0).expand_as(x)

    started = time.perf_counter()
    print(f"\nStage A (IN-SAMPLE on {x.shape[0]} maps -- plumbing only)")
    for step in range(args.ae_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_huber(decoder(encoder(x)), x, valid)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.ae_steps - 1:
            print(f"  step {step:>4}  masked_huber {float(loss):.5f}")
    timings["autoencoder"] = time.perf_counter() - started

    encoder.eval()
    with torch.no_grad():
        latents = encoder(x).cpu()
    print(
        f"\nphysics latents {tuple(latents.shape)}  finite={bool(torch.isfinite(latents).all())}"
        f"  std={float(latents.std()):.4f}"
    )
    separation = float((latents[0] - latents[1]).norm())
    print(f"latent distance between the two ARs: {separation:.4f}")

    payload = {
        "noaa": list(manifest.keys()),
        "readout": stacked,
        "readout_scaled": scaled,
        "z_physics": latents,
        "scaler_median": median,
        "scaler_iqr": iqr,
        "diagnostics": diagnostics,
        "timings": timings,
    }
    torch.save(payload, out / "caches" / "pilot_physics.pt")
    print(f"\nwrote {out/'caches'/'pilot_physics.pt'}")
    print("timings: " + "  ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
