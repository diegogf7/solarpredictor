"""Stream SDO frames from S3 -> frozen Surya -> AR-pooled latents.

Frames are never kept: each is downloaded, encoded, and deleted. A full-disk
frame is 587 MB; a pooled latent is 2.5 KB in fp16. That ratio is the whole
reason this is tractable.

Normalization follows surya/datasets/helio.py exactly -- signum-log then
standardize, per channel:
    val = raw * sl_scale_factor
    val = sign(val) * log1p(|val|)
    out = (val - mean) / (std + epsilon)
Getting this wrong yields finite but meaningless latents, so it is taken from
the released scalers.yaml rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.encoders.surya_encoder import SuryaEncoder  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402

S3_ROOT = "https://nasa-surya-bench.s3.amazonaws.com"
ARCSEC_PER_PIXEL = 0.6  # SDO 4096 plate scale
DISK_CENTER = 2048.0


def signum_log_normalize(
    data: np.ndarray, scalers: dict, channels: list[str]
) -> np.ndarray:
    out = np.empty_like(data, dtype=np.float32)
    for index, name in enumerate(channels):
        s = scalers[name]
        val = data[index] * float(s["sl_scale_factor"])
        val = np.sign(val) * np.log1p(np.abs(val))
        out[index] = (val - float(s["mean"])) / (float(s["std"]) + float(s["epsilon"]))
    return out


def heliographic_to_pixel(lat_deg: float, lon_deg: float, when: datetime):
    """Stonyhurst (lat, lon) -> full-disk (x, y) pixel. Returns None if behind."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, get_earth

    observer = get_earth(when)
    point = SkyCoord(
        lon_deg * u.deg,
        lat_deg * u.deg,
        frame=HeliographicStonyhurst,
        obstime=when,
        observer=observer,
    )
    projected = point.transform_to(Helioprojective(observer=observer, obstime=when))
    tx = projected.Tx.to_value(u.arcsec)
    ty = projected.Ty.to_value(u.arcsec)
    if not (np.isfinite(tx) and np.isfinite(ty)):
        return None
    return DISK_CENTER + tx / ARCSEC_PER_PIXEL, DISK_CENTER - ty / ARCSEC_PER_PIXEL


def load_positions(path: Path) -> dict:
    """Positions are fetched on the laptop by 06 and shipped here as JSON.

    Compute nodes have restricted egress; drms returns NaN keywords from them
    while the identical query succeeds off-cluster. Keeping JSOC out of the
    GPU job also means a network hiccup can't kill a running allocation.
    """
    raw = json.loads(path.read_text())
    parsed = {}
    for noaa, rows in raw.items():
        parsed[noaa] = [
            {
                "t": datetime.fromisoformat(r["t"]).replace(tzinfo=timezone.utc),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            }
            for r in rows
        ]
    return parsed


def frame_url(when: datetime) -> str:
    return (
        f"{S3_ROOT}/{when.year:04d}/{when.month:02d}/"
        f"{when.year:04d}{when.month:02d}{when.day:02d}_"
        f"{when.hour:02d}{when.minute:02d}.nc"
    )


def download(url: str, target: Path, attempts: int = 3) -> bool:
    import urllib.error
    import urllib.request

    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(url, target)
            return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--cubes", default="flare_forecaster/cache/pilot")
    parser.add_argument("--out", default="artifacts_pilot/caches/pilot_surya.pt")
    parser.add_argument("--scratch", default=None, help="frame staging dir")
    parser.add_argument("--email", default="diego.gaf28@gmail.com")
    parser.add_argument("--n-frames", type=int, default=25)
    parser.add_argument("--box-arcsec", type=float, default=200.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-budget-s", type=float, default=7200)
    args = parser.parse_args()

    import xarray as xr

    set_seed(0)
    device = torch.device(args.device)
    scratch = Path(args.scratch or Path(args.out).parent / "frames")
    scratch.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights_dir)
    scalers = yaml.safe_load((weights / "scalers.yaml").read_text())
    encoder = SuryaEncoder(weights, pool="crop_mean", device=device)
    encoder.model.to(torch.bfloat16)
    channels = list(encoder.channels)
    print(f"surya ready | {len(channels)} channels | grid {encoder.token_grid}^2")

    manifest = json.loads((Path(args.cubes) / "pilot_manifest.json").read_text())
    started = time.perf_counter()

    # ---- pick the shared timeline -----------------------------------------
    positions = load_positions(Path(args.cubes) / "pilot_positions.json")
    for noaa, rows in positions.items():
        print(f"{noaa}: {len(rows)} positions, "
              f"lat {rows[0]['lat']:.1f} lon {rows[0]['lon']:.1f} -> "
              f"lat {rows[-1]['lat']:.1f} lon {rows[-1]['lon']:.1f}")

    common = sorted(
        set.intersection(*({r["t"].replace(minute=0) for r in rows}
                           for rows in positions.values()))
    )[-args.n_frames:]
    print(f"\nshared timeline: {len(common)} hourly frames "
          f"{common[0]} .. {common[-1]}")

    latents = {noaa: [] for noaa in manifest}
    times: list[float] = []
    timings = {"download": 0.0, "encode": 0.0}
    half = args.box_arcsec / ARCSEC_PER_PIXEL / 2.0

    for step, when in enumerate(common):
        if time.perf_counter() - started > args.time_budget_s:
            print(f"\nTIME BUDGET reached at frame {step}; stopping early")
            break

        url = frame_url(when)
        target = scratch / Path(url).name
        t0 = time.perf_counter()
        if not download(url, target):
            print(f"  {when} MISSING, skipped")
            continue
        timings["download"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        with xr.open_dataset(target) as ds:
            raw = np.stack([ds[name].to_numpy() for name in channels])
        normalized = signum_log_normalize(raw, scalers, channels)
        del raw

        # Surya wants two timesteps (t-60, t). With hourly frames the previous
        # frame is exactly t-60, so reuse it; the first step doubles itself.
        current = torch.from_numpy(normalized).to(device=device, dtype=torch.bfloat16)
        if step == 0 or "previous" not in dir():
            previous = current
        ts = torch.stack([previous, current], dim=1).unsqueeze(0)  # [1,C,T,H,W]
        dt = torch.tensor([[-60.0, 0.0]], device=device, dtype=torch.bfloat16)

        boxes = []
        for noaa in manifest:
            row = min(positions[noaa], key=lambda r: abs((r["t"] - when).total_seconds()))
            pixel = heliographic_to_pixel(row["lat"], row["lon"], when)
            if pixel is None:
                boxes.append(None)
                continue
            cx, cy = pixel
            boxes.append([cx - half, cy - half, cx + half, cy + half])

        for noaa, box in zip(manifest, boxes):
            if box is None:
                latents[noaa].append(torch.zeros(encoder.embed_dim))
                continue
            box_t = torch.tensor([box], device=device)
            mask = encoder.box_to_token_mask(box_t, margin_px=0)
            pooled = encoder(ts, dt, mask).float().squeeze(0).cpu()
            latents[noaa].append(pooled)
            if step == 0:
                print(f"  {noaa}: box {[round(v) for v in box]} -> "
                      f"{int(mask.sum())} tokens")

        previous = current
        times.append(when.timestamp())
        timings["encode"] += time.perf_counter() - t0
        target.unlink()
        print(f"  [{step+1}/{len(common)}] {when} ok "
              f"(dl {timings['download']:.0f}s enc {timings['encode']:.0f}s)")

    payload = {
        "noaa": list(manifest.keys()),
        "z_surya": {k: torch.stack(v) for k, v in latents.items() if v},
        "t": torch.tensor(times, dtype=torch.float64),
        "timings": timings,
        "box_arcsec": args.box_arcsec,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    for noaa, z in payload["z_surya"].items():
        print(f"\n{noaa}: z_surya {tuple(z.shape)} finite={bool(torch.isfinite(z).all())} "
              f"std={float(z.std()):.4f} "
              f"frame-to-frame delta={float((z[1:]-z[:-1]).norm(dim=1).mean()):.4f}")
    print(f"\nwrote {out} | download {timings['download']:.0f}s encode {timings['encode']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
