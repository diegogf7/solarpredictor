"""Fetch SHARP cubes for the two-AR pilot.

Pair: 11429 (flaring, X5.4 producer) and 11431 (quiet), both 2012-03-10 and
both in the real train split. Same date on purpose -- they share full-disk SDO
frames, which halves the Surya download later.

HARP<->NOAA mapping is JSOC-verified; do not substitute invented HARP numbers,
an unknown HARPNUM returns an empty result with no error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.data.fetch import fetch_sharp_cube, stack_frames  # noqa: E402

PILOT = {
    "11429": {"harp": 1449, "role": "flaring"},
    "11431": {"harp": 1455, "role": "quiet"},
}


def fetch_with_retry(harp, tstart, tend, email, cadence, attempts=4):
    """JSOC drops connections under load; v1 lost a pilot run to exactly this."""
    for attempt in range(attempts):
        try:
            return fetch_sharp_cube(harp, tstart, tend, email, cadence=cadence)
        except Exception as error:  # noqa: BLE001 - network faults are varied
            wait = 20 * (attempt + 1)
            print(f"    attempt {attempt+1} failed ({type(error).__name__}: {error}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"JSOC fetch failed for HARP {harp} after {attempts} attempts")


def fetch_positions(harp, tstart, tend, email, cadence):
    """Flux-weighted AR centre per timestamp. Keywords only, no images."""
    import drms

    client = drms.Client(email=email)
    t0 = tstart.replace("-", ".") + "_00:00:00_TAI"
    t1 = tend.replace("-", ".") + "_00:00:00_TAI"
    keys = client.query(
        f"hmi.sharp_cea_720s[{harp}][{t0}-{t1}@{cadence}]",
        key=["T_REC", "QUALITY", "LAT_FWT", "LON_FWT", "SIZE_ACR"],
    )
    rows = []
    for _, row in keys.iterrows():
        if not np.isfinite(row["LAT_FWT"]) or not np.isfinite(row["LON_FWT"]):
            continue
        if int(row["QUALITY"]) != 0:
            continue
        stamp = str(row["T_REC"])[:19].replace(".", "-", 2).replace("_", "T")
        rows.append(
            {
                "t": stamp,
                "lat": float(row["LAT_FWT"]),
                "lon": float(row["LON_FWT"]),
                "size": float(row["SIZE_ACR"]),
            }
        )
    if not rows:
        raise RuntimeError(f"HARP {harp}: no usable position rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="diego.gaf28@gmail.com")
    parser.add_argument("--tstart", default="2012-03-09")
    parser.add_argument("--tend", default="2012-03-11")
    parser.add_argument("--cadence", default="1h")
    parser.add_argument("--out", default="flare_forecaster/cache/pilot")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for noaa, meta in PILOT.items():
        target = out / f"noaa{noaa}_cube.npz"
        if target.exists():
            print(f"{noaa} ({meta['role']}): cached, skipping")
            continue

        print(f"{noaa} ({meta['role']}, HARP {meta['harp']}) {args.tstart} -> {args.tend}")
        started = time.perf_counter()
        timestamps, frames = fetch_with_retry(
            meta["harp"], args.tstart, args.tend, args.email, args.cadence
        )
        if not frames:
            raise RuntimeError(
                f"HARP {meta['harp']} returned no frames -- check the NOAA<->HARP "
                f"mapping and the date window"
            )
        cube = stack_frames(frames)
        np.savez_compressed(
            target, cube=cube, timestamps=np.array(timestamps), harp=meta["harp"]
        )
        elapsed = time.perf_counter() - started
        print(
            f"  {cube.shape} float32, {len(timestamps)} frames, "
            f"{target.stat().st_size/1e6:.1f} MB, {elapsed:.0f}s"
        )
        print(f"  {timestamps[0]} .. {timestamps[-1]}")
        manifest[noaa] = {
            "harp": meta["harp"],
            "role": meta["role"],
            "n_frames": len(timestamps),
            "shape": list(cube.shape),
            "t_first": timestamps[0],
            "t_last": timestamps[-1],
        }

    if manifest:
        (out / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2))

    # AR positions, fetched HERE rather than on the cluster. Compute nodes have
    # restricted egress and drms comes back with NaN keywords there.
    positions = {}
    for noaa, meta in PILOT.items():
        rows = fetch_positions(meta["harp"], args.tstart, args.tend, args.email, args.cadence)
        positions[noaa] = rows
        print(
            f"{noaa}: {len(rows)} positions, "
            f"lat {rows[0]['lat']:.1f} lon {rows[0]['lon']:.1f} -> "
            f"lat {rows[-1]['lat']:.1f} lon {rows[-1]['lon']:.1f}"
        )
    (out / "pilot_positions.json").write_text(json.dumps(positions, indent=2))
    print(f"\ncubes + positions in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
