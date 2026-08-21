"""Fetch SHARP cubes for the PINN campaign. Laptop-side (JSOC needs egress).

Downsampled at write time: the reconstruction samples n_col collocation points
per step regardless of boundary size, so full resolution costs transfer without
buying accuracy at stride 4. This drops ~30 GB to ~2 GB.

NaN is left in place here and sanitized at reconstruction time, so the raw
masked fraction stays visible for frame selection.

Resumable per AR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def stack_frames(frames):
    """Center-crop to the smallest common shape, then stack.

    Copied from data.fetch rather than imported: that module pulls in
    sunpy.net at import time, which drags in bs4/zeep and is not needed here.
    """
    height = min(f.shape[1] for f in frames)
    width = min(f.shape[2] for f in frames)
    cropped = []
    for f in frames:
        top = (f.shape[1] - height) // 2
        left = (f.shape[2] - width) // 2
        cropped.append(f[:, top : top + height, left : left + width])
    return np.stack(cropped)


def fetch_cube_parallel(harp, tstart, tend, email, cadence, workers=8, allowed=None):
    """Concurrent FITS segment reads. Each frame is 3 separate HTTP fetches
    (Br, Bt, Bp); serializing them is what made this ~45 s per AR.

    QUALITY is deliberately not used as a filter. The cluster's JSOC access
    returns NaN for that keyword (segment URLs and FITS reads work fine), so
    quality filtering happens once, off-cluster, when the positions file is
    built -- and `allowed` replays that decision here.
    """
    from concurrent.futures import ThreadPoolExecutor

    import drms
    from astropy.io import fits

    client = drms.Client(email=email)
    t0 = tstart[:10].replace("-", ".") + "_00:00:00_TAI"
    t1 = tend[:10].replace("-", ".") + "_00:00:00_TAI"
    keys, segments = client.query(
        f"hmi.sharp_cea_720s[{harp}][{t0}-{t1}@{cadence}]",
        key=["T_REC", "QUALITY"],
        seg=["Br", "Bt", "Bp"],
    )

    jobs = []
    for i, row in keys.iterrows():
        stamp = str(row["T_REC"])[:19].replace(".", "-", 2).replace("_", "T")
        if allowed is not None:
            if stamp not in allowed:
                continue
        else:
            quality = row["QUALITY"]
            if not np.isfinite(quality) or int(quality) != 0:
                continue
        if not isinstance(segments["Br"][i], str) or not segments["Br"][i]:
            continue
        for comp in ("Br", "Bt", "Bp"):
            jobs.append((i, stamp, comp, "http://jsoc.stanford.edu" + segments[comp][i]))
    if not jobs:
        return [], []

    def pull(job):
        i, stamp, comp, url = job
        for attempt in range(3):
            try:
                with fits.open(url) as hdul:
                    return i, stamp, comp, hdul[1].data.astype(np.float32)
            except Exception:
                time.sleep(3 * (attempt + 1))
        return i, stamp, comp, None

    parts: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, stamp, comp, data in pool.map(pull, jobs):
            if data is None:
                continue
            parts.setdefault(i, {"t": stamp})[comp] = data

    timestamps, frames = [], []
    for i in sorted(parts):
        entry = parts[i]
        if not all(c in entry for c in ("Br", "Bt", "Bp")):
            continue
        timestamps.append(entry["t"])
        frames.append(np.stack([entry["Br"], entry["Bt"], entry["Bp"]]))
    return timestamps, frames


def fetch_with_retry(harp, tstart, tend, email, cadence, attempts=3, workers=8,
                     allowed=None):
    for attempt in range(attempts):
        try:
            return fetch_cube_parallel(harp, tstart, tend, email, cadence, workers,
                                       allowed=allowed)
        except Exception as error:  # noqa: BLE001
            wait = 15 * (attempt + 1)
            print(f"      attempt {attempt+1}: {type(error).__name__}; retry in {wait}s")
            time.sleep(wait)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--email", default="diego.gaf28@gmail.com")
    parser.add_argument("--cadence", default="3h")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-nan-fraction", type=float, default=0.35)
    args = parser.parse_args()

    positions = json.loads(Path(args.positions).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(positions)} ARs -> {out}")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for index, (noaa, entry) in enumerate(sorted(positions.items())):
        target = out / f"noaa{noaa}.npz"
        if target.exists():
            continue
        harp = entry["harp"]
        stamps = [r["t"] for r in entry["rows"]]
        tstart, tend = stamps[0][:10], stamps[-1][:10]

        print(f"[{index+1}/{len(positions)}] {noaa} harp {harp} {tstart}..{tend}")
        started = time.perf_counter()
        # Replay the QUALITY filter already applied when positions were
        # built off-cluster; JSOC returns NaN for that keyword here.
        allowed = {r["t"] for r in entry["rows"]}
        timestamps, frames = fetch_with_retry(
            harp, tstart, tend, args.email, args.cadence, allowed=allowed
        )
        if not frames:
            print("      no frames, skipped")
            continue

        cube = stack_frames(frames)[:, :, :: args.stride, :: args.stride]
        nan_fraction = np.isnan(cube).mean(axis=(1, 2, 3))
        keep = np.nonzero(nan_fraction <= args.max_nan_fraction)[0]
        if keep.size == 0:
            print(f"      every frame above {args.max_nan_fraction:.0%} NaN, skipped")
            continue

        cube = cube[keep].astype(np.float32)
        timestamps = [timestamps[i] for i in keep]
        np.savez_compressed(
            target,
            cube=cube,
            timestamps=np.array(timestamps),
            harp=harp,
            label=entry["label"],
            stride=args.stride,
        )
        manifest[noaa] = {
            "harp": harp,
            "label": entry["label"],
            "n_frames": len(timestamps),
            "shape": list(cube.shape),
            "dropped_to_nan": int(len(nan_fraction) - keep.size),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(
            f"      {cube.shape} {target.stat().st_size/1e6:.1f} MB "
            f"({len(nan_fraction)-keep.size} frames dropped to NaN) "
            f"{time.perf_counter()-started:.0f}s"
        )

    total = sum(p.stat().st_size for p in out.glob("*.npz")) / 1e9
    frames = sum(m["n_frames"] for m in manifest.values())
    print(f"\n{len(manifest)} ARs, {frames} frames, {total:.2f} GB in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
