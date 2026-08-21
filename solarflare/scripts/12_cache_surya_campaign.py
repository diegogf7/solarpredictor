"""Surya latent campaign: parallel download, serial GPU encode, resumable.

The pilot measured 25 s per frame single-stream, which puts a real campaign in
the hundreds of hours. Downloads therefore run on a thread pool that stays a
few frames ahead of the GPU, so I/O and compute overlap instead of alternating.

Every frame is deleted immediately after encoding. Disk never holds more than
`--prefetch + 1` frames (~600 MB each), regardless of campaign size.

Resumable at frame granularity: progress is checkpointed, and re-running skips
timestamps already encoded.
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.encoders.surya_encoder import SuryaEncoder  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402

S3_ROOT = "https://nasa-surya-bench.s3.amazonaws.com"
ARCSEC_PER_PIXEL = 0.6
DISK_CENTER = 2048.0
SENTINEL = object()


def frame_url(when: datetime) -> str:
    return (
        f"{S3_ROOT}/{when.year:04d}/{when.month:02d}/"
        f"{when.year:04d}{when.month:02d}{when.day:02d}_"
        f"{when.hour:02d}{when.minute:02d}.nc"
    )


def signum_log_normalize(data, scalers, channels):
    out = np.empty_like(data, dtype=np.float32)
    for index, name in enumerate(channels):
        s = scalers[name]
        val = data[index] * float(s["sl_scale_factor"])
        val = np.sign(val) * np.log1p(np.abs(val))
        out[index] = (val - float(s["mean"])) / (float(s["std"]) + float(s["epsilon"]))
    return out


def heliographic_to_pixel(lat_deg, lon_deg, when):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, get_earth

    observer = get_earth(when)
    point = SkyCoord(
        lon_deg * u.deg, lat_deg * u.deg,
        frame=HeliographicStonyhurst, obstime=when, observer=observer,
    )
    projected = point.transform_to(Helioprojective(observer=observer, obstime=when))
    tx, ty = projected.Tx.to_value(u.arcsec), projected.Ty.to_value(u.arcsec)
    if not (np.isfinite(tx) and np.isfinite(ty)):
        return None
    return DISK_CENTER + tx / ARCSEC_PER_PIXEL, DISK_CENTER - ty / ARCSEC_PER_PIXEL


def downloader(
    work: queue.Queue,
    ready: queue.Queue,
    scratch: Path,
    stop: threading.Event,
    slots: threading.Semaphore,
):
    while not stop.is_set():
        item = work.get()
        if item is SENTINEL:
            work.task_done()
            break
        when = item
        # Bound files on disk: one slot per downloaded-but-unconsumed frame,
        # released by the consumer after it unlinks. Workers pull `work` in
        # timeline order, so the frame the consumer is waiting for is always
        # among the first in flight and cannot be starved.
        slots.acquire()
        target = scratch / Path(frame_url(when)).name
        ok = False
        for attempt in range(3):
            if stop.is_set():
                break
            try:
                urllib.request.urlretrieve(frame_url(when), target)
                ok = True
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    break
                time.sleep(4 * (attempt + 1))
            except Exception:
                time.sleep(4 * (attempt + 1))
        if not ok:
            slots.release()
        ready.put((when, target if ok else None))
        work.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--positions", default="flare_forecaster/cache/campaign_positions.json")
    parser.add_argument("--out", default="artifacts/caches/surya_campaign.pt")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=12)
    parser.add_argument("--box-arcsec", type=float, default=200.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-budget-s", type=float, default=39600)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--socket-timeout", type=float, default=120.0,
        help="urlretrieve has no default timeout; without one a hung S3 "
             "connection holds its prefetch slot forever and deadlocks",
    )
    parser.add_argument(
        "--max-pair-gap-min", type=float, default=240.0,
        help="beyond this gap, self-pair rather than claim a stale frame "
             "is the preceding timestep",
    )
    args = parser.parse_args()

    import xarray as xr

    socket.setdefaulttimeout(args.socket_timeout)

    set_seed(0)
    device = torch.device(args.device)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights_dir)
    scalers = yaml.safe_load((weights / "scalers.yaml").read_text())
    encoder = SuryaEncoder(weights, pool="crop_mean", device=device)
    encoder.model.to(torch.bfloat16)
    channels = list(encoder.channels)

    raw = json.loads(Path(args.positions).read_text())
    # timestamp -> [(noaa, lat, lon), ...]
    by_time: dict[datetime, list] = defaultdict(list)
    for noaa, entry in raw.items():
        for row in entry["rows"]:
            when = datetime.fromisoformat(row["t"]).replace(tzinfo=timezone.utc)
            by_time[when].append((noaa, row["lat"], row["lon"]))

    # resume
    store: dict[str, dict[str, list]] = {}
    done: set[str] = set()
    if out.exists():
        payload = torch.load(out, map_location="cpu", weights_only=False)
        store = payload["store"]
        done = set(payload["done"])
        print(f"resuming: {len(done)} frames already encoded")

    timeline = sorted(t for t in by_time if t.isoformat() not in done)
    print(
        f"{len(raw)} ARs | {len(by_time)} distinct frames | {len(timeline)} remaining"
        f" | ~{len(timeline)*0.587:.0f} GB to stream"
    )
    if not timeline:
        print("nothing to do")
        return 0

    work: queue.Queue = queue.Queue()
    ready: queue.Queue = queue.Queue()
    stop = threading.Event()
    slots = threading.Semaphore(max(args.prefetch, args.workers + 2))
    threads = [
        threading.Thread(
            target=downloader, args=(work, ready, scratch, stop, slots), daemon=True
        )
        for _ in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for when in timeline:
        work.put(when)
    for _ in threads:
        work.put(SENTINEL)

    half = args.box_arcsec / ARCSEC_PER_PIXEL / 2.0
    started = time.perf_counter()
    encoded = missing = self_paired = 0
    previous = None
    previous_time = None
    pending: dict[datetime, Path | None] = {}
    gaps: list[float] = []

    try:
        for index in range(len(timeline)):
            if time.perf_counter() - started > args.time_budget_s:
                print("\nTIME BUDGET reached; checkpointing and stopping")
                stop.set()
                break

            # Consume in TIMELINE order, not completion order. Downloads finish
            # out of order under concurrency; taking them as they land made the
            # second Surya timestep an unrelated frame from another AR or year.
            when = timeline[index]
            while when not in pending:
                arrived, path = ready.get()
                pending[arrived] = path
            target = pending.pop(when)

            if target is None:
                missing += 1
                done.add(when.isoformat())
                continue

            try:
                with xr.open_dataset(target) as ds:
                    frame = np.stack([ds[name].to_numpy() for name in channels])
                normalized = signum_log_normalize(frame, scalers, channels)
                del frame
                current = torch.from_numpy(normalized).to(device=device, dtype=torch.bfloat16)

                # Surya conditions on the true gap via time_delta_input. Pair
                # with the preceding frame when it is recent enough; otherwise
                # self-pair, which is a well-defined "no motion" input rather
                # than a lie about a frame we do not have.
                gap_minutes = (
                    (when - previous_time).total_seconds() / 60.0
                    if previous_time is not None
                    else float("inf")
                )
                if previous is None or gap_minutes > args.max_pair_gap_min:
                    pair_previous, delta = current, 0.0
                    self_paired += 1
                else:
                    pair_previous, delta = previous, -gap_minutes
                    gaps.append(gap_minutes)

                ts = torch.stack([pair_previous, current], dim=1).unsqueeze(0)
                dt = torch.tensor(
                    [[delta, 0.0]], device=device, dtype=torch.bfloat16
                )

                for noaa, lat, lon in by_time[when]:
                    pixel = heliographic_to_pixel(lat, lon, when)
                    if pixel is None:
                        continue
                    cx, cy = pixel
                    box = torch.tensor(
                        [[cx - half, cy - half, cx + half, cy + half]], device=device
                    )
                    mask = encoder.box_to_token_mask(box, margin_px=0)
                    if int(mask.sum()) == 0:
                        continue
                    pooled = encoder(ts, dt, mask).float().squeeze(0).cpu()
                    bucket = store.setdefault(noaa, {"z": [], "t": [], "dt": []})
                    bucket["z"].append(pooled)
                    bucket["t"].append(when.timestamp())
                    bucket["dt"].append(delta)

                previous = current
                previous_time = when
                encoded += 1
                done.add(when.isoformat())
            finally:
                target.unlink(missing_ok=True)
                slots.release()

            if encoded and encoded % args.checkpoint_every == 0:
                torch.save({"store": store, "done": sorted(done)}, out)
                elapsed = time.perf_counter() - started
                rate = elapsed / max(encoded, 1)
                remaining = (len(timeline) - index - 1) * rate
                print(
                    f"  [{index+1}/{len(timeline)}] {encoded} encoded, {missing} missing"
                    f" | {rate:.1f}s/frame | ~{remaining/3600:.1f}h left"
                )
    finally:
        stop.set()
        for leftover in list(pending.values()):
            if leftover is not None:
                Path(leftover).unlink(missing_ok=True)
        while not ready.empty():
            _, leftover = ready.get()
            if leftover is not None:
                Path(leftover).unlink(missing_ok=True)
        torch.save({"store": store, "done": sorted(done)}, out)

    packed = {
        noaa: {
            "z_surya": torch.stack(v["z"]),
            "t": torch.tensor(v["t"], dtype=torch.float64),
            "label": raw[noaa]["label"],
            "harp": raw[noaa]["harp"],
        }
        for noaa, v in store.items()
        if v["z"]
    }
    torch.save({"store": store, "done": sorted(done), "packed": packed}, out)

    import statistics
    print(f"\npairing: {encoded-self_paired} true-pair, {self_paired} self-paired"
          + (f" | gap min {min(gaps):.0f} median {statistics.median(gaps):.0f} "
             f"max {max(gaps):.0f} min" if gaps else ""))
    counts = [v["z_surya"].shape[0] for v in packed.values()]
    print(f"\n{len(packed)} ARs with latents | frames per AR: "
          f"min {min(counts)} median {int(np.median(counts))} max {max(counts)}")
    print(f"encoded {encoded}, missing {missing}, "
          f"{(time.perf_counter()-started)/3600:.2f}h elapsed")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
