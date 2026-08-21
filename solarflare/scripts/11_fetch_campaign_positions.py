"""Fetch AR positions for the Surya campaign. Runs on the laptop, not the cluster.

JSOC keyword queries only -- no images. Compute nodes return NaN keywords, so
this has to happen off-cluster and travel as JSON.

Resumable: re-running only fetches ARs missing from the output file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.data.splits import load_split  # noqa: E402
from flare_forecaster.manifests import ARTIFACTS  # noqa: E402


def parse_t_ref(t_ref) -> str:
    """'2010.06.01_00:00:00_TAI' -> '2010-06-01' (date only)."""
    text = str(t_ref).strip().removesuffix("_TAI").removesuffix("_UTC")
    return text.split("_")[0].replace(".", "-")


def fetch_one(client, harp: int, date: str, days: int, cadence: str):
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(date) - timedelta(days=days // 2)
    end = start + timedelta(days=days)
    t0 = start.strftime("%Y.%m.%d") + "_00:00:00_TAI"
    t1 = end.strftime("%Y.%m.%d") + "_00:00:00_TAI"
    keys = client.query(
        f"hmi.sharp_cea_720s[{harp}][{t0}-{t1}@{cadence}]",
        key=["T_REC", "QUALITY", "LAT_FWT", "LON_FWT", "SIZE_ACR"],
    )
    rows = []
    for _, row in keys.iterrows():
        if not (np.isfinite(row["LAT_FWT"]) and np.isfinite(row["LON_FWT"])):
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
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="flare_forecaster/cache/ar_catalog.json")
    parser.add_argument("--triage", default="flare_forecaster/cache/triage_dataset.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-ars", type=int, default=100)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--cadence", default="3h")
    parser.add_argument("--max-central-lon", type=float, default=60.0,
                        help="drop ARs never within this longitude of centre")
    parser.add_argument("--email", default="diego.gaf28@gmail.com")
    parser.add_argument("--out", default="flare_forecaster/cache/campaign_positions.json")
    args = parser.parse_args()

    import drms

    catalog = {str(r["noaa"]): r for r in json.loads(Path(args.catalog).read_text())}
    labels = {
        str(r["noaa"]): int(bool(r.get("label")))
        for r in json.loads(Path(args.triage).read_text())
    }
    split_ars = [a for a in load_split(args.split, ARTIFACTS) if a in catalog]
    print(f"{args.split} split: {len(split_ars)} ARs with catalog entries")

    # Keep the flaring/quiet balance of the split while subsampling.
    flaring = [a for a in split_ars if labels.get(a)]
    quiet = [a for a in split_ars if not labels.get(a)]
    take_flaring = min(len(flaring), max(1, int(args.n_ars * len(flaring) / len(split_ars))))
    chosen = flaring[:take_flaring] + quiet[: args.n_ars - take_flaring]
    print(f"selected {len(chosen)} ({take_flaring} flaring)")

    out = Path(args.out)
    positions = json.loads(out.read_text()) if out.exists() else {}
    print(f"already cached: {len(positions)}")

    client = drms.Client(email=args.email)
    skipped_limb = 0
    for index, noaa in enumerate(chosen):
        if noaa in positions:
            continue
        harp = int(catalog[noaa]["harpnum"])
        date = parse_t_ref(catalog[noaa]["t_ref"])
        try:
            rows = fetch_one(client, harp, date, args.days, args.cadence)
        except Exception as error:  # noqa: BLE001
            print(f"  [{index+1}/{len(chosen)}] {noaa} FAILED: {type(error).__name__}")
            time.sleep(5)
            continue
        if not rows:
            print(f"  [{index+1}/{len(chosen)}] {noaa} no rows")
            continue
        # An AR that never comes within max_central_lon is foreshortened
        # throughout; the pilot showed a limb region is a poor sample.
        if min(abs(r["lon"]) for r in rows) > args.max_central_lon:
            skipped_limb += 1
            print(f"  [{index+1}/{len(chosen)}] {noaa} limb-only, skipped")
            continue
        positions[noaa] = {
            "harp": harp,
            "label": labels.get(noaa, 0),
            "rows": rows,
        }
        print(f"  [{index+1}/{len(chosen)}] {noaa} harp {harp} "
              f"{len(rows)} rows lon {rows[0]['lon']:.0f}..{rows[-1]['lon']:.0f} "
              f"label {labels.get(noaa,0)}")
        if len(positions) % 10 == 0:
            out.write_text(json.dumps(positions))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(positions))
    n_flaring = sum(1 for v in positions.values() if v["label"])
    stamps = {r["t"] for v in positions.values() for r in v["rows"]}
    print(f"\n{len(positions)} ARs ({n_flaring} flaring), {skipped_limb} limb-only dropped")
    print(f"{len(stamps)} distinct timestamps -> ~{len(stamps)*0.587:.0f} GB to stream")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
