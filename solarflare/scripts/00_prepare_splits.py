from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.config import Config
from flare_forecaster.data.splits import (
    assert_no_overlap,
    make_ar_splits,
    write_splits,
)

from flare_forecaster.manifests import ARTIFACTS, split_hash
from flare_forecaster.utils.seed import set_seed


def parse_t_ref(t_ref) -> int:
    """JSOC TAI string '2010.06.01_00:00:00_TAI' -> unix seconds.

    Already-numeric values pass through, so a future catalog storing unix
    times needs no change here.
    """
    if isinstance(t_ref, (int, float)):
        return int(t_ref)
    text = str(t_ref).strip().removesuffix("_TAI").removesuffix("_UTC")
    stamp = datetime.strptime(text, "%Y.%m.%d_%H:%M:%S")
    return int(stamp.replace(tzinfo=timezone.utc).timestamp())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default = "flare_forecaster/cache/triage_dataset.json",
        help = "JSON list of records with 'noaa', 't_ref', 'label'",
    )
    parser.add_argument("--n-ars", type = int, default = 200)
    parser.add_argument("--embargo-hours", type = int, default = 48)
    parser.add_argument(
        "--flaring-frac",
        type = float,
        default = 0.131,
        help = "target flaring fraction; 0.131 is the natural rate in the "
               "962-AR catalog. Raising it buys positives at the cost of a "
               "base rate that no longer matches deployment.",
    )

    args = parser.parse_args()

    cfg = Config()
    cfg.validate()
    set_seed(cfg.seed)

    records = json.loads(Path(args.catalog).read_text())
    print(f"catalog: {len(records)} ARs")

    # Both classes must be sampled ACROSS TIME, not off the front of the
    # catalog. The catalog is chronological, so quiet[:n] would put every
    # quiet AR in the earliest period and leave val/test 100% flaring.
    flaring = sorted(
        (r for r in records if r.get("label")), key=lambda r: parse_t_ref(r["t_ref"])
    )
    quiet = sorted(
        (r for r in records if not r.get("label")), key=lambda r: parse_t_ref(r["t_ref"])
    )

    def spread(items: list, k: int) -> list:
        """Evenly spaced subsample preserving the time distribution."""
        if k >= len(items):
            return list(items)
        step = len(items) / k
        return [items[int(i * step)] for i in range(k)]

    n_flaring = min(len(flaring), int(round(args.n_ars * args.flaring_frac)))
    n_quiet = args.n_ars - n_flaring
    chosen = spread(flaring, n_flaring) + spread(quiet, n_quiet)
    print(f"selected {len(chosen)} ARs ({sum(1 for r in chosen if r.get('label'))} flaring)")

    ar_times = {str(r["noaa"]): parse_t_ref(r["t_ref"]) for r in chosen}
    splits = make_ar_splits(ar_times, embargo_hours=args.embargo_hours)

    labels = {str(r["noaa"]): int(bool(r.get("label"))) for r in chosen}
    for name in ("train", "val", "test"):
        ars = splits[name]
        pos = sum(labels[a] for a in ars)
        print(f"  {name:<6} {len(ars):>4} ARs  {pos:>3} flaring")
    print(f"  dropped to embargo: {len(splits['_dropped_to_embargo'])}")

    write_splits(splits)
    assert_no_overlap()
    print(f"\nwrote {ARTIFACTS/'splits'}  split_hash={split_hash()[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
