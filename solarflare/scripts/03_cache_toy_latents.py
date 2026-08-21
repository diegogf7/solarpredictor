"""Toy latents from existing SHARP scalars -- no Surya, no PINN, no GPU.

Stands in for scripts 03/04/06 so stages B-E can be exercised end to end
before any compute is committed. Latents are fixed random projections of real
SHARP features, so they carry real signal: if the pipeline learns nothing
here, that is informative.

Writes its own split files under a separate artifact root so the real
400-AR splits are left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.config import Config, resolve_surya_dim
from flare_forecaster.data.splits import assert_no_overlap, make_ar_splits, write_splits
from flare_forecaster.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="flare_forecaster/cache/dataset_sharp.json"
    )
    parser.add_argument("--artifacts", default="artifacts_toy")
    parser.add_argument("--out", default="artifacts_toy/caches/toy_latents.pt")
    parser.add_argument(
        "--d-surya",
        type=int,
        default=1280,
        help="matches the measured Surya width so shapes mirror the real run",
    )
    args = parser.parse_args()

    cfg = resolve_surya_dim(Config(), args.d_surya)
    set_seed(cfg.seed)
    artifacts = Path(args.artifacts)

    records = json.loads(Path(args.dataset).read_text())
    print(f"dataset: {len(records)} records")

    by_ar: dict[str, list[dict]] = {}
    for r in records:
        by_ar.setdefault(str(r["ar_id"]), []).append(r)

    n_features = len(records[0]["features"])
    generator = torch.Generator().manual_seed(cfg.seed)
    proj_surya = torch.randn(n_features, cfg.d_surya, generator=generator) / n_features**0.5
    proj_physics = torch.randn(n_features, cfg.d_physics, generator=generator) / n_features**0.5

    # Robust per-feature scaling on ALL toy ARs. This is a toy; the real
    # pipeline fits scalers on training ARs only (spec sec 9 / sec 11).
    all_features = torch.tensor(
        [r["features"] for r in records], dtype=torch.float64
    )
    median = all_features.median(dim=0).values
    iqr = (
        all_features.quantile(0.75, dim=0) - all_features.quantile(0.25, dim=0)
    ).clamp_min(1e-9)

    payload: dict[str, dict] = {}
    ar_times: dict[str, int] = {}
    for ar_id, rows in by_ar.items():
        rows = sorted(rows, key=lambda r: r["timestamp"])
        features = torch.tensor([r["features"] for r in rows], dtype=torch.float64)
        scaled = ((features - median) / iqr).clamp(-10, 10).float()

        times = torch.tensor(
            [
                datetime.fromisoformat(r["timestamp"]).timestamp()
                for r in rows
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([float(r["label"]) for r in rows])

        payload[ar_id] = {
            "z_surya": scaled @ proj_surya,
            "z_physics": scaled @ proj_physics,
            "t": times,
            "label": labels,
        }
        ar_times[ar_id] = int(times[0])
        print(
            f"  {ar_id}  {len(rows):>4} frames  {int(labels.sum()):>3} positive"
        )

    splits = make_ar_splits(ar_times, embargo_hours=0)
    write_splits(splits, artifacts=artifacts)
    assert_no_overlap(artifacts=artifacts)
    for name in ("train", "val", "test"):
        print(f"  {name:<6} {splits[name]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"\nwrote {out}  ({len(payload)} ARs, d_surya={cfg.d_surya}, d_physics={cfg.d_physics})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
