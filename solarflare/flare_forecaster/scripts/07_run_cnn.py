import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
from datetime import datetime, timedelta
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

from data.fetch import fetch_goes_flares
from eval.splits import make_splits, apply_splits
from eval.harness import evaluate
from eval.baselines import ClimateBaseline
from models.kappa_cnn import KappaCNNForecaster

FLARE_RANK = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "twist_maps")


records = []
for path in sorted(glob.glob(os.path.join(CACHE, "*npz"))):
    d = np.load(path, allow_pickle= True)

    maps = d["maps"]
    timestamps = [str(t) for t in d["timestamps"]]
    noaa = int(d["noaa"])

    t0 = timestamps[0][:10]
    t1 = (datetime.fromisoformat(timestamps[-1][:19]) + timedelta(days = 2)).strftime("%Y-%m-%d")
    flares = fetch_goes_flares(t0, t1)

    relevant = [

        f for f in flares # 
        if f["noaa_ar"] == noaa
        and FLARE_RANK.get((f["goes_class"] or "")[:1].upper(), 0) >= 4
    ]

    positives = 0
    for i, ts in enumerate(timestamps):

        start = datetime.fromisoformat(ts[:19])
        end = start + timedelta(hours = 24)
        label = int(any(start < datetime.fromisoformat(f["peak_time"][:19]) <= end for f in relevant))
        positives += label
        records.append({
            "ar_id": str(noaa),
            "timestamp": ts[:19],
            "map": maps[i].astype(np.float32),
            "label": label,
        })
    print(f"AR {noaa}: {len(timestamps)} maps, {positives} positives")


print(f"\nTotal: {len(records)} maps")
manifest = make_splits(records)

train, validation, test = apply_splits(records, manifest)
print(f"Train {len(train)}, validation {len(validation)}, test {len(test)}, test positives {sum(record['label'] for record in test)}")


for name, model in [
    ("Climatology", ClimateBaseline()),
    ("Kappa-CNN", KappaCNNForecaster()),


]:
    print(f"\n==={name}===")
    results = evaluate(model,train, validation, test)
    for k,v in results.items():

        print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")