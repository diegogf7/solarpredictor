import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

from eval.splits import make_splits, apply_splits
from eval.harness import evaluate
from eval.baselines import ClimateBaseline, ScalarBaseline
from models.forecaster import TemporalForecaster

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
records = json.load(open(os.path.join(CACHE, "dataset_sharp.json")))

manifest = make_splits(records)
train, validation, test = apply_splits(records, manifest)
print(f"train {len(train)} validation {len(validation)} test{len(test)}")
print(f"test positives: {sum([record["label"] for record in test])}" )

for name, model in [
    ("Climatology", ClimateBaseline()),
    ("Sharp-scaler", ScalarBaseline()),
    ("Temporal-GRU", TemporalForecaster()),
]:
    print(f"\n== {name} ==")
    results = evaluate(model, train, validation, test)
    for k, v in results.items():

        print(f"{k}: {v:.5f}" if isinstance(v, float) else f" {k}: {v}")