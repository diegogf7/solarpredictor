import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import random
from eval.harness import evaluate
from eval.splits import make_splits, apply_splits
from eval.baselines import Baseline, ClimateBaseline, ScalarBaseline


random.seed(42)

records = []

for ar_num in range(20):
    ar_id = f"AR{ar_num:04d}"

    for i in range(5):

        year = 2010 + ar_num // 4
        month = (ar_num % 12 ) +1
        records.append({
            "ar_id": ar_id,
            "timestamp": f"{year}-{month:02d}-{(i+1):02d}T00:00:00",
            "label": 1 if random.random() < 0.1 else 0,
            "features": [random.gauss(0, 1) for _ in range(4)],

        })

temp = make_splits(records)

train, validation, test = apply_splits(records, temp)

print(f"Split - train: {len(train)}, Split - Validation: {len(validation)}, Split - Testing: {len(test)}")

for name, model in [
    ("Persistence", Baseline()),
    ("Climatology", ClimateBaseline()),
    ("Sharp-scalar", ScalarBaseline())]:
    
    results = evaluate(model, train, validation, test)
    print(f"\n{name}:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")



