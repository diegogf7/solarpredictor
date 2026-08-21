"""Rung 0 vs rung 3: do Surya latents beat SHARP scalars?

Protocol, and it is the point of the script:
  - fit on TRAIN active regions only
  - standardization, calibration, and the decision threshold selected on VAL
  - TEST touched once, after everything above is frozen
  - paired bootstrap over ARs on the DIFFERENCE between models, not separate
    intervals per model

Labels are per-AR (M+ within +/-5d of the reference time), so predictions are
per-AR too: an AR's latents are pooled over its frames. This is active-region
triage, directly comparable to the 962-AR result where USFLUX reached AUC
0.863 and hand-crafted twist reached 0.522.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.eval.metrics import (  # noqa: E402
    bss,
    expected_calibration_error,
    tss,
)
from flare_forecaster.utils.seed import set_seed  # noqa: E402

SPLITS = ("train", "val", "test")


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC, ties averaged."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    unique, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    for index in np.nonzero(counts > 1)[0]:
        mask = inverse == index
        ranks[mask] = ranks[mask].mean()
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_split(path: Path, sharp: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    packed = payload["packed"]
    ars, x_surya, x_sharp, y = [], [], [], []
    for noaa, entry in sorted(packed.items()):
        if noaa not in sharp:
            continue
        ars.append(noaa)
        # Mean over the AR's frames: the label is a property of the region.
        x_surya.append(entry["z_surya"].float().mean(dim=0).numpy())
        x_sharp.append(sharp[noaa])
        y.append(float(entry["label"]))
    return {
        "ar": ars,
        "surya": np.stack(x_surya),
        "sharp": np.array(x_sharp, dtype=np.float64),
        "y": np.array(y),
    }


def fit_logistic(x, y, steps=4000, l2=1e-2, lr=0.5):
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    w = np.zeros(x.shape[1])
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1.0)
    sample_w = np.where(y == 1, pos_weight, 1.0)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = x.T @ (sample_w * (p - y)) / sample_w.sum() + l2 * w
        w -= lr * grad
    return w


def predict_logistic(w, x):
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))


class SuryaHead(torch.nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def platt(scores_val, y_val, scores):
    """Single-feature logistic calibration fitted on validation only."""
    w = fit_logistic(scores_val.reshape(-1, 1), y_val, steps=3000, l2=1e-3)
    return predict_logistic(w, scores.reshape(-1, 1))


def best_threshold(y, p):
    candidates = np.unique(np.round(p, 4))
    scores = [(tss(y, (p >= t).astype(float)), t) for t in candidates]
    return max(scores)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caches", default="artifacts/caches")
    parser.add_argument("--triage", default="flare_forecaster/cache/triage_dataset.json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", default="artifacts/rung3_results.json")
    args = parser.parse_args()

    set_seed(42)
    rng = np.random.default_rng(42)
    triage = {str(r["noaa"]): r["sharp"] for r in json.loads(Path(args.triage).read_text())}

    caches = Path(args.caches)
    data = {
        "train": load_split(caches / "surya_campaign.pt", triage),
        "val": load_split(caches / "surya_campaign_val.pt", triage),
        "test": load_split(caches / "surya_campaign_test.pt", triage),
    }
    for name in SPLITS:
        d = data[name]
        print(f"{name:6} {len(d['y']):3} ARs, {int(d['y'].sum()):2} flaring "
              f"({d['y'].mean():.1%})")

    overlap = set(data["train"]["ar"]) & set(data["test"]["ar"])
    assert not overlap, f"AR overlap between train and test: {overlap}"
    print("splits disjoint\n")

    # ---- standardization, fitted on TRAIN only --------------------------
    scalers = {}
    for key in ("surya", "sharp"):
        x = data["train"][key]
        centre = np.median(x, axis=0)
        spread = (np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0))
        spread = np.where(spread < 1e-12, 1.0, spread)
        scalers[key] = (centre, spread)
        for name in SPLITS:
            data[name][key + "_z"] = np.clip(
                (data[name][key] - centre) / spread, -10, 10
            )

    y = {name: data[name]["y"] for name in SPLITS}
    raw_scores: dict[str, dict[str, np.ndarray]] = {}

    # ---- rung 0a: single strongest SHARP scalar (USFLUX) ----------------
    for index, label in [(0, "USFLUX alone")]:
        raw_scores[label] = {n: data[n]["sharp_z"][:, index] for n in SPLITS}

    # ---- rung 0b: all four SHARP scalars --------------------------------
    w = fit_logistic(data["train"]["sharp_z"], y["train"])
    raw_scores["SHARP-4 logistic"] = {
        n: predict_logistic(w, data[n]["sharp_z"]) for n in SPLITS
    }

    # ---- rung 3: Surya latents + head -----------------------------------
    xt = torch.tensor(data["train"]["surya_z"], dtype=torch.float32)
    yt = torch.tensor(y["train"], dtype=torch.float32)
    xv = torch.tensor(data["val"]["surya_z"], dtype=torch.float32)
    yv = torch.tensor(y["val"], dtype=torch.float32)
    pos_weight = torch.tensor((len(yt) - yt.sum()) / max(float(yt.sum()), 1.0))

    head = SuryaHead(xt.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    best_state, best_val = None, -np.inf
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(xt), yt, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                val_auc = auc(y["val"], head(xv).numpy())
            # Early stopping on VALIDATION, never on test.
            if val_auc > best_val:
                best_val, best_state = val_auc, {
                    k: v.clone() for k, v in head.state_dict().items()
                }
    head.load_state_dict(best_state)
    head.eval()
    print(f"Surya head: best val AUC {best_val:.3f} (early-stopped on val)\n")
    with torch.no_grad():
        raw_scores["Surya latents + head"] = {
            n: head(torch.tensor(data[n]["surya_z"], dtype=torch.float32)).numpy()
            for n in SPLITS
        }

    # ---- calibrate on val, threshold on val, then score test ------------
    results = {}
    probabilities = {}
    for model, scores in raw_scores.items():
        p_val = platt(scores["val"], y["val"], scores["val"])
        p_test = platt(scores["val"], y["val"], scores["test"])
        threshold = best_threshold(y["val"], p_val)
        probabilities[model] = p_test
        hard = (p_test >= threshold).astype(float)
        results[model] = {
            "auc": auc(y["test"], scores["test"]),
            "tss": float(tss(y["test"], hard)),
            "bss": float(bss(y["test"], p_test)),
            "ece": float(expected_calibration_error(y["test"], p_test)),
            "threshold": float(threshold),
            "val_auc": auc(y["val"], scores["val"]),
        }

    print("=== TEST (60 ARs, 21 flaring) — threshold & calibration fixed on val ===")
    print(f"{'model':<24}{'AUC':>7}{'TSS':>7}{'BSS':>8}{'ECE':>7}{'val AUC':>9}")
    for model, r in results.items():
        print(f"{model:<24}{r['auc']:>7.3f}{r['tss']:>7.3f}{r['bss']:>8.3f}"
              f"{r['ece']:>7.3f}{r['val_auc']:>9.3f}")

    # ---- paired bootstrap over ARs on the DIFFERENCE --------------------
    print(f"\n=== paired bootstrap over test ARs, {args.bootstrap} resamples ===")
    baseline = "SHARP-4 logistic"
    comparisons = {}
    n_test = len(y["test"])
    for model in raw_scores:
        if model == baseline:
            continue
        deltas = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, n_test, n_test)
            if y["test"][idx].sum() in (0, len(idx)):
                continue
            deltas.append(
                auc(y["test"][idx], raw_scores[model]["test"][idx])
                - auc(y["test"][idx], raw_scores[baseline]["test"][idx])
            )
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        comparisons[model] = {
            "delta_auc": float(np.mean(deltas)),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_better": float((deltas > 0).mean()),
        }
        verdict = "significant" if lo > 0 else ("worse" if hi < 0 else "not distinguishable")
        print(f"{model:<24} dAUC {np.mean(deltas):+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"results": results, "comparisons": comparisons,
         "n": {n: int(len(y[n])) for n in SPLITS},
         "n_flaring": {n: int(y[n].sum()) for n in SPLITS}}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
