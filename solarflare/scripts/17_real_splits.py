"""Physics branch on the REAL frozen splits, against the SHARP baseline.

Unlike 16_mini_pipeline.py, this does NOT sub-divide a split. It uses the
pre-registered train/val/test AR partition, which is possible because the PINN
readouts finished for all three splits (100/59/60 ARs) even though the Surya
encode has not.

So this is ONE branch, not two. It answers "does the physics arm carry signal
the SHARP scalars do not" on a real held-out test set. The fused two-branch
number has to wait for the Surya cache.

Protocol: fit on train, choose standardization and the decision threshold on
val, touch test once. Paired bootstrap over test ARs on the DIFFERENCE.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.contracts import PHYSICS_CHANNELS  # noqa: E402
from flare_forecaster.encoders.physics_decoder import PhysicsDecoder, masked_huber  # noqa: E402
from flare_forecaster.encoders.physics_encoder import PhysicsEncoder  # noqa: E402
from flare_forecaster.models.deterministic import DeterministicForecaster  # noqa: E402
from flare_forecaster.models.flare_head import RolloutAwareFlareHead  # noqa: E402
from flare_forecaster.models.flow_transformer import FlowMatchingTransformer  # noqa: E402
from flare_forecaster.models.rollout import rollout, summarize_rollouts  # noqa: E402
from flare_forecaster.checkpoints import freeze  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402

SPLITS = ("train", "val", "test")


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for i in np.nonzero(cnt > 1)[0]:
        m = inv == i
        ranks[m] = ranks[m].mean()
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tss_at(y, p, thr):
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = ((y == 1) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    tn = ((y == 0) & (pred == 0)).sum()
    return float(tp / max(tp + fn, 1) - fp / max(fp + tn, 1))


def pick_threshold(y, p):
    grid = np.unique(np.round(p, 3))
    return max(((tss_at(y, p, t), t) for t in grid))[1]


def paired_bootstrap(y, a, b, n=2000, seed=0):
    """AUC(a) - AUC(b), resampling ARs. Positive means a is better."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if y[take].sum() in (0, len(take)):
            continue
        d = auc(y[take], np.asarray(a)[take]) - auc(y[take], np.asarray(b)[take])
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.array(deltas)
    return (
        float(np.mean(deltas)),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
        float((deltas > 0).mean()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts-root", default="/work1/jeroenaudenaert/diegogon/readouts")
    ap.add_argument("--triage", default="flare_forecaster/cache/triage_dataset.json")
    ap.add_argument("--out", default="artifacts/real_splits_results.json")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--ae-epochs", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--rollout-samples", type=int, default=8)
    ap.add_argument("--flow-steps", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_seed(42)
    t0 = time.perf_counter()
    dev = torch.device(args.device)
    H = args.history

    # ---------- load readouts, keyed by the frozen split -------------------
    ars, member = {}, {s: [] for s in SPLITS}
    for split in SPLITS:
        d = Path(args.readouts_root) / f"readouts_{split}"
        for fp in sorted(d.glob("noaa*.pt")):
            noaa = fp.stem[4:]
            rec = torch.load(fp, map_location="cpu", weights_only=False)
            n = rec["readout"].shape[0]
            if n < H:
                continue
            ars[noaa] = {
                "readout": rec["readout"].float(),
                "label": int(rec["label"]),
                "split": split,
            }
            member[split].append(noaa)
    for s in SPLITS:
        pos = sum(ars[a]["label"] for a in member[s])
        print(f"  {s:5} {len(member[s]):3} ARs, {pos:2} flaring")
    assert not (set(member["train"]) & set(member["test"])), "AR leak"

    # ---------- Stage A: physics autoencoder, train ARs only ---------------
    res = args.resolution

    def prep(a):
        r = ars[a]["readout"]
        return torch.nn.functional.avg_pool2d(r, r.shape[-1] // res)

    train_maps = torch.cat([prep(a) for a in member["train"]])
    flat = train_maps.permute(1, 0, 2, 3).reshape(len(PHYSICS_CHANNELS), -1)
    med = flat.median(dim=1).values
    iqr = (flat.quantile(.75, dim=1) - flat.quantile(.25, dim=1)).clamp_min(1e-9)

    def scale(x):
        y = ((x - med[:, None, None]) / iqr[:, None, None]).clamp(-10, 10)
        y = torch.nan_to_num(y)
        for nm in ("trace_valid", "strong_field_mask"):
            i = PHYSICS_CHANNELS.index(nm)
            y[:, i] = x[:, i]
        return y

    enc = PhysicsEncoder(len(PHYSICS_CHANNELS), 256).to(dev)
    dec = PhysicsDecoder(256, len(PHYSICS_CHANNELS), base=res // 16).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)
    xtr = scale(train_maps).to(dev)
    vi = PHYSICS_CHANNELS.index("trace_valid")
    valid = (xtr[:, vi:vi + 1] > 0).expand_as(xtr)
    print(f"\n[A] physics autoencoder on {xtr.shape[0]} train maps")
    for ep in range(args.ae_epochs):
        perm = torch.randperm(xtr.shape[0])
        tot = 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad(set_to_none=True)
            loss = masked_huber(dec(enc(xtr[b])), xtr[b], valid[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        if ep % 5 == 0 or ep == args.ae_epochs - 1:
            print(f"  epoch {ep:>3}  masked_huber {tot / len(perm):.5f}")
    freeze(enc)
    enc.eval()
    with torch.no_grad():
        for a in ars:
            ars[a]["z_phys"] = enc(scale(prep(a)).to(dev)).cpu()

    # ---------- causal windows, grouped by AR ------------------------------
    def windows(names):
        zp, y, g = [], [], []
        for a in names:
            P, L = ars[a]["z_phys"], ars[a]["label"]
            for e in range(H, len(P) + 1):
                zp.append(P[e - H:e])
                y.append(float(L))
                g.append(a)
        return torch.stack(zp), torch.tensor(y), np.array(g)

    W = {s: windows(member[s]) for s in SPLITS}
    for s in SPLITS:
        print(f"  {s:5} {W[s][0].shape[0]:5} windows")
    d_p = W["train"][0].shape[-1]
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def pw(y):
        return torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))

    def run(name, mod, lossfn, epochs, lr):
        ps = [p for p in mod.parameters() if p.requires_grad]
        o = torch.optim.AdamW(ps, lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            o.zero_grad(set_to_none=True)
            l = lossfn()
            if not torch.isfinite(l):
                raise FloatingPointError(f"{name} non-finite")
            l.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            o.step()
            if ep % max(epochs // 3, 1) == 0 or ep == epochs - 1:
                print(f"  [{name}] {ep:>4} {float(l):.5f}")

    ytr = W["train"][1]

    # ---------- rung 2: pooled physics head --------------------------------
    print("\n[2] pooled physics head")
    phys_head = torch.nn.Sequential(
        torch.nn.Linear(d_p, 128), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1)
    )
    pooled = {s: W[s][0].mean(1) for s in SPLITS}
    run("physics", phys_head,
        lambda: bce(phys_head(pooled["train"]).squeeze(-1), ytr, pos_weight=pw(ytr)),
        args.epochs, 3e-4)

    # ---------- rungs 3-4: temporal stack ----------------------------------
    hist = {s: W[s][0][:, :-1] for s in SPLITS}
    tgt = {s: W[s][0][:, -1] for s in SPLITS}
    dl = {s: torch.ones(W[s][0].shape[0], H - 1) for s in SPLITS}
    vm = {s: torch.ones(W[s][0].shape[0], H - 1, dtype=torch.bool) for s in SPLITS}

    det = DeterministicForecaster(256, 256, 2)
    print("\n[C] deterministic")
    run("determ", det,
        lambda: torch.nn.functional.huber_loss(det(hist["train"], dl["train"]), tgt["train"]),
        args.epochs, 3e-4)
    freeze(det)
    with torch.no_grad():
        resid = tgt["train"] - det(hist["train"], dl["train"])
    rs = (resid - resid.median(0).values).abs().median(0).values.mul(1.4826).clamp_min(1e-4)

    flow = FlowMatchingTransformer(256, 256, 4, 8, 1024, 0.10, max_history=H)

    def dloss():
        with torch.no_grad():
            mu = det(hist["train"], dl["train"])
        x0 = mu + rs * torch.randn_like(mu)
        x1 = tgt["train"]
        s = torch.rand(x1.shape[0])
        xs = (1 - s[:, None]) * x0 + s[:, None] * x1
        return torch.nn.functional.mse_loss(
            flow(xs, s, hist["train"], dl["train"], vm["train"]), x1 - x0
        )

    print("\n[D] flow matching")
    run("flow", flow, dloss, args.epochs, 2e-4)
    freeze(flow)

    print(f"\n[E] rollouts ({args.rollout_samples} traj x {H} steps)")
    summ = {}
    for s in SPLITS:
        ens = rollout(det, flow, hist[s], dl[s], vm[s], rs, future_steps=H,
                      samples=args.rollout_samples, n_flow_steps=args.flow_steps)
        summ[s] = summarize_rollouts(hist[s], vm[s], ens)
    head = RolloutAwareFlareHead(256)
    run("flare", head, lambda: bce(head(summ["train"]), ytr, pos_weight=pw(ytr)),
        args.epochs, 3e-4)

    # ---------- rung 0: SHARP scalars, the traditional bar ------------------
    triage = {str(r["noaa"]): r for r in json.load(open(args.triage))}
    have = {s: [a for a in member[s] if a in triage] for s in SPLITS}
    print(f"\n[0] SHARP baseline coverage: " +
          ", ".join(f"{s} {len(have[s])}/{len(member[s])}" for s in SPLITS))

    Xs = {s: np.array([triage[a]["sharp"] for a in have[s]], float) for s in SPLITS}
    Ys = {s: np.array([ars[a]["label"] for a in have[s]], float) for s in SPLITS}
    Xs = {s: np.log10(np.abs(v) + 1e-9) for s, v in Xs.items()}
    mu, sd = Xs["train"].mean(0), Xs["train"].std(0).clip(1e-9)
    Xs = {s: (v - mu) / sd for s, v in Xs.items()}
    w = np.zeros(Xs["train"].shape[1] + 1)
    Xtr = np.c_[Xs["train"], np.ones(len(Xs["train"]))]
    posw = (Ys["train"] == 0).sum() / max((Ys["train"] == 1).sum(), 1)
    sw = np.where(Ys["train"] == 1, posw, 1.0)
    for _ in range(4000):
        p = 1 / (1 + np.exp(-Xtr @ w))
        w -= 1e-2 * (Xtr.T @ (sw * (p - Ys["train"]))) / len(Ys["train"])
    sharp_score = {s: np.c_[Xs[s], np.ones(len(Xs[s]))] @ w for s in SPLITS}

    # ---------- per-AR aggregation + report --------------------------------
    def per_ar(split, scores):
        g = W[split][2]
        out = {}
        for a in member[split]:
            m = g == a
            if m.any():
                out[a] = float(np.asarray(scores)[m].mean())
        return out

    with torch.no_grad():
        model_scores = {
            "physics pooled": {s: phys_head(pooled[s]).squeeze(-1).numpy() for s in SPLITS},
            "flow rollout": {s: head(summ[s]).numpy() for s in SPLITS},
        }

    results, curves = {}, {}
    for nm, sc in model_scores.items():
        agg = {s: per_ar(s, sc[s]) for s in SPLITS}
        names = {s: sorted(agg[s]) for s in SPLITS}
        y = {s: np.array([ars[a]["label"] for a in names[s]], float) for s in SPLITS}
        v = {s: np.array([agg[s][a] for a in names[s]], float) for s in SPLITS}
        pv, pt = 1 / (1 + np.exp(-v["val"])), 1 / (1 + np.exp(-v["test"]))
        thr = pick_threshold(y["val"], pv)
        results[nm] = dict(val_auc=auc(y["val"], pv), test_auc=auc(y["test"], pt),
                           test_tss=tss_at(y["test"], pt, thr), threshold=float(thr),
                           n_test=len(y["test"]), n_test_flaring=int(y["test"].sum()))
        curves[nm] = (names["test"], y["test"], pt)

    yv, yt = Ys["val"], Ys["test"]
    pv, pt = 1 / (1 + np.exp(-sharp_score["val"])), 1 / (1 + np.exp(-sharp_score["test"]))
    thr = pick_threshold(yv, pv)
    results["SHARP-4 logistic"] = dict(val_auc=auc(yv, pv), test_auc=auc(yt, pt),
                                       test_tss=tss_at(yt, pt, thr), threshold=float(thr),
                                       n_test=len(yt), n_test_flaring=int(yt.sum()))
    sharp_by_ar = dict(zip(have["test"], pt))

    comparisons = {}
    for nm, (names_t, y_t, p_t) in curves.items():
        common = [i for i, a in enumerate(names_t) if a in sharp_by_ar]
        if len(common) < 10:
            continue
        yy = y_t[common]
        aa = p_t[common]
        bb = np.array([sharp_by_ar[names_t[i]] for i in common])
        d, lo, hi, pb = paired_bootstrap(yy, aa, bb)
        comparisons[nm] = dict(vs="SHARP-4", delta_auc=d, ci_low=lo, ci_high=hi,
                               p_better=pb, n_common=len(common))

    print("\n=== REAL SPLITS (threshold from val, reported on test) ===")
    print(f"{'model':<22}{'val AUC':>9}{'test AUC':>10}{'test TSS':>10}")
    for nm, r in results.items():
        print(f"{nm:<22}{r['val_auc']:>9.3f}{r['test_auc']:>10.3f}{r['test_tss']:>10.3f}")
    print("\n=== vs SHARP-4, paired bootstrap over test ARs ===")
    for nm, c in comparisons.items():
        print(f"{nm:<22} dAUC {c['delta_auc']:+.3f}  "
              f"95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
              f"P(better) {c['p_better']:.3f}")

    payload = dict(results=results, comparisons=comparisons,
                   n={s: len(member[s]) for s in SPLITS},
                   n_flaring={s: int(sum(ars[a]["label"] for a in member[s])) for s in SPLITS},
                   note="physics branch only; Surya arm not yet encoded for train/test")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\ntotal {time.perf_counter() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
