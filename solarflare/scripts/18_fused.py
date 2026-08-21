"""Two-branch: PINN readouts + Surya latents, fused, through the flow stack.

This is the fused pipeline 16_mini_pipeline.py describes, run on the frozen
AR splits instead of a sub-divided one -- but restricted to the ARs whose
Surya latents have actually been encoded (the campaign job is unfinished, so
train and test are partial). Every model here is scored on the SAME AR set,
which is the point: it isolates "does adding Surya help" from "does having
more ARs help".

--roles frozen : the pre-registered partition, restricted to available ARs.
--roles swap   : train on the val-split ARs (the only complete Surya split),
                 select on the train-split ARs, test on the test-split ARs.
                 AR-disjoint either way, but NOT pre-registered -- provisional.

The physics autoencoder is self-supervised, so it trains on ALL train-split
readouts regardless of whether Surya covers them.
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
from flare_forecaster.models.fusion import FusionMLP  # noqa: E402
from flare_forecaster.models.observed_head import ObservedHistoryHead  # noqa: E402
from flare_forecaster.models.rollout import rollout, summarize_rollouts  # noqa: E402
from flare_forecaster.checkpoints import freeze  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402

SPLITS = ("train", "val", "test")
CACHE = {"train": "surya_campaign.pt", "val": "surya_campaign_val.pt", "test": "surya_campaign_test.pt"}


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
    return max(((tss_at(y, p, t), t) for t in np.unique(np.round(p, 3))))[1]


def paired_bootstrap(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = np.arange(len(y))
    d = []
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if y[take].sum() in (0, len(take)):
            continue
        v = auc(y[take], np.asarray(a)[take]) - auc(y[take], np.asarray(b)[take])
        if np.isfinite(v):
            d.append(v)
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def to_unix(x):
    if isinstance(x, str):
        return datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp()
    return float(x)


def load_surya(root):
    """Both cache formats -> {noaa: {"z": (n,1280) float, "t": (n,) unix}}."""
    out = {}
    for split, name in CACHE.items():
        p = Path(root) / name
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "packed" in d:
            for a, v in d["packed"].items():
                out[a] = {"z": v["z_surya"].float(),
                          "t": np.array([to_unix(x) for x in v["t"].tolist()])}
        elif "store" in d:
            for a, v in d["store"].items():
                if a in out or not len(v["z"]):
                    continue
                out[a] = {"z": torch.stack([z.float() for z in v["z"]]),
                          "t": np.array([to_unix(x) for x in v["t"]])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts-root", default="/work1/jeroenaudenaert/diegogon/readouts")
    ap.add_argument("--caches", default="artifacts/caches")
    ap.add_argument("--triage", default="flare_forecaster/cache/triage_dataset.json")
    ap.add_argument("--roles", choices=["frozen", "swap"], default="frozen")
    ap.add_argument("--out", default="artifacts/fused_results.json")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--ae-epochs", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--rollout-samples", type=int, default=8)
    ap.add_argument("--flow-steps", type=int, default=6)
    args = ap.parse_args()

    set_seed(42)
    t0 = time.perf_counter()
    H = args.history

    surya = load_surya(args.caches)
    print(f"surya latents available for {len(surya)} ARs")

    # ---------- readouts, all train-split ARs kept for the AE ---------------
    ars, native = {}, {s: [] for s in SPLITS}
    ae_pool = []
    for split in SPLITS:
        for fp in sorted((Path(args.readouts_root) / f"readouts_{split}").glob("noaa*.pt")):
            noaa = fp.stem[4:]
            rec = torch.load(fp, map_location="cpu", weights_only=False)
            r = rec["readout"].float()
            if split == "train":
                ae_pool.append(r)
            if noaa not in surya or r.shape[0] < H:
                continue
            s_t = surya[noaa]["t"]
            r_t = np.array([to_unix(x) for x in rec["timestamps"]])
            idx = np.abs(s_t[:, None] - r_t[None, :]).argmin(axis=1)
            keep = np.abs(s_t - r_t[idx]) < 5400
            if keep.sum() < H:
                continue
            ars[noaa] = {
                "z_surya": surya[noaa]["z"][torch.from_numpy(np.nonzero(keep)[0])],
                "readout": r[torch.from_numpy(idx[keep])],
                "label": int(rec["label"]),
            }
            native[split].append(noaa)

    roles = ({"train": "train", "val": "val", "test": "test"} if args.roles == "frozen"
             else {"train": "val", "val": "train", "test": "test"})
    member = {r: native[src] for r, src in roles.items()}
    print(f"roles={args.roles}  (train<-{roles['train']}, val<-{roles['val']}, test<-{roles['test']})")
    for s in SPLITS:
        print(f"  {s:5} {len(member[s]):3} ARs, {sum(ars[a]['label'] for a in member[s]):2} flaring")
    assert not (set(member["train"]) & set(member["test"])), "AR leak"
    for s in SPLITS:
        if not member[s]:
            print(f"FATAL: {s} empty")
            return 1

    # ---------- Stage A: physics AE on all train-split readouts -------------
    res = args.resolution

    def pool(r):
        return torch.nn.functional.avg_pool2d(r, r.shape[-1] // res)

    train_maps = torch.cat([pool(r) for r in ae_pool])
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

    enc = PhysicsEncoder(len(PHYSICS_CHANNELS), 256)
    dec = PhysicsDecoder(256, len(PHYSICS_CHANNELS), base=res // 16)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)
    xtr = scale(train_maps)
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
            ars[a]["z_phys"] = enc(scale(pool(ars[a]["readout"])))

    # ---------- windows ----------------------------------------------------
    def windows(names):
        zs, zp, y, g = [], [], [], []
        for a in names:
            S, P, L = ars[a]["z_surya"], ars[a]["z_phys"], ars[a]["label"]
            n = min(len(S), len(P))
            for e in range(H, n + 1):
                zs.append(S[e - H:e])
                zp.append(P[e - H:e])
                y.append(float(L))
                g.append(a)
        return torch.stack(zs), torch.stack(zp), torch.tensor(y), np.array(g)

    W = {s: windows(member[s]) for s in SPLITS}
    for s in SPLITS:
        print(f"  {s:5} {W[s][0].shape[0]:5} windows")
    d_s, d_p = W["train"][0].shape[-1], W["train"][1].shape[-1]
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    ytr = W["train"][2]

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
                print(f"  [{name}] {ep:>4} {float(l.detach()):.5f}")

    # ---------- Stage B: fusion + observed head ----------------------------
    fusion = FusionMLP(d_s, d_p, 512, 256, 0.10)
    obs = ObservedHistoryHead(256)
    vmask = torch.ones(W["train"][0].shape[0], H, dtype=torch.bool)
    print("\n[B] fusion + observed head")
    run("fusion", torch.nn.ModuleList([fusion, obs]),
        lambda: bce(obs(fusion(W["train"][0], W["train"][1]), vmask), ytr, pos_weight=pw(ytr)),
        args.epochs, 3e-4)
    freeze(fusion)

    with torch.no_grad():
        F = {s: fusion(W[s][0], W[s][1]) for s in SPLITS}
    hist = {s: F[s][:, :-1] for s in SPLITS}
    tgt = {s: F[s][:, -1] for s in SPLITS}
    dl = {s: torch.ones(F[s].shape[0], H - 1) for s in SPLITS}
    vm = {s: torch.ones(F[s].shape[0], H - 1, dtype=torch.bool) for s in SPLITS}

    # ---------- Stage C/D/E ------------------------------------------------
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
            flow(xs, s, hist["train"], dl["train"], vm["train"]), x1 - x0)

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
    run("flare", head, lambda: bce(head(summ["train"]), ytr, pos_weight=pw(ytr)), args.epochs, 3e-4)

    # ---------- single-branch comparators, same ARs ------------------------
    mlp = lambda d: torch.nn.Sequential(
        torch.nn.Linear(d, 128), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))
    ps = {s: W[s][1].mean(1) for s in SPLITS}
    ss = {s: W[s][0].mean(1) for s in SPLITS}
    phys_head, surya_head = mlp(d_p), mlp(d_s)
    print("\n[cmp] single-branch heads")
    run("physics", phys_head,
        lambda: bce(phys_head(ps["train"]).squeeze(-1), ytr, pos_weight=pw(ytr)), args.epochs, 3e-4)
    run("surya", surya_head,
        lambda: bce(surya_head(ss["train"]).squeeze(-1), ytr, pos_weight=pw(ytr)), args.epochs, 3e-4)

    # ---------- SHARP, same ARs --------------------------------------------
    triage = {str(r["noaa"]): r for r in json.load(open(args.triage))}
    have = {s: [a for a in member[s] if a in triage] for s in SPLITS}
    X = {s: np.log10(np.abs(np.array([triage[a]["sharp"] for a in have[s]], float)) + 1e-9) for s in SPLITS}
    Y = {s: np.array([ars[a]["label"] for a in have[s]], float) for s in SPLITS}
    mu, sd = X["train"].mean(0), X["train"].std(0).clip(1e-9)
    X = {s: (v - mu) / sd for s, v in X.items()}
    Xtr = np.c_[X["train"], np.ones(len(X["train"]))]
    w = np.zeros(Xtr.shape[1])
    posw = (Y["train"] == 0).sum() / max((Y["train"] == 1).sum(), 1)
    sw = np.where(Y["train"] == 1, posw, 1.0)
    for _ in range(4000):
        p = 1 / (1 + np.exp(-Xtr @ w))
        w -= 1e-2 * (Xtr.T @ (sw * (p - Y["train"]))) / len(Y["train"])
    sharp = {s: np.c_[X[s], np.ones(len(X[s]))] @ w for s in SPLITS}

    # ---------- per-AR aggregation + report --------------------------------
    def per_ar(split, scores):
        g = W[split][3]
        return {a: float(np.asarray(scores)[g == a].mean()) for a in member[split] if (g == a).any()}

    with torch.no_grad():
        raw = {
            "physics only": {s: phys_head(ps[s]).squeeze(-1).numpy() for s in SPLITS},
            "Surya only": {s: surya_head(ss[s]).squeeze(-1).numpy() for s in SPLITS},
            "fused observed": {s: obs(F[s], torch.ones(F[s].shape[0], H, dtype=torch.bool)).numpy() for s in SPLITS},
            "fused flow rollout": {s: head(summ[s]).numpy() for s in SPLITS},
        }

    results, test_p = {}, {}
    for nm, sc in raw.items():
        agg = {s: per_ar(s, sc[s]) for s in SPLITS}
        names = {s: sorted(agg[s]) for s in SPLITS}
        y = {s: np.array([ars[a]["label"] for a in names[s]], float) for s in SPLITS}
        pv = 1 / (1 + np.exp(-np.array([agg["val"][a] for a in names["val"]])))
        pt = 1 / (1 + np.exp(-np.array([agg["test"][a] for a in names["test"]])))
        thr = pick_threshold(y["val"], pv)
        results[nm] = dict(val_auc=auc(y["val"], pv), test_auc=auc(y["test"], pt),
                           test_tss=tss_at(y["test"], pt, thr), threshold=float(thr))
        test_p[nm] = dict(zip(names["test"], pt))

    pv = 1 / (1 + np.exp(-sharp["val"]))
    pt = 1 / (1 + np.exp(-sharp["test"]))
    thr = pick_threshold(Y["val"], pv)
    results["SHARP-4 logistic"] = dict(val_auc=auc(Y["val"], pv), test_auc=auc(Y["test"], pt),
                                       test_tss=tss_at(Y["test"], pt, thr), threshold=float(thr))
    test_p["SHARP-4 logistic"] = dict(zip(have["test"], pt))

    ytest = {a: ars[a]["label"] for a in member["test"]}
    comparisons = {}
    for nm in ("fused flow rollout", "fused observed", "Surya only"):
        for ref in ("SHARP-4 logistic", "physics only"):
            common = sorted(set(test_p[nm]) & set(test_p[ref]))
            if len(common) < 10:
                continue
            yy = np.array([ytest[a] for a in common], float)
            d, lo, hi, pb = paired_bootstrap(yy, [test_p[nm][a] for a in common],
                                             [test_p[ref][a] for a in common])
            comparisons[f"{nm} vs {ref}"] = dict(delta_auc=d, ci_low=lo, ci_high=hi,
                                                 p_better=pb, n_common=len(common))

    print(f"\n=== TWO-BRANCH, roles={args.roles} (threshold from val, reported on test) ===")
    print(f"{'model':<22}{'val AUC':>9}{'test AUC':>10}{'test TSS':>10}")
    for nm, r in results.items():
        print(f"{nm:<22}{r['val_auc']:>9.3f}{r['test_auc']:>10.3f}{r['test_tss']:>10.3f}")
    print("\n=== paired bootstrap over test ARs ===")
    for nm, c in comparisons.items():
        print(f"{nm:<42} dAUC {c['delta_auc']:+.3f} CI [{c['ci_low']:+.3f},{c['ci_high']:+.3f}] "
              f"P(better) {c['p_better']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(roles=args.roles, results=results, comparisons=comparisons,
             n={s: len(member[s]) for s in SPLITS},
             n_flaring={s: int(sum(ars[a]["label"] for a in member[s])) for s in SPLITS},
             note="Surya campaign incomplete; restricted to encoded ARs"), indent=2))
    print(f"\ntotal {time.perf_counter() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
