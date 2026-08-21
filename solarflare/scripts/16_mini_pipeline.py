"""End-to-end smoke run of the FULL pipeline on real data, laptop-sized.

Uses the one split that finished cleanly (val: 59 ARs with both Surya latents
and PINN readouts) and sub-splits it by active region. Every downstream stage
that has never touched real data runs here: Stage A, fusion, deterministic,
flow matching, rollout, and the baseline ladder.

NOT a scientific result -- these 59 ARs were meant to be a single split, so
carving them up gives no held-out guarantee against the real experiment. This
exists to find bugs before another 12-hour campaign.
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
from flare_forecaster.checkpoints import freeze, assert_no_grads  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    order = np.argsort(s); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for i in np.nonzero(cnt > 1)[0]:
        m = inv == i; ranks[m] = ranks[m].mean()
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0: return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tss_at(y, p, thr):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = ((y == 1) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum(); tn = ((y == 0) & (pred == 0)).sum()
    return float(tp / max(tp + fn, 1) - fp / max(fp + tn, 1))


def to_unix(stamp: str) -> float:
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).timestamp()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surya", default="artifacts/caches/surya_campaign_val.pt")
    ap.add_argument("--readouts", default="artifacts/readouts_val")
    ap.add_argument("--triage", default="flare_forecaster/cache/triage_dataset.json")
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--ae-epochs", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--rollout-samples", type=int, default=8)
    ap.add_argument("--flow-steps", type=int, default=6)
    args = ap.parse_args()

    set_seed(42)
    t0 = time.perf_counter()
    dev = torch.device("cpu")

    # ---------- load both arms, align by timestamp ------------------------
    surya = torch.load(args.surya, map_location="cpu", weights_only=False)["packed"]
    ars = {}
    for fp in sorted(Path(args.readouts).glob("noaa*.pt")):
        noaa = fp.stem[4:]
        if noaa not in surya:
            continue
        d = torch.load(fp, map_location="cpu", weights_only=False)
        r_t = np.array([to_unix(s) for s in d["timestamps"]])
        s_t = surya[noaa]["t"].numpy()
        # nearest readout within half a cadence step
        idx = np.abs(s_t[:, None] - r_t[None, :]).argmin(axis=1)
        keep = np.abs(s_t - r_t[idx]) < 5400
        if keep.sum() < args.history + 1:
            continue
        ars[noaa] = {
            "z_surya": surya[noaa]["z_surya"][torch.from_numpy(np.nonzero(keep)[0])],
            "readout": d["readout"][torch.from_numpy(idx[keep])].float(),
            "t": s_t[keep],
            "label": int(d["label"]),
        }
    print(f"aligned {len(ars)} ARs "
          f"({sum(v['label'] for v in ars.values())} flaring) "
          f"| {sum(len(v['t']) for v in ars.values())} frames")

    # ---------- AR-grouped chronological sub-split -------------------------
    order = sorted(ars, key=lambda k: ars[k]["t"][0])
    n = len(order)
    sub = {"train": order[: int(.6 * n)], "val": order[int(.6 * n): int(.8 * n)], "test": order[int(.8 * n):]}
    for k, v in sub.items():
        print(f"  {k:5} {len(v):2} ARs, {sum(ars[a]['label'] for a in v):2} flaring")
    assert not (set(sub["train"]) & set(sub["test"])), "AR leak"

    # ---------- Stage A: physics autoencoder (train ARs only) --------------
    res = args.resolution
    def prep(a):
        r = ars[a]["readout"]
        return torch.nn.functional.avg_pool2d(r, r.shape[-1] // res)
    train_maps = torch.cat([prep(a) for a in sub["train"]])
    flat = train_maps.permute(1, 0, 2, 3).reshape(len(PHYSICS_CHANNELS), -1)
    med = flat.median(dim=1).values
    iqr = (flat.quantile(.75, dim=1) - flat.quantile(.25, dim=1)).clamp_min(1e-9)
    def scale(x):
        y = ((x - med[:, None, None]) / iqr[:, None, None]).clamp(-10, 10)
        y = torch.nan_to_num(y)
        for nm in ("trace_valid", "strong_field_mask"):
            i = PHYSICS_CHANNELS.index(nm); y[:, i] = x[:, i]
        return y

    enc, dec = PhysicsEncoder(len(PHYSICS_CHANNELS), 256), PhysicsDecoder(256, len(PHYSICS_CHANNELS), base=res // 16)
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
            loss.backward(); opt.step(); tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == args.ae_epochs - 1:
            print(f"  epoch {ep:>3}  masked_huber {tot/len(perm):.5f}")
    freeze(enc); enc.eval()

    with torch.no_grad():
        for a in ars:
            ars[a]["z_phys"] = enc(scale(prep(a)))
    print(f"  z_physics {ars[order[0]]['z_phys'].shape}")

    # ---------- windows ----------------------------------------------------
    def windows(names):
        zs, zp, y, g = [], [], [], []
        for a in names:
            S, P, L = ars[a]["z_surya"].float(), ars[a]["z_phys"], ars[a]["label"]
            for e in range(args.history, len(S) + 1):
                zs.append(S[e - args.history:e]); zp.append(P[e - args.history:e])
                y.append(float(L)); g.append(a)
        return (torch.stack(zs), torch.stack(zp), torch.tensor(y), g)

    W = {k: windows(v) for k, v in sub.items()}
    for k in sub: print(f"  {k:5} {W[k][0].shape[0]:4} windows")
    d_s, d_p = W["train"][0].shape[-1], W["train"][1].shape[-1]
    H = args.history
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    pw = lambda y: torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))

    def run(name, mod, lossfn, epochs, lr):
        ps = [p for p in mod.parameters() if p.requires_grad]
        o = torch.optim.AdamW(ps, lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            o.zero_grad(set_to_none=True); l = lossfn()
            if not torch.isfinite(l): raise FloatingPointError(f"{name} non-finite")
            l.backward(); torch.nn.utils.clip_grad_norm_(ps, 1.0); o.step()
            if ep % max(epochs // 3, 1) == 0 or ep == epochs - 1:
                print(f"  [{name}] {ep:>4} {float(l):.5f}")

    # ---------- Stage B ----------------------------------------------------
    fusion = FusionMLP(d_s, d_p, 512, 256, 0.10); obs = ObservedHistoryHead(256)
    zs, zp, y, _ = W["train"]
    vmask = torch.ones(zs.shape[0], H, dtype=torch.bool)
    print("\n[B] fusion + observed head")
    run("fusion", torch.nn.ModuleList([fusion, obs]),
        lambda: bce(obs(fusion(zs, zp), vmask), y, pos_weight=pw(y)), args.epochs, 3e-4)
    freeze(fusion)

    def fused(split):
        a, b, _, _ = W[split]
        with torch.no_grad(): return fusion(a, b)
    F = {k: fused(k) for k in sub}
    hist = {k: F[k][:, :-1] for k in sub}
    tgt = {k: F[k][:, -1] for k in sub}
    dl = {k: torch.ones(F[k].shape[0], H - 1) for k in sub}
    vm = {k: torch.ones(F[k].shape[0], H - 1, dtype=torch.bool) for k in sub}

    # ---------- Stage C ----------------------------------------------------
    det = DeterministicForecaster(256, 256, 2)
    print("\n[C] deterministic")
    run("determ", det, lambda: torch.nn.functional.huber_loss(det(hist["train"], dl["train"]), tgt["train"]),
        args.epochs, 3e-4)
    freeze(det)
    with torch.no_grad(): resid = tgt["train"] - det(hist["train"], dl["train"])
    rs = (resid - resid.median(0).values).abs().median(0).values.mul(1.4826).clamp_min(1e-4)

    # ---------- Stage D ----------------------------------------------------
    flow = FlowMatchingTransformer(256, 256, 4, 8, 1024, 0.10, max_history=H)
    def dloss():
        with torch.no_grad(): mu = det(hist["train"], dl["train"])
        x0 = mu + rs * torch.randn_like(mu); x1 = tgt["train"]
        s = torch.rand(x1.shape[0]); xs = (1 - s[:, None]) * x0 + s[:, None] * x1
        return torch.nn.functional.mse_loss(flow(xs, s, hist["train"], dl["train"], vm["train"]), x1 - x0)
    print("\n[D] flow matching")
    run("flow", flow, dloss, args.epochs, 2e-4)
    assert_no_grads(det, "deterministic"); freeze(flow)

    # ---------- Stage E + ladder ------------------------------------------
    print(f"\n[E] rollouts ({args.rollout_samples} traj x {H} steps)")
    summ = {}
    for k in sub:
        ens = rollout(det, flow, hist[k], dl[k], vm[k], rs, future_steps=H,
                      samples=args.rollout_samples, n_flow_steps=args.flow_steps)
        summ[k] = summarize_rollouts(hist[k], vm[k], ens)
        if k == "train":
            print(f"  ensemble {tuple(ens.shape)} finite={bool(torch.isfinite(ens).all())} "
                  f"spread={float(ens.std(0).mean()):.4f}")
    head = RolloutAwareFlareHead(256)
    run("flare", head, lambda: bce(head(summ["train"]), W["train"][2], pos_weight=pw(W["train"][2])),
        args.epochs, 3e-4)

    # rung 2-ish: physics only; rung 3: surya only; rung 5: full
    surya_head = torch.nn.Sequential(torch.nn.Linear(d_s, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
    pooled = {k: W[k][0].mean(1) for k in sub}
    run("surya-only", surya_head,
        lambda: bce(surya_head(pooled["train"]).squeeze(-1), W["train"][2], pos_weight=pw(W["train"][2])),
        args.epochs, 3e-4)
    phys_head = torch.nn.Sequential(torch.nn.Linear(d_p, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
    pooledp = {k: W[k][1].mean(1) for k in sub}
    run("physics-only", phys_head,
        lambda: bce(phys_head(pooledp["train"]).squeeze(-1), W["train"][2], pos_weight=pw(W["train"][2])),
        args.epochs, 3e-4)

    print("\n=== LADDER (threshold from val, reported on test) ===")
    print(f"{'model':<22}{'val AUC':>9}{'test AUC':>10}{'test TSS':>10}")
    with torch.no_grad():
        models = {
            "physics only": lambda k: phys_head(pooledp[k]).squeeze(-1),
            "Surya only": lambda k: surya_head(pooled[k]).squeeze(-1),
            "fused observed": lambda k: obs(F[k], torch.ones(F[k].shape[0], H, dtype=torch.bool)),
            "flow rollout": lambda k: head(summ[k]),
        }
        out = {}
        for nm, fn in models.items():
            sv, st = fn("val").numpy(), fn("test").numpy()
            yv, yt = W["val"][2].numpy(), W["test"][2].numpy()
            thr = max(((tss_at(yv, 1/(1+np.exp(-sv)), t), t) for t in np.unique(np.round(1/(1+np.exp(-sv)), 3))))[1]
            out[nm] = dict(val_auc=auc(yv, sv), test_auc=auc(yt, st),
                           test_tss=tss_at(yt, 1/(1+np.exp(-st)), thr))
            print(f"{nm:<22}{out[nm]['val_auc']:>9.3f}{out[nm]['test_auc']:>10.3f}{out[nm]['test_tss']:>10.3f}")
    Path("artifacts/mini_results.json").write_text(json.dumps(out, indent=2))
    print(f"\ntotal {time.perf_counter()-t0:.0f}s | NOT a science result: val split sub-divided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
