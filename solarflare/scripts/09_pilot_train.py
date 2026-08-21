"""Stages B-E on the real two-AR pilot data, plus the five comparison models.

IN-SAMPLE THROUGHOUT. Two ARs cannot support a train/test split, so every
number here is fit and evaluated on the same data. Per the pilot spec this is
a plumbing and diagnostics run: no TSS/BSS/AUC, no confidence intervals, and
the flaring AR is not expected to score higher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.checkpoints import assert_no_grads, freeze  # noqa: E402
from flare_forecaster.config import Config, resolve_surya_dim  # noqa: E402
from flare_forecaster.models.deterministic import DeterministicForecaster  # noqa: E402
from flare_forecaster.models.flare_head import RolloutAwareFlareHead  # noqa: E402
from flare_forecaster.models.flow_transformer import FlowMatchingTransformer  # noqa: E402
from flare_forecaster.models.fusion import FusionMLP  # noqa: E402
from flare_forecaster.models.observed_head import ObservedHistoryHead  # noqa: E402
from flare_forecaster.models.rollout import rollout, summarize_rollouts  # noqa: E402
from flare_forecaster.utils.seed import set_seed  # noqa: E402

LABELS = {"11429": 1.0, "11431": 0.0}


def windows(z_surya, z_physics, history):
    """Sliding windows. z_physics is one vector repeated across the history."""
    n_frames = z_surya.shape[0]
    out = []
    for end in range(history, n_frames + 1):
        out.append(
            (
                z_surya[end - history : end],
                z_physics.unsqueeze(0).expand(history, -1),
            )
        )
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 2000) -> np.ndarray:
    """Plain logistic regression with L2, no sklearn dependency."""
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        w -= 0.1 * (x.T @ (p - y) / len(y) + 1e-3 * w)
    return w


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts_pilot")
    parser.add_argument("--sharp", default="flare_forecaster/cache/dataset_sharp.json")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rollout-samples", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.artifacts)
    cfg = resolve_surya_dim(Config(), 1280)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    started = time.perf_counter()
    timings: dict[str, float] = {}

    surya = torch.load(root / "caches" / "pilot_surya.pt", weights_only=False)
    physics = torch.load(root / "caches" / "pilot_physics.pt", weights_only=False)
    noaa_list = list(surya["z_surya"].keys())
    z_phys = {n: physics["z_physics"][i] for i, n in enumerate(physics["noaa"])}
    print(f"ARs {noaa_list} | history {args.history}")

    xs, xp, ys, groups = [], [], [], []
    for noaa in noaa_list:
        for a, b in windows(surya["z_surya"][noaa], z_phys[noaa], args.history):
            xs.append(a)
            xp.append(b)
            ys.append(LABELS[noaa])
            groups.append(noaa)
    z_surya = torch.stack(xs)
    z_physics = torch.stack(xp)
    labels = torch.tensor(ys)
    delta = torch.ones(len(ys), args.history)
    valid = torch.ones(len(ys), args.history, dtype=torch.bool)
    print(f"windows {tuple(z_surya.shape)} | labels {labels.tolist()}")

    pos_weight = torch.tensor(
        float((labels == 0).sum()) / max(float((labels == 1).sum()), 1.0)
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    curves: dict[str, list[float]] = {}

    def run(name, module, loss_fn, epochs, lr):
        params = [p for p in module.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        history_curve = []
        t0 = time.perf_counter()
        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            history_curve.append(float(loss))
            if epoch % max(epochs // 5, 1) == 0 or epoch == epochs - 1:
                print(f"  [{name}] epoch {epoch:>4} loss {float(loss):.5f}")
        curves[name] = history_curve
        timings[name] = time.perf_counter() - t0
        return history_curve[-1]

    # ---------------- Stage B ------------------------------------------------
    fusion = FusionMLP(cfg.d_surya, cfg.d_physics, cfg.fusion_hidden,
                       cfg.d_fused, cfg.fusion_dropout)
    obs_head = ObservedHistoryHead(cfg.d_fused)
    print("\nStage B: fusion + observed head")
    run("fusion", torch.nn.ModuleList([fusion, obs_head]),
        lambda: bce(obs_head(fusion(z_surya, z_physics), valid), labels,
                    pos_weight=pos_weight),
        args.epochs, cfg.lr_fusion)
    freeze(fusion)
    with torch.no_grad():
        fused = fusion(z_surya, z_physics)
        p_observed = torch.sigmoid(obs_head(fused, valid))

    history = fused[:, :-1]
    hist_delta = delta[:, :-1]
    hist_valid = valid[:, :-1]
    target = fused[:, -1]

    # ---------------- Stage C ------------------------------------------------
    determ = DeterministicForecaster(cfg.d_fused, cfg.deterministic_hidden,
                                     cfg.deterministic_layers)
    print("\nStage C: deterministic forecaster")
    run("deterministic", determ,
        lambda: torch.nn.functional.huber_loss(determ(history, hist_delta), target),
        args.epochs, cfg.lr_deterministic)
    freeze(determ)
    with torch.no_grad():
        residual = target - determ(history, hist_delta)
    residual_median = residual.median(dim=0).values
    residual_scale = ((residual - residual_median).abs().median(dim=0).values
                      .mul(1.4826).clamp_min(1e-4))

    # ---------------- Stage D ------------------------------------------------
    flow = FlowMatchingTransformer(cfg.d_fused, cfg.flow_width, cfg.flow_layers,
                                   cfg.flow_heads, cfg.flow_ff, cfg.flow_dropout,
                                   max_history=args.history)

    def flow_loss():
        with torch.no_grad():
            mu = determ(history, hist_delta)
        eps = torch.randn_like(mu)
        x0 = mu + cfg.base_noise_scale * residual_scale * eps
        s = torch.rand(target.shape[0])
        xs_ = (1 - s[:, None]) * x0 + s[:, None] * target
        return torch.nn.functional.mse_loss(
            flow(xs_, s, history, hist_delta, hist_valid), target - x0
        )

    print("\nStage D: flow matching")
    run("flow", flow, flow_loss, args.epochs, cfg.lr_flow)
    assert_no_grads(determ, "deterministic")
    freeze(flow)

    # ---------------- rollouts -----------------------------------------------
    print(f"\ngenerating {args.rollout_samples} trajectories x {cfg.forecast_steps}h")
    t0 = time.perf_counter()
    ensemble = rollout(determ, flow, history, hist_delta, hist_valid, residual_scale,
                       future_steps=cfg.forecast_steps, samples=args.rollout_samples,
                       base_noise_scale=cfg.base_noise_scale, n_flow_steps=args.flow_steps)
    timings["rollout"] = time.perf_counter() - t0
    summary = summarize_rollouts(history, hist_valid, ensemble)
    print(f"ensemble {tuple(ensemble.shape)} -> summary {tuple(summary.shape)} "
          f"({timings['rollout']:.0f}s)")

    # deterministic-only future, for model 4
    with torch.no_grad():
        det_future = []
        h, d, v = history.clone(), hist_delta.clone(), hist_valid.clone()
        for _ in range(cfg.forecast_steps):
            nxt = determ(h, d)
            det_future.append(nxt)
            h = torch.cat([h[:, 1:], nxt.unsqueeze(1)], dim=1)
            d = torch.cat([d[:, 1:], torch.ones_like(d[:, :1])], dim=1)
            v = torch.cat([v[:, 1:], torch.ones_like(v[:, :1])], dim=1)
        det_future = torch.stack(det_future, dim=1)
    det_summary = summarize_rollouts(history, hist_valid, det_future.unsqueeze(0).repeat(2, 1, 1, 1))

    # ---------------- Stage E ------------------------------------------------
    flare_head = RolloutAwareFlareHead(cfg.d_fused)
    print("\nStage E: rollout-aware flare head")
    run("flare_head", flare_head,
        lambda: bce(flare_head(summary), labels, pos_weight=pos_weight),
        args.epochs, cfg.lr_flare_head)
    with torch.no_grad():
        p_flow = torch.sigmoid(flare_head(summary))

    det_head = RolloutAwareFlareHead(cfg.d_fused)
    print("\nModel 4: deterministic-rollout head")
    run("det_head", det_head,
        lambda: bce(det_head(det_summary), labels, pos_weight=pos_weight),
        args.epochs, cfg.lr_flare_head)
    with torch.no_grad():
        p_det = torch.sigmoid(det_head(det_summary))

    # ---------------- Model 2: Surya only ------------------------------------
    surya_head = torch.nn.Sequential(
        torch.nn.Linear(cfg.d_surya, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
    )
    pooled_surya = z_surya.mean(dim=1)
    print("\nModel 2: Surya only")
    run("surya_only", surya_head,
        lambda: bce(surya_head(pooled_surya).squeeze(-1), labels, pos_weight=pos_weight),
        args.epochs, 3e-4)
    with torch.no_grad():
        p_surya = torch.sigmoid(surya_head(pooled_surya).squeeze(-1))

    # ---------------- Model 1: SHARP baseline --------------------------------
    sharp_rows = json.loads(Path(args.sharp).read_text())
    by_ar: dict[str, list] = {}
    for r in sharp_rows:
        by_ar.setdefault(str(r["ar_id"]), []).append(r["features"])
    sharp_x, sharp_y = [], []
    for noaa in noaa_list:
        feats = np.array(by_ar[noaa], dtype=np.float64)
        for _ in range(sum(1 for g in groups if g == noaa)):
            sharp_x.append(feats.mean(axis=0))
            sharp_y.append(LABELS[noaa])
    sharp_x = np.array(sharp_x)
    sharp_x = (sharp_x - sharp_x.mean(0)) / (sharp_x.std(0) + 1e-9)
    w = fit_logistic(sharp_x, np.array(sharp_y))
    p_sharp = 1.0 / (1.0 + np.exp(-(np.concatenate(
        [sharp_x, np.ones((len(sharp_x), 1))], axis=1) @ w)))

    # ---------------- diagnostics --------------------------------------------
    diagnostics = {}
    for noaa in noaa_list:
        mask = torch.tensor([g == noaa for g in groups])
        ens = ensemble[:, mask]
        flat = ens.reshape(ens.shape[0], -1)
        pairwise = torch.cdist(flat, flat)
        n = pairwise.shape[0]
        diagnostics[noaa] = {
            "ensemble_variance": float(ens.var(dim=0).mean()),
            "mean_pairwise_distance": float(pairwise.sum() / (n * (n - 1))),
            "n_nonfinite": int((~torch.isfinite(ens)).sum()),
            "all_identical": bool(float(pairwise.max()) < 1e-8),
            "det_vs_ensemble_mean_final": float(
                (det_future[mask][:, -1] - ens.mean(dim=0)[:, -1]).norm(dim=-1).mean()
            ),
        }

    probabilities = {}
    for noaa in noaa_list:
        mask = torch.tensor([g == noaa for g in groups])
        probabilities[noaa] = {
            "SHARP baseline": float(np.mean(p_sharp[mask.numpy()])),
            "Surya only": float(p_surya[mask].mean()),
            "Surya + PINN observed": float(p_observed[mask].mean()),
            "Deterministic rollout": float(p_det[mask].mean()),
            "Flow-matching rollout": float(p_flow[mask].mean()),
        }

    timings["total"] = time.perf_counter() - started
    payload = {
        "noaa": noaa_list,
        "labels": {n: LABELS[n] for n in noaa_list},
        "groups": groups,
        "curves": curves,
        "probabilities": probabilities,
        "diagnostics": diagnostics,
        "timings": timings,
        "fused": fused,
        "ensemble": ensemble,
        "det_future": det_future,
        "history": history,
        "physics_diagnostics": physics["diagnostics"],
        "readout": physics["readout"],
        "history_steps": args.history,
    }
    torch.save(payload, root / "caches" / "pilot_results.pt")

    print("\n=== probabilities (IN-SAMPLE, 2 ARs) ===")
    models = list(next(iter(probabilities.values())).keys())
    print(f"{'model':<24}" + "".join(f"{n:>12}" for n in noaa_list))
    for m in models:
        print(f"{m:<24}" + "".join(f"{probabilities[n][m]:>12.4f}" for n in noaa_list))
    print("\n=== rollout diagnostics ===")
    for noaa, d in diagnostics.items():
        print(f"{noaa}: " + "  ".join(f"{k}={v}" for k, v in d.items()))
    print(f"\ntimings: " + "  ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
    print(f"\nwrote {root/'caches'/'pilot_results.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
