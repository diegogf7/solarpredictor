"""Run stages B-E end to end on toy latents. CPU, minutes, no GPU.

This is a plumbing test, not a science run: it answers "does the training
machinery work" before any compute is committed. It trains on the train split
only and prints training losses. Metrics stay locked behind
scripts/16_evaluate.py exactly as in the real pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.checkpoints import assert_no_grads, freeze
from flare_forecaster.config import Config, resolve_surya_dim
from flare_forecaster.data.latent_dataset import LatentSequenceDataset, collate
from flare_forecaster.models.deterministic import DeterministicForecaster
from flare_forecaster.models.flare_head import RolloutAwareFlareHead
from flare_forecaster.models.flow_transformer import FlowMatchingTransformer
from flare_forecaster.models.fusion import FusionMLP
from flare_forecaster.models.observed_head import ObservedHistoryHead
from flare_forecaster.models.rollout import rollout, summarize_rollouts
from flare_forecaster.training.common import run_stage
from flare_forecaster.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", default="artifacts_toy/caches/toy_latents.pt")
    parser.add_argument("--artifacts", default="artifacts_toy")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--rollout-samples", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=4)
    args = parser.parse_args()

    cfg = resolve_surya_dim(Config(), 1280)
    set_seed(cfg.seed)
    device = torch.device("cpu")

    dataset = LatentSequenceDataset(
        args.latents,
        split="train",
        history_steps=cfg.history_steps,
        artifacts=Path(args.artifacts),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    pos_weight = dataset.positive_weight
    n_pos = int(sum(w["label"] for w in dataset.windows))
    print(
        f"train windows {len(dataset)} over {len(set(dataset.ar_ids))} ARs "
        f"| {n_pos} positive | pos_weight {float(pos_weight):.2f}"
    )

    fusion = FusionMLP(
        cfg.d_surya, cfg.d_physics, cfg.fusion_hidden, cfg.d_fused, cfg.fusion_dropout
    )
    obs_head = ObservedHistoryHead(cfg.d_fused)
    determ = DeterministicForecaster(
        cfg.d_fused, cfg.deterministic_hidden, cfg.deterministic_layers
    )
    flow = FlowMatchingTransformer(
        cfg.d_fused,
        cfg.flow_width,
        cfg.flow_layers,
        cfg.flow_heads,
        cfg.flow_ff,
        cfg.flow_dropout,
        max_history=cfg.history_steps,
    )
    flare_head = RolloutAwareFlareHead(cfg.d_fused)

    # ---------------- Stage B: fusion + temporary observed head -----------
    class StageB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion, self.head = fusion, obs_head

    def loss_b(model, batch):
        batch = batch.to(device)
        fused = model.fusion(batch.z_surya, batch.z_physics)
        logits = model.head(fused, batch.frame_valid)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch.label, pos_weight=pos_weight
        )
        return loss, batch.label.shape[0]

    run_stage(
        "B fusion", StageB(), loader, loss_b, args.epochs, cfg.lr_fusion,
        cfg.weight_decay, cfg.grad_clip, device, log_every=5,
    )
    freeze(fusion)

    # ---------------- cache fused latents ---------------------------------
    with torch.no_grad():
        fused_all, delta_all, valid_all, label_all = [], [], [], []
        for batch in DataLoader(
            dataset, batch_size=cfg.batch_size, collate_fn=collate
        ):
            fused_all.append(fusion(batch.z_surya, batch.z_physics))
            delta_all.append(batch.delta_hours)
            valid_all.append(batch.frame_valid)
            label_all.append(batch.label)
    fused_all = torch.cat(fused_all)
    delta_all = torch.cat(delta_all)
    valid_all = torch.cat(valid_all)
    label_all = torch.cat(label_all)
    print(f"\ncached fused latents {tuple(fused_all.shape)}")

    history = fused_all[:, :-1]
    hist_delta = delta_all[:, :-1]
    hist_valid = valid_all[:, :-1]
    target = fused_all[:, -1]

    latent_ds = torch.utils.data.TensorDataset(
        history, hist_delta, hist_valid, target, label_all
    )
    latent_loader = DataLoader(latent_ds, batch_size=cfg.batch_size, shuffle=True)

    # ---------------- Stage C: deterministic ------------------------------
    def loss_c(model, batch):
        h, d, _, y, _ = batch
        return torch.nn.functional.huber_loss(model(h, d), y, delta=1.0), h.shape[0]

    run_stage(
        "C deterministic", determ, latent_loader, loss_c, args.epochs,
        cfg.lr_deterministic, cfg.weight_decay, cfg.grad_clip, device, log_every=5,
    )
    freeze(determ)

    with torch.no_grad():
        residual = target - determ(history, hist_delta)
    residual_median = residual.median(dim=0).values
    residual_scale = (
        (residual - residual_median).abs().median(dim=0).values.mul(1.4826).clamp_min(1e-4)
    )
    print(f"residual_scale  median {float(residual_scale.median()):.4f}")

    # ---------------- Stage D: flow matching ------------------------------
    def loss_d(model, batch):
        h, d, v, y, _ = batch
        with torch.no_grad():
            mu = determ(h, d)
        eps = torch.randn_like(mu)
        x0 = mu + cfg.base_noise_scale * residual_scale * eps
        s = torch.rand(y.shape[0], device=y.device)
        xs = (1 - s[:, None]) * x0 + s[:, None] * y
        v_pred = model(xs, s, h, d, v)
        return torch.nn.functional.mse_loss(v_pred, y - x0), h.shape[0]

    run_stage(
        "D flow", flow, latent_loader, loss_d, args.epochs, cfg.lr_flow,
        cfg.weight_decay, cfg.grad_clip, device, log_every=5,
    )
    assert_no_grads(determ, "deterministic")
    freeze(flow)

    # ---------------- Stage E: rollout-aware flare head -------------------
    print("\ngenerating rollouts (this is the slow part)")
    # Windows are ordered by AR, so a leading slice would be all-negative and
    # the head would trivially learn "never flares". Sample at random.
    subset = torch.randperm(history.shape[0])[: min(256, history.shape[0])]
    print(f"rollout subset: {int(label_all[subset].sum())} positive of {len(subset)}")
    ens = rollout(
        determ, flow,
        history[subset], hist_delta[subset], hist_valid[subset],
        residual_scale,
        future_steps=cfg.forecast_steps,
        samples=args.rollout_samples,
        base_noise_scale=cfg.base_noise_scale,
        n_flow_steps=args.flow_steps,
    )
    summary = summarize_rollouts(history[subset], hist_valid[subset], ens)
    print(f"rollout {tuple(ens.shape)} -> summary {tuple(summary.shape)}")

    head_loader = DataLoader(
        torch.utils.data.TensorDataset(summary, label_all[subset]),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    def loss_e(model, batch):
        x, y = batch
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x), y, pos_weight=pos_weight
        )
        return loss, x.shape[0]

    run_stage(
        "E flare head", flare_head, head_loader, loss_e, args.epochs,
        cfg.lr_flare_head, cfg.weight_decay, cfg.grad_clip, device, log_every=5,
    )

    with torch.no_grad():
        probs = torch.sigmoid(flare_head(summary))
    print(
        f"\nP(M/X within 24h) on TRAIN windows: "
        f"min {float(probs.min()):.3f}  mean {float(probs.mean()):.3f}  "
        f"max {float(probs.max()):.3f}"
    )
    print("\nPIPELINE RAN END TO END (stages B-E). No metrics -- eval stays locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
