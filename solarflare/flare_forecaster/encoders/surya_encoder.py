"""Frozen Surya-1.0 encoder wrapper.

Resolves the placeholder API in the build spec against the real
NASA-IMPACT/Surya release:

    HelioSpectFormer.forward(batch) where
        batch["ts"]               -> [B, C, T, H, W]   (channels BEFORE time)
        batch["time_delta_input"] -> [B, T]            (minutes, e.g. [-60, 0])

    finetune=True returns backbone tokens [B, L, D] and skips the unembed
    decoder. The name is backwards from what you would guess: finetune=False
    builds the full forecasting model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from surya.models.helio_spectformer import HelioSpectFormer

WEIGHT_FILE = "surya.366m.v1.pt"
CONFIG_FILE = "config.yaml"


def load_surya_config(config_path: str | Path) -> tuple[dict, list[str], dict]:
    """Translate the released config.yaml into HelioSpectFormer kwargs.

    The `model:` block has no `in_chans` or `time_embedding`; both are implied
    by the `data:` block, so they are derived rather than hard-coded.
    """
    raw = yaml.safe_load(Path(config_path).read_text())
    model, data = dict(raw["model"]), dict(raw["data"])
    channels = list(data["sdo_channels"])

    kwargs = dict(
        img_size=model["img_size"],
        patch_size=model["patch_size"],
        in_chans=len(channels),
        embed_dim=model["embed_dim"],
        time_embedding={"type": "linear", "time_dim": data["n_input_timestamps"]},
        depth=model["depth"],
        n_spectral_blocks=model["n_spectral_blocks"],
        num_heads=model["num_heads"],
        mlp_ratio=model["mlp_ratio"],
        drop_rate=model["drop_rate"],
        window_size=model["window_size"],
        dp_rank=model["dp_rank"],
        learned_flow=model.get("learned_flow", False),
        use_latitude_in_learned_flow=model.get("use_latitude_in_learned_flow", False),
        rpe=model.get("rpe", False),
        ensemble=model.get("ensemble"),
        finetune=True,  # encoder-only
    )
    return kwargs, channels, data


def build_surya_backbone(
    weights_dir: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[HelioSpectFormer, dict, list[str], dict]:
    weights_dir = Path(weights_dir)
    kwargs, channels, data = load_surya_config(weights_dir / CONFIG_FILE)

    model = HelioSpectFormer(**kwargs)
    payload = torch.load(weights_dir / WEIGHT_FILE, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    result = model.load_state_dict(state, strict=False)
    # finetune=True never builds `unembed`, so those checkpoint keys are
    # expected to be left over. Anything else means a real mismatch.
    unexpected = [k for k in result.unexpected_keys if not k.startswith("unembed.")]
    if result.missing_keys or unexpected:
        raise RuntimeError(
            f"Surya state_dict mismatch.\n"
            f"  missing:    {result.missing_keys[:8]}\n"
            f"  unexpected: {unexpected[:8]}"
        )

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, kwargs, channels, data


class SuryaEncoder(nn.Module):
    """Frozen Surya backbone + AR-restricted token pooling.

    NASA/IBM's own downstream heads pool globally over all 65,536 tokens. We
    pool only tokens inside the active-region box, which is the point of
    difference from their flare model.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        pool: str = "crop_mean",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if pool not in ("crop_mean", "crop_max"):
            raise ValueError(f"unknown pool: {pool}")

        self.model, self.kwargs, self.channels, self.data_config = build_surya_backbone(
            weights_dir, device
        )
        self.pool = pool
        self.img_size = self.kwargs["img_size"]
        self.patch_size = self.kwargs["patch_size"]
        self.embed_dim = self.kwargs["embed_dim"]
        self.token_grid = self.img_size // self.patch_size
        self.n_tokens = self.token_grid**2
        self.time_delta_input_minutes = list(self.data_config["time_delta_input_minutes"])

    @torch.inference_mode()
    def encode_tokens(
        self,
        ts: torch.Tensor,
        time_delta_input: torch.Tensor,
    ) -> torch.Tensor:
        """ts: [B, C, T, H, W] -> tokens [B, L, D]."""
        if ts.ndim != 5:
            raise AssertionError(f"ts must be [B,C,T,H,W], got {tuple(ts.shape)}")
        tokens = self.model({"ts": ts, "time_delta_input": time_delta_input})
        if tokens.ndim != 3:
            raise AssertionError(f"expected [B,L,D] tokens, got {tuple(tokens.shape)}")
        return tokens

    @torch.inference_mode()
    def forward(
        self,
        ts: torch.Tensor,
        time_delta_input: torch.Tensor,
        ar_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """-> [B, D_SURYA]."""
        tokens = self.encode_tokens(ts, time_delta_input)
        if ar_token_mask.shape != tokens.shape[:2]:
            raise AssertionError(
                f"ar_token_mask {tuple(ar_token_mask.shape)} does not match "
                f"tokens {tuple(tokens.shape[:2])}"
            )
        if ar_token_mask.dtype != torch.bool:
            raise AssertionError("ar_token_mask must be bool")

        mask = ar_token_mask.unsqueeze(-1)
        if self.pool == "crop_mean":
            numerator = (tokens * mask).sum(dim=1)
            denominator = mask.sum(dim=1).clamp_min(1)
            pooled = numerator / denominator
        else:
            filled = tokens.masked_fill(~mask, float("-inf"))
            pooled = filled.amax(dim=1)
            # An AR box that misses the token grid would otherwise yield -inf.
            pooled = torch.where(
                ar_token_mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled)
            )

        if not torch.isfinite(pooled).all():
            raise FloatingPointError("Surya pooled latent contains non-finite values")
        return pooled

    def box_to_token_mask(
        self,
        boxes: torch.Tensor,
        margin_px: int = 0,
    ) -> torch.Tensor:
        """AR pixel boxes [B,4] as (x0, y0, x1, y1) -> token mask [B, n_tokens].

        Boxes are in full-disk pixel coordinates at `img_size`. A token is
        selected when its patch overlaps the box expanded by `margin_px`.
        """
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise AssertionError(f"boxes must be [B,4], got {tuple(boxes.shape)}")

        grid = self.token_grid
        expanded = boxes.float() + torch.tensor(
            [-margin_px, -margin_px, margin_px, margin_px],
            device=boxes.device,
            dtype=torch.float32,
        )
        # patch index range covered by each box, clamped to the grid
        t0 = (expanded[:, :2] / self.patch_size).floor().clamp(0, grid - 1).long()
        t1 = (expanded[:, 2:] / self.patch_size).ceil().clamp(1, grid).long()

        mask = torch.zeros(
            boxes.shape[0], grid, grid, dtype=torch.bool, device=boxes.device
        )
        for i in range(boxes.shape[0]):
            mask[i, t0[i, 1] : t1[i, 1], t0[i, 0] : t1[i, 0]] = True
        return mask.flatten(1)
