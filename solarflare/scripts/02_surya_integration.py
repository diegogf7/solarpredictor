"""Measure the Surya encoder and resolve d_surya.

Runs one real forward pass and reads D off the token tensor. Nothing here is
guessed: the resolved config is written only from what the model actually
returned. Per spec §8 this records the checkpoint identity, channel order,
token grid, and preprocessing metadata alongside it.

Compute node only -- a 4096^2 forward is not a login-node job:

    srun -p mi3001x -t 00:20:00 --pty python scripts/02_surya_integration.py \
        --weights-dir $WORK/surya_weights
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.checkpoints import git_commit, sha256_file  # noqa: E402
from flare_forecaster.config import (  # noqa: E402
    Config,
    RESOLVED_CONFIG_PATH,
    resolve_surya_dim,
    save_resolved,
)
from flare_forecaster.encoders.surya_encoder import (  # noqa: E402
    CONFIG_FILE,
    WEIGHT_FILE,
    SuryaEncoder,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pool", default="crop_mean", choices=["crop_mean", "crop_max"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print(f"device={device} dtype={dtype}")
    encoder = SuryaEncoder(weights_dir, pool=args.pool, device=device)
    n_params = sum(p.numel() for p in encoder.model.parameters())
    print(
        f"loaded: {n_params/1e6:.1f}M params | img={encoder.img_size} "
        f"patch={encoder.patch_size} grid={encoder.token_grid}^2={encoder.n_tokens} "
        f"channels={len(encoder.channels)}"
    )

    if any(p.requires_grad for p in encoder.model.parameters()):
        raise RuntimeError("Surya backbone is not frozen")

    # --- one real forward pass -------------------------------------------
    n_time = len(encoder.time_delta_input_minutes)
    ts = torch.zeros(
        1,
        len(encoder.channels),
        n_time,
        encoder.img_size,
        encoder.img_size,
        dtype=dtype,
        device=device,
    )
    dt = torch.tensor(
        [encoder.time_delta_input_minutes], dtype=dtype, device=device
    )

    encoder.model.to(dtype)
    tokens = encoder.encode_tokens(ts, dt)
    n_batch, n_tokens, d_surya = tokens.shape
    print(f"tokens: {tuple(tokens.shape)}  ->  d_surya = {d_surya}")

    if not torch.isfinite(tokens.float()).all():
        raise FloatingPointError("Surya returned non-finite tokens")
    if d_surya != encoder.embed_dim:
        raise RuntimeError(
            f"measured D={d_surya} disagrees with config embed_dim={encoder.embed_dim}"
        )
    if n_tokens != encoder.n_tokens:
        raise RuntimeError(
            f"measured L={n_tokens} disagrees with grid^2={encoder.n_tokens}"
        )

    # --- pooling sanity ---------------------------------------------------
    box = torch.tensor([[1800, 1800, 2300, 2300]], device=device)
    mask = encoder.box_to_token_mask(box, margin_px=64)
    pooled = encoder(ts, dt, mask)
    print(
        f"pooled: {tuple(pooled.shape)} from {int(mask.sum())}/{encoder.n_tokens} "
        f"AR tokens ({args.pool})"
    )
    if pooled.shape != (n_batch, d_surya):
        raise RuntimeError(f"pooled shape {tuple(pooled.shape)} != {(n_batch, d_surya)}")

    # --- record and resolve ----------------------------------------------
    record = {
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint_name": WEIGHT_FILE,
        "checkpoint_sha256": sha256_file(weights_dir / WEIGHT_FILE),
        "config_sha256": sha256_file(weights_dir / CONFIG_FILE),
        "d_surya": int(d_surya),
        "n_tokens": int(n_tokens),
        "token_grid": int(encoder.token_grid),
        "patch_size": int(encoder.patch_size),
        "img_size": int(encoder.img_size),
        "n_params": int(n_params),
        "channel_order": list(encoder.channels),
        "n_input_timestamps": int(n_time),
        "time_delta_input_minutes": list(encoder.time_delta_input_minutes),
        "input_tensor_order": "B,C,T,H,W",
        "encoder_output": "backbone tokens (finetune=True, unembed skipped)",
        "pool": args.pool,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
    }

    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surya_integration.json").write_text(json.dumps(record, indent=2))

    resolved = resolve_surya_dim(Config(), int(d_surya))
    save_resolved(resolved)

    print(f"\nwrote {out_dir / 'surya_integration.json'}")
    print(f"wrote {RESOLVED_CONFIG_PATH}  (d_surya={resolved.d_surya})")
    print(f"checkpoint sha256 {record['checkpoint_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
