"""Shared training loop. Logs optimization diagnostics only -- never metrics."""

from __future__ import annotations

from typing import Callable

import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    compute_loss: Callable,
    device: torch.device,
    grad_clip: float,
    scaler=None,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            loss, batch_size = compute_loss(model, batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss}")

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def run_stage(
    name: str,
    model,
    loader,
    compute_loss: Callable,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    log_every: int = 10,
) -> float:
    """Train one stage to completion. Returns the final training loss."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{name}: nothing to train, every parameter is frozen")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    print(f"\n[{name}] {sum(p.numel() for p in trainable)/1e3:.1f}k trainable params")

    loss = float("nan")
    for epoch in range(epochs):
        loss = train_one_epoch(
            model, loader, optimizer, compute_loss, device, grad_clip
        )
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3}  train_loss {loss:.5f}")
    return loss
