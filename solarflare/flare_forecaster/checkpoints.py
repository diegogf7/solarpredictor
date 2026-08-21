from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr = subprocess.DEVNULL, text = True
        ).strip()
    except Exception:
        return "unknown what's going on"

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def config_hash(config: Any) -> str:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    blob = json.dumps(payload, sort_keys = True, separators =(",", ":"))

    return hashlib.sha256(blob.encode()).hexdigest()

#rest of the code from Claude 

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    training_loss: float,
    config: Any,
    split_hash: str,
    source_checkpoint_hashes: Mapping[str, str],
    seed: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "training_loss": float(training_loss),
        "config": asdict(config) if is_dataclass(config) else dict(config),
        "config_hash": config_hash(config),
        "split_hash": split_hash,
        "source_checkpoint_hashes": dict(source_checkpoint_hashes),
        "git_commit": git_commit(),
        "seed": seed,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_frozen(
    module: nn.Module,
    path: str | Path,
    expected_config_hash: str | None = None,
    expected_split_hash: str | None = None,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    payload = load_checkpoint(path, map_location)
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise RuntimeError(
            f"{path}: config hash {payload['config_hash'][:12]} != {expected_config_hash[:12]}"
        )
    if expected_split_hash is not None and payload["split_hash"] != expected_split_hash:
        raise RuntimeError(
            f"{path}: split hash {payload['split_hash'][:12]} != {expected_split_hash[:12]}"
        )
    module.load_state_dict(payload["model_state"])
    return freeze(module)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def assert_frozen(module: nn.Module, name: str) -> None:
    live = [n for n, p in module.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"{name} is not frozen: {live[:5]} ({len(live)} params)")


def assert_no_grads(module: nn.Module, name: str) -> None:
    leaked = [n for n, p in module.named_parameters() if p.grad is not None]
    if leaked:
        raise RuntimeError(f"gradient leaked into frozen {name}: {leaked[:5]}")


def trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    found: list[nn.Parameter] = []
    for module in modules:
        found.extend(p for p in module.parameters() if p.requires_grad)
    if not found:
        raise RuntimeError("no trainable parameters — every module is frozen")
    return found
