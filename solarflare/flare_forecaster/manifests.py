from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .checkpoints import git_commit, sha256_file

# Anchored to the package, never the process CWD. The evaluation lock has to
# resolve to the same artifacts/ whether a script is launched from solarflare/,
# the repo root, or a cluster job directory — otherwise "locked" can silently
# mean "looking in the wrong place".
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PACKAGE_ROOT / "artifacts"
TRAINING_COMPLETE = ARTIFACTS / "TRAINING_COMPLETE"
MANIFEST_PATH = ARTIFACTS / "manifests" / "training_manifest.json"

REQUIRED_CHECKPOINTS: tuple[str, ...] = (
    "physics_encoder.pt",
    "fusion_mlp.pt",
    "deterministic.pt",
    "flow_ema.pt",
    "rollout_flare_head.pt",
)

REQUIRED_STATS: tuple[str, ...] = (
    "latent_standardization.pt",
    "deterministic_residual_scale.pt",
    "physics_channel_stats.pt",
)


def split_hash() -> str:
    """Order-independent hash over the three immutable AR lists."""
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        path = ARTIFACTS / "splits" / f"{split}_ar_ids.json"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        ids = sorted(json.loads(path.read_text()))
        digest.update(split.encode())
        digest.update(json.dumps(ids, separators=(",", ":")).encode())
    return digest.hexdigest()


def write_manifest(
    config_hash: str,
    surya_checkpoint_hash: str,
    extra: dict | None = None,
) -> Path:
    checkpoints = {}
    for name in REQUIRED_CHECKPOINTS:
        path = ARTIFACTS / "checkpoints" / name
        if not path.exists():
            raise FileNotFoundError(f"required checkpoint missing: {path}")
        checkpoints[name] = sha256_file(path)

    stats = {}
    for name in REQUIRED_STATS:
        path = ARTIFACTS / "stats" / name
        if not path.exists():
            raise FileNotFoundError(f"required stats file missing: {path}")
        stats[name] = sha256_file(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash,
        "split_hash": split_hash(),
        "surya_checkpoint_hash": surya_checkpoint_hash,
        "checkpoints": checkpoints,
        "stats": stats,
        **(extra or {}),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    tmp.replace(MANIFEST_PATH)
    return MANIFEST_PATH


def mark_training_complete() -> Path:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("write_manifest() must succeed before marking completion")
    tmp = TRAINING_COMPLETE.with_suffix(".tmp")
    tmp.write_text(MANIFEST_PATH.read_text())
    tmp.replace(TRAINING_COMPLETE)
    return TRAINING_COMPLETE


def require_training_complete() -> dict:
    """Called at the top of 16_evaluate.py. Nothing else may call it."""
    if not TRAINING_COMPLETE.exists():
        raise RuntimeError(
            "Evaluation is locked until the complete model is built and trained."
        )

    # Read the SEALED copy, not training_manifest.json. That file stays
    # writable, so a later write_manifest() must not be able to retroactively
    # bless a model that changed after finalization.
    manifest = json.loads(TRAINING_COMPLETE.read_text())

    for kind, required in (
        ("checkpoints", REQUIRED_CHECKPOINTS),
        ("stats", REQUIRED_STATS),
    ):
        recorded = manifest.get(kind, {})
        missing = set(required) - set(recorded)
        if missing:
            raise RuntimeError(f"manifest is missing {kind} entries: {sorted(missing)}")
        for name, digest in recorded.items():
            path = ARTIFACTS / kind / name
            if not path.exists():
                raise RuntimeError(
                    f"{kind}/{name} is recorded in the manifest but missing on disk"
                )
            actual = sha256_file(path)
            if actual != digest:
                raise RuntimeError(
                    f"{kind}/{name} changed since finalization "
                    f"({actual[:12]} != {digest[:12]}) — retrain or re-finalize."
                )

    if manifest["split_hash"] != split_hash():
        raise RuntimeError("split files changed since finalization")
    return manifest
