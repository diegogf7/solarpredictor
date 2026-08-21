import json
from dataclasses import dataclass, replace, asdict

from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # Time
    cadence_hours: int = 1
    history_steps: int = 24
    forecast_steps: int = 24
    max_gap_hours: float = 2.0

    # Inputs
    surya_channels: int = 13
    surya_input_steps: int = 2
    physics_channels: int = 8
    physics_height: int = 128
    physics_width: int = 128

    # Latents
    d_surya: int = -1          # measured by 02_surya_integration.py
    d_physics: int = 256
    d_fused: int = 256

    # Fusion
    fusion_hidden: int = 512
    fusion_dropout: float = 0.10

    # Deterministic model
    deterministic_hidden: int = 256
    deterministic_layers: int = 2

    # Flow transformer
    flow_width: int = 256
    flow_layers: int = 4
    flow_heads: int = 8
    flow_ff: int = 1024
    flow_dropout: float = 0.10
    base_noise_scale: float = 1.0
    flow_integration_steps: int = 16
    rollout_samples_train: int = 16
    rollout_samples_eval: int = 32

    # Training
    batch_size: int = 32
    physics_epochs: int = 50
    fusion_epochs: int = 30
    deterministic_epochs: int = 50
    flow_epochs: int = 100
    flare_head_epochs: int = 30
    lr_physics: float = 3e-4
    lr_fusion: float = 3e-4
    lr_deterministic: float = 3e-4
    lr_flow: float = 2e-4
    lr_flare_head: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    def validate(self) -> None:
        assert self.history_steps > 0
        assert self.forecast_steps * self.cadence_hours == 24
        assert self.d_physics > 0
        assert self.d_fused > 0
        assert self.flow_width % self.flow_heads == 0


RESOLVED_CONFIG_PATH = Path("artifacts/manifests/resolved_config.json")

def require_resolved(config: Config) -> Config:

    if config.d_surya <= 0:
        raise RuntimeError(
            "d_surya is not working. Run scripts/02_surya_integration.py against official checkpoint before training"
        )

    config.validate()
    return config


def resolve_surya_dim(config: Config, d_surya: int) -> Config:

    if d_surya <= 0:
        raise ValueError(f"measured d_surya must be positive, got {d_surya}")

    return replace(config, d_surya = d_surya)

def save_resolved(config: Config, path: Path = RESOLVED_CONFIG_PATH) -> Path:
    require_resolved(config)
    path.parent.mkdir(parents = True, exist_ok = True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(asdict(config), sort_keys = True, indent = 2))

    temp.replace(path)
    return path

def load_resolved(path: Path = RESOLVED_CONFIG_PATH) -> Config:

    if not path.exists():
        raise RuntimeError(
            f"{path} missing. Run scripts/02_surya_integration.py first."
        )

    return require_resolved(Config(**json.loads(path.read_text())))
