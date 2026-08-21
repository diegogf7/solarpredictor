from __future__ import annotations

from dataclasses import dataclass

import torch

PHYSICS_CHANNELS: tuple[str, ...] = (
    "alpha",
    "curvature",
    "twist",
    "int_jz",
    "b_mag",
    "closure_error",
    "trace_valid",
    "strong_field_mask",
)
N_PHYSICS_CHANNELS = len(PHYSICS_CHANNELS)
TRACE_VALID_INDEX = PHYSICS_CHANNELS.index("trace_valid")
STRONG_FIELD_INDEX = PHYSICS_CHANNELS.index("strong_field_mask")

SPLITS: tuple[str, ...] = ("train", "val", "test")
MINIMUM_PEAK_FLUX = 1e-5  # GOES M1.0


def check_shape(
    tensor: torch.Tensor,
    expected: tuple[int | None, ...],
    name: str,
) -> torch.Tensor:
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: rank {len(actual)} != {len(expected)}, got {actual}")
    for axis, (got, want) in enumerate(zip(actual, expected)):
        if want is not None and got != want:
            raise AssertionError(f"{name}: axis {axis} is {got}, expected {want} ({actual})")
    return tensor


@dataclass
class SequenceBatch:
    ar_id: list[str]
    frame_times_unix: torch.Tensor  # [B,L]
    delta_hours: torch.Tensor       # [B,L]
    frame_valid: torch.Tensor       # [B,L] bool
    z_surya: torch.Tensor           # [B,L,D_SURYA]
    z_physics: torch.Tensor         # [B,L,D_PHYS]
    label: torch.Tensor             # [B] float32

    def validate(self, d_surya: int, d_physics: int) -> "SequenceBatch":
        batch = len(self.ar_id)
        length = int(self.frame_times_unix.shape[1])
        check_shape(self.frame_times_unix, (batch, length), "frame_times_unix")
        check_shape(self.delta_hours, (batch, length), "delta_hours")
        check_shape(self.frame_valid, (batch, length), "frame_valid")
        check_shape(self.z_surya, (batch, length, d_surya), "z_surya")
        check_shape(self.z_physics, (batch, length, d_physics), "z_physics")
        check_shape(self.label, (batch,), "label")

        if self.frame_valid.dtype != torch.bool:
            raise AssertionError(f"frame_valid must be bool, got {self.frame_valid.dtype}")
        if self.label.dtype != torch.float32:
            raise AssertionError(f"label must be float32, got {self.label.dtype}")
        if not torch.isfinite(self.z_surya).all():
            raise AssertionError("z_surya contains non-finite values")
        if not torch.isfinite(self.z_physics).all():
            raise AssertionError("z_physics contains non-finite values")
        if not self.frame_valid.any(dim=1).all():
            raise AssertionError("at least one sequence has zero valid frames")
        return self

    def to(self, device: torch.device) -> "SequenceBatch":
        return SequenceBatch(
            ar_id=self.ar_id,
            frame_times_unix=self.frame_times_unix.to(device),
            delta_hours=self.delta_hours.to(device),
            frame_valid=self.frame_valid.to(device),
            z_surya=self.z_surya.to(device).float(),
            z_physics=self.z_physics.to(device).float(),
            label=self.label.to(device),
        )


@dataclass
class LabelRecord:
    ar_id: str
    t_start_unix: int
    label: int
    event_ids: list[str]
    source_catalog: str
    association_method: str
    quality_flag: str
