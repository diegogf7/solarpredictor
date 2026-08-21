"""Cached-latent sequence windows.

Never runs Surya, the PINN, or field-line tracing. It reads latents that some
earlier caching script already wrote, and slices them into causal windows.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..contracts import SequenceBatch
from .splits import load_split

EVAL_CALLER = "16_evaluate"


class LatentSequenceDataset(Dataset):
    """Sliding causal windows of cached latents, grouped by active region.

    Every window's label is the label of its LAST frame, i.e. the forecast is
    made from history ending at t and scored on [t, t+24h).
    """

    def __init__(
        self,
        cache_path: str | Path,
        split: str,
        history_steps: int,
        artifacts: Path | None = None,
        max_gap_hours: float = 2.0,
        caller: str = "",
    ) -> None:
        if split != "train" and EVAL_CALLER not in caller:
            raise RuntimeError(
                f"split={split!r} requested by caller={caller!r}. Only "
                f"scripts/{EVAL_CALLER}.py may open validation or test data."
            )

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        allowed = set(load_split(split) if artifacts is None else load_split(split, artifacts))

        self.history_steps = history_steps
        self.windows: list[dict] = []
        self.d_surya = -1
        self.d_physics = -1

        for ar_id, series in sorted(payload.items()):
            if str(ar_id) not in allowed:
                continue
            z_surya = series["z_surya"].float()
            z_physics = series["z_physics"].float()
            times = series["t"].double()
            labels = series["label"].float()

            n_frames = z_surya.shape[0]
            self.d_surya = z_surya.shape[1]
            self.d_physics = z_physics.shape[1]
            if n_frames < history_steps:
                continue

            delta = torch.zeros(n_frames, dtype=torch.float32)
            delta[1:] = ((times[1:] - times[:-1]) / 3600.0).float()

            for end in range(history_steps, n_frames + 1):
                start = end - history_steps
                gaps = delta[start + 1 : end]
                # A window spanning a data gap is not a 24-hour history.
                if gaps.numel() and float(gaps.max()) > max_gap_hours:
                    continue
                self.windows.append(
                    {
                        "ar_id": str(ar_id),
                        "z_surya": z_surya[start:end],
                        "z_physics": z_physics[start:end],
                        "delta_hours": delta[start:end].clone(),
                        "times": times[start:end],
                        "label": labels[end - 1],
                    }
                )

        if not self.windows:
            raise RuntimeError(f"no usable windows for split={split!r}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        return self.windows[index]

    @property
    def positive_weight(self) -> torch.Tensor:
        """pos_weight for BCEWithLogits, computed on THIS split only."""
        labels = torch.stack([w["label"] for w in self.windows])
        n_pos = float(labels.sum())
        n_neg = float(len(labels) - n_pos)
        return torch.tensor(n_neg / max(n_pos, 1.0))

    @property
    def ar_ids(self) -> list[str]:
        return [w["ar_id"] for w in self.windows]


def collate(batch: list[dict]) -> SequenceBatch:
    return SequenceBatch(
        ar_id=[b["ar_id"] for b in batch],
        frame_times_unix=torch.stack([b["times"] for b in batch]),
        delta_hours=torch.stack([b["delta_hours"] for b in batch]),
        frame_valid=torch.ones(
            len(batch), batch[0]["delta_hours"].shape[0], dtype=torch.bool
        ),
        z_surya=torch.stack([b["z_surya"] for b in batch]),
        z_physics=torch.stack([b["z_physics"] for b in batch]),
        label=torch.stack([b["label"] for b in batch]).float(),
    )
