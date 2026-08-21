"""AR-grouped, chronological splits with a temporal embargo."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import SPLITS
from ..manifests import ARTIFACTS


def make_ar_splits(
    ar_times: dict[str, int],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    embargo_hours: int = 48,
) -> dict[str, list[str]]:
    """ar_times: {ar_id: reference unix time} -> {split: [ar_id, ...]}.

    Chronological: earliest ARs train, latest test. ARs whose reference time
    falls inside the embargo band around a boundary are dropped entirely --
    dropping is safer than assigning, since a window straddling the cut would
    leak either way.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("fractions must be in (0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must leave room for test")

    ordered = sorted(ar_times.items(), key=lambda kv: kv[1])
    n = len(ordered)
    if n < 3:
        raise ValueError(f"need at least 3 ARs to split, got {n}")

    # Assign by POSITION in chronological order, not by comparing against a
    # threshold time. ARs cluster into observing epochs and frequently share
    # a reference time; a threshold comparison silently empties a split when
    # the two boundary ARs happen to tie.
    i_train = max(int(n * train_frac), 1)
    i_val = max(int(n * (train_frac + val_frac)), i_train + 1)
    i_val = min(i_val, n - 1)

    t_train_end = ordered[i_train][1]
    t_val_end = ordered[i_val][1]
    embargo = embargo_hours * 3600

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    dropped: list[str] = []
    for index, (ar_id, t) in enumerate(ordered):
        if embargo and (
            abs(t - t_train_end) < embargo or abs(t - t_val_end) < embargo
        ):
            dropped.append(ar_id)
        elif index < i_train:
            splits["train"].append(ar_id)
        elif index < i_val:
            splits["val"].append(ar_id)
        else:
            splits["test"].append(ar_id)

    splits["_dropped_to_embargo"] = dropped
    return splits


def write_splits(splits: dict[str, list[str]], artifacts: Path = ARTIFACTS) -> None:
    out = artifacts / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = out / f"{split}_ar_ids.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sorted(splits[split]), indent=2))
        tmp.replace(path)
    if "_dropped_to_embargo" in splits:
        (out / "dropped_to_embargo.json").write_text(
            json.dumps(sorted(splits["_dropped_to_embargo"]), indent=2)
        )


def load_split(split: str, artifacts: Path = ARTIFACTS) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    path = artifacts / "splits" / f"{split}_ar_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run scripts/00_prepare_splits.py")
    return json.loads(path.read_text())


def assert_no_overlap(artifacts: Path = ARTIFACTS) -> None:
    """Hard guard. Any AR in two splits is a leak."""
    sets = {s: set(load_split(s, artifacts)) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            shared = sets[a] & sets[b]
            if shared:
                raise AssertionError(
                    f"AR overlap between {a} and {b}: {sorted(shared)[:10]}"
                )
    if not all(sets.values()):
        empty = [s for s, v in sets.items() if not v]
        raise AssertionError(f"empty split(s): {empty}")
