from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..contracts import MINIMUM_PEAK_FLUX, LabelRecord

GOES_CLASS_WATTS = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def goes_class_to_watts(goes_class: str) -> float:

    if not goes_class:
        return 0.0
    letter = goes_class[0].upper()

    if letter not in GOES_CLASS_WATTS:
        return 0.0
    try:
        multiplier = float(goes_class[1:]) if len(goes_class) > 1 else 1.0
    except ValueError:
        multiplier = 1.0

    return GOES_CLASS_WATTS[letter] * multiplier


def label_window(
    ar_id: str,
    t_start: datetime,
    flares: list[dict],
    horizon_hours: int = 24,
    minimum_peak_flux: float = MINIMUM_PEAK_FLUX,
    source_catalog: str = "GOES/HEK",
) -> LabelRecord:
    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo = timezone.utc)

    t_end = t_start + timedelta(hours = horizon_hours)

    matched: list[str] = []
    unattributed = 0
    for flare in flares:
        peak = flare.get("peak_time")
        if peak is None:
            continue
        if peak.tzinfo is None:
            peak = peak.replace(tzinfo = timezone.utc)

        if not (t_start <= peak < t_end):
            continue

        flare_ar = str(flare.get("noaa_ar") or "").strip()
        if not flare_ar or flare_ar in ("0", "None"):
            unattributed += 1
            continue
        if flare_ar != str(ar_id):
            continue
        if goes_class_to_watts(flare.get("goes_class", "")) < minimum_peak_flux:
            continue
        matched.append(str(flare.get("event_id", f"{flare_ar}@{peak.isoformat()}")))

    if unattributed and not matched:
        quality = "unattributed_flares_in_window"
    elif unattributed:
        quality = "ok_with_unattributed"
    else:
        quality = "ok"

    return LabelRecord(
        ar_id=str(ar_id),
        t_start_unix=int(t_start.timestamp()),
        label=int(bool(matched)),
        event_ids=matched,
        source_catalog=source_catalog,
        association_method="noaa_ar_exact_match",
        quality_flag=quality,
    )
