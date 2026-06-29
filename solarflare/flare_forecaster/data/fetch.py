from datetime import datetime, timedelta
import drms
from sunpy.net import Fido, attrs as a

FLARE_RANK = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}


def fetch_sharp_keys(harpnum, tstart, tend, email):
    c = drms.Client(email=email)
    t0 = tstart[:10].replace("-", ".") + "_00:00:00_TAI"
    t1 = tend[:10].replace("-", ".") + "_00:00:00_TAI"
    keys = c.query(
        f"hmi.sharp_cea_720s[{harpnum}][{t0}-{t1}@720s]",
        key=["T_REC", "NOAA_AR", "QUALITY", "USFLUX", "MEANGBT", "MEANPOT", "SHRGT45"]
    )
    records = []
    for _, row in keys.iterrows():
        if int(row["QUALITY"]) != 0:
            continue
        records.append({
            "timestamp": row["T_REC"][:19].replace(".", "-", 2).replace("_", "T"),
            "noaa_ar":   int(row["NOAA_AR"]),
            "features":  [float(row[k]) for k in ["USFLUX", "MEANGBT", "MEANPOT", "SHRGT45"]],
        })
    return records


def fetch_goes_flares(tstart, tend):
    result = Fido.search(
        a.Time(tstart, tend),
        a.hek.EventType("FL"),
        a.hek.OBS.Observatory == "GOES",
    )
    if not result or len(result[0]) == 0:
        return []
    return [
        {
            "peak_time":  row["event_peaktime"].isot[:19],
            "goes_class": row["fl_goescls"],
            "noaa_ar":    int(row["ar_noaanum"]) if row["ar_noaanum"] else None,
        }
        for row in result[0]
    ]


def label_records(sharp_records, flares, noaa_ar, horizon_hours=24, min_class="M"):
    min_rank = FLARE_RANK.get(min_class[0].upper(), 4)
    relevant = [
        f for f in flares
        if f["noaa_ar"] == noaa_ar
        and FLARE_RANK.get((f["goes_class"] or "")[:1].upper(), 0) >= min_rank
    ]
    labeled = []
    for rec in sharp_records:
        t0 = datetime.fromisoformat(rec["timestamp"])
        t1 = t0 + timedelta(hours=horizon_hours)
        labeled.append({
            "ar_id":     str(noaa_ar),
            "timestamp": rec["timestamp"],
            "label":     int(any(t0 <= datetime.fromisoformat(f["peak_time"]) <= t1 for f in relevant)),
            "features":  rec["features"],
        })
    return labeled


if __name__ == "__main__":
    import json

    EMAIL, HARPNUM, NOAA_AR = "diego.gaf28@gmail.com", 4315, 12017
    TSTART, TEND = "2014-03-28", "2014-03-31"

    print("Fetching SHARP keywords...")
    sharp = fetch_sharp_keys(HARPNUM, TSTART, TEND, EMAIL)
    print(f"  {len(sharp)} frames")
    assert len(sharp) > 0 and all(len(r["features"]) == 4 for r in sharp)

    print("Fetching GOES flares...")
    flares = fetch_goes_flares(TSTART, TEND)
    print(f"  {len(flares)} events")

    print("Labeling...")
    records = label_records(sharp, flares, NOAA_AR)
    print(f"  {len(records)} records, {sum(r['label'] for r in records)} positive")
    assert all(r["label"] in (0, 1) for r in records)
    print("All assertions passed.")
    print(json.dumps(records[0], indent=2))
