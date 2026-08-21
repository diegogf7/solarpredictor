import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import drms
from datetime import date, timedelta

EMAIL = "diego.gaf28@gmail.com"
TARGET = 1000
c = drms.Client(email=EMAIL)

seen = {}   # harpnum -> [noaa, t_ref, usflux]  (keep the max-flux appearance)
d, end, step = date(2010, 6, 1), date(2020, 12, 31), timedelta(days=6)
while d < end and len(seen) < TARGET:
    tai = d.strftime("%Y.%m.%d") + "_00:00:00_TAI"
    try:
        k = c.query(f"hmi.sharp_cea_720s[][{tai}]", key=["HARPNUM", "NOAA_AR", "USFLUX", "T_REC"])
    except Exception:
        d += step; continue
    if len(k):
        for _, r in k.iterrows():
            noaa = int(r["NOAA_AR"])
            if noaa <= 0:
                continue
            try:
                uf = float(r["USFLUX"])
            except (ValueError, TypeError):
                continue
            if uf != uf:
                continue
            h = int(r["HARPNUM"])
            if h not in seen or uf > seen[h][2]:
                seen[h] = [noaa, r["T_REC"], uf]
    d += step
    if len(seen) % 100 < 7:
        print(f"{d}  ...  {len(seen)} ARs")

catalog = [{"harpnum": h, "noaa": v[0], "t_ref": v[1], "usflux": v[2]} for h, v in seen.items()]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "ar_catalog.json")
json.dump(catalog, open(out, "w"))
print(f"\nsaved {len(catalog)} ARs to {out}")
