import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import binary_dilation
import drms
from astropy.io import fits
from data.fetch import fetch_goes_flares

EMAIL = "diego.gaf28@gmail.com"; MM = 0.3645; RANK = {"A":1,"B":2,"C":3,"M":4,"X":5}
c = drms.Client(email=EMAIL)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
catalog = json.load(open(os.path.join(ROOT, "cache", "ar_catalog.json")))
OUT = os.path.join(ROOT, "cache", "triage_dataset.json")

def pil_line(bz, frac=0.05, w=2):
    t = frac * np.abs(bz).max()
    return binary_dilation(bz > t, iterations=w) & binary_dilation(bz < -t, iterations=w)

def twist_scalar(Br, Bt, Bp):
    Bx = np.nan_to_num(Bp); By = np.nan_to_num(-Bt); Bz = np.nan_to_num(Br)
    jz = np.gradient(By, axis=1) - np.gradient(Bx, axis=0)
    bmag = np.sqrt(Bx**2 + By**2 + Bz**2); strong = bmag > 0.1 * bmag.max()
    pil = pil_line(Bz) & strong
    if pil.sum() == 0: return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.abs(jz / Bz) / MM
    v = a[pil]; v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 0.0

flares = []
for yr in range(2010, 2017):
    try: flares += fetch_goes_flares(f"{yr}-01-01", f"{yr}-12-31")
    except Exception as e: print("flare fetch", yr, e, flush=True)
flares = [f for f in flares if RANK.get((f["goes_class"] or "")[:1].upper(), 0) >= 4]
print(f"{len(flares)} M+ flares loaded", flush=True)

def flaring(noaa, tref):
    t = datetime.strptime(tref[:19], "%Y.%m.%d_%H:%M:%S")
    lo, hi = t - timedelta(days=5), t + timedelta(days=5)
    return int(any(f["noaa_ar"] == noaa and lo <= datetime.fromisoformat(f["peak_time"][:19]) <= hi for f in flares))

data = []
for i, ar in enumerate(catalog):
    harp, noaa, tref = ar["harpnum"], ar["noaa"], ar["t_ref"]
    try:
        keys, segs = c.query(f"hmi.sharp_cea_720s[{harp}][{tref}]",
                             key=["USFLUX","MEANGBT","MEANPOT","SHRGT45"], seg=["Br","Bt","Bp"])
        if len(keys) == 0: continue
        comp = []
        for s in ["Br", "Bt", "Bp"]:
            with fits.open("http://jsoc.stanford.edu" + segs[s][0]) as h:
                comp.append(h[1].data.astype(np.float32))
        sharp = [float(keys[k][0]) for k in ["USFLUX","MEANGBT","MEANPOT","SHRGT45"]]
        if not np.isfinite(sharp).all(): continue
        data.append({"noaa": noaa, "t_ref": tref, "twist": twist_scalar(*comp),
                     "sharp": sharp, "label": flaring(noaa, tref)})
    except Exception as e:
        print(f"AR {noaa} err {e}", flush=True); continue
    if (i + 1) % 50 == 0:
        json.dump(data, open(OUT, "w"))
        print(f"{i+1}/{len(catalog)}  {len(data)} done  {sum(d['label'] for d in data)} flaring", flush=True)

json.dump(data, open(OUT, "w"))
print(f"DONE {len(data)} ARs, {sum(d['label'] for d in data)} flaring -> {OUT}", flush=True)
