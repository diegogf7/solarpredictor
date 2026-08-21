import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from scipy.ndimage import zoom

from scipy.ndimage import zoom, binary_dilation
from data.fetch import fetch_sharp_cube

EMAIL = "diego.gaf28@gmail.com"
MM_PER_PX = 0.3645 
OUT_SIZE = 64

ARS = [
    (814, 11277, "2011-09-01", "2011-09-14"),
    (824, 11281, "2011-09-01", "2011-09-14"),
    (833, 11283, "2011-09-01", "2011-09-14"),
    (843, 11287, "2011-09-01", "2011-09-14"),
    (1447, 11428, "2012-03-01", "2012-03-14"),
    (1449, 11429, "2012-03-01", "2012-03-14"),
    (1455, 11431, "2012-03-01", "2012-03-14"),
    (2716, 11739, "2013-05-08", "2013-05-20"),
    (2718, 11740, "2013-05-08", "2013-05-20"),
    (2727, 11741, "2013-05-08", "2013-05-20"),
    (2733, 11743, "2013-05-08", "2013-05-20"),
    (3894, 12017, "2014-03-25", "2014-04-02"),
]


CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "twist_maps")
os.makedirs(CACHE, exist_ok=True)

#the import doesn't work the same so I'm just making a new function for it instead
def polarity_inversion_line(bz, fraction = 0.05, width = 2):

    threshold = fraction * np.abs(bz).max()
    positive = bz > threshold
    negative = bz < -threshold
    final = binary_dilation(positive, iterations = width) & binary_dilation(negative, iterations = width)
    
    return final


def twist_map(frame):
    
    Bx = np.nan_to_num(frame[2])
    By = np.nan_to_num(-frame[1])
    Bz = np.nan_to_num(frame[0])

    jz = np.gradient(By, axis = 1) - np.gradient(Bx, axis = 0)

    bmag = np.sqrt(Bx ** 2 + By ** 2 + Bz **2)

    strong = bmag > 0.1 * bmag.max()

    pil = polarity_inversion_line(Bz) & strong

    with np.errstate(divide = "ignore", invalid = "ignore"):
        alpha = np.abs(jz / Bz) / MM_PER_PX

    alpha[~pil] = 0.0
    alpha[~strong] = 0.0
    alpha[~np.isfinite(alpha)] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha.astype(np.float32)

for harp, noaa, t0, t1 in ARS:

    out = os.path.join(CACHE, f"harp{harp}.npz")
    if os.path.exists(out):

        print(f"Harp {harp}: cached, skipping")
        continue

    for attempt in range(4):
        try:
            ts, frames = fetch_sharp_cube(harp, t0, t1, EMAIL, cadence = "6h")
            break
        except Exception as e:

            print(f"HARP {harp}: attempt {attempt+1} failed: {e}")
            time.sleep(15)

        else:
            print(f"Harp {harp}: FAILED, skipping")
            continue
    maps = np.array([twist_map(fr) for fr in frames], dtype = object)

    np.savez(out, maps = maps, timestamps = np.array(ts), noaa = noaa)
    print(f"Harp {harp} / AR {noaa}: {maps.shape[0]} maps saved")


print("Done")