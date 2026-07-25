import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from data.fetch import fetch_sharp_keys, fetch_goes_flares, label_records

EMAIL = "diego.gaf28@gmail.com"

#from stanford
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


CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
OUT = os.path.join(CACHE, "dataset_sharp.json")

all_records = []

for harp, noaa, t0, t1 in ARS:

    sharp = fetch_sharp_keys(harp, t0, t1, EMAIL)
    sharp = sharp[::5]
    flares = fetch_goes_flares(t0, t1)

    records = label_records(sharp, flares, noaa)
    positives = sum(record["label"] for record in records)
    print(f"HARP {harp}, AR {noaa}: {len(records)} records, {positives} positive")
    all_records.extend(records)


os.makedirs(CACHE, exist_ok = True)
with open(OUT, "w") as f:

    json.dump(all_records, f)

print(f"\nTotal: {len(all_records)} records across {len(ARS)} ARs")
print(f"Positives: {sum(record["label"] for record in all_records)}")
print(f"We saved to {OUT}")