import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from data.fetch import fetch_sharp_keys, fetch_goes_flares

EMAIL = "diego.gaf28@gmail.com"
RANK = {"A":1, "B":2, "C":3, "M":4, "X":5}

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "twist_maps")
WIN = {814:("2011-09-01","2011-09-14"),824:("2011-09-01","2011-09-14"),833:("2011-09-01","2011-09-14"),
       843:("2011-09-01","2011-09-14"),1447:("2012-03-01","2012-03-14"),1449:("2012-03-01","2012-03-14"),
       1455:("2012-03-01","2012-03-14"),2716:("2013-05-08","2013-05-20"),2718:("2013-05-08","2013-05-20"),
       2727:("2013-05-08","2013-05-20"),2733:("2013-05-08","2013-05-20"),3894:("2014-03-25","2014-04-02")}


twist, sharp, label = [], [], []
for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
    d = np.load(p, allow_pickle = True); maps = d["maps"]; noaa = int(d["noaa"])

    harp = int(p.split("harp")[1].split(".")[0]); t0, t1 = WIN[harp]

    twist.append(float(np.median([np.median(m[m>0]) if (m>0).any() else 0 for m in maps])))

    feats = [r["features"] for r in fetch_sharp_keys(harp, t0, t1, EMAIL) if np.isfinite(r["features"]).all()]

    sharp.append(np.mean(feats, axis = 0))
    flares = fetch_goes_flares(t0, t1)
    label.append(int(any(f["noaa_ar"] == noaa and RANK.get((f["goes_class"] or "")[:1].upper(), 0) >= 4 for f in flares)))

twist, sharp, label = np.array(twist), np.array(sharp), np.array(label)
sharp = (sharp - sharp.mean(0)) / (sharp.std(0) + 1e-9)

print(f"{len(label)} ARs, {label.sum()} flare-productive\n")

def loo_auc(X):
    if X.ndim == 1: return roc_auc_score(label, X)   # single feature = direct ranking
    pred = np.zeros(len(label))
    for i in range(len(label)):
        tr = [j for j in range(len(label)) if j != i]
        m = LogisticRegression(max_iter=1000).fit(X[tr], label[tr])
        pred[i] = m.predict_proba(X[i:i+1])[0,1]
    return roc_auc_score(label, pred)

print(f"twist alone        AUC = {loo_auc(twist):.3f}")
for k, nm in enumerate(["USFLUX","MEANGBT","MEANPOT","SHRGT45"]):
    print(f"  {nm:8s}         AUC = {loo_auc(sharp[:,k]):.3f}")
print(f"SHARP (4 feats)    AUC = {loo_auc(sharp):.3f}")
print(f"SHARP + twist      AUC = {loo_auc(np.column_stack([sharp, (twist-twist.mean())/(twist.std()+1e-9)])):.3f}")