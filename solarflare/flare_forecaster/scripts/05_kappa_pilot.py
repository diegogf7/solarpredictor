import sys
import os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

from data.fetch import fetch_sharp_cube
from reconstruction.pinn import train
from physics.kappa import alpha_map, kappa_features

EMAIL = "diego.gaf28@gmail.com"

def kappa_series(harp, t0, t1, cadence = "12h"):

    timestamps, frames = fetch_sharp_cube(harp, t0, t1, EMAIL, cadence = cadence)

    values = []

    for frame in frames:

        f = frame[:, ::4, ::4]
        Br, Bt, Bp = f[0], f[1], f[2]
        boundary = np.nan_to_num(np.stack([Bp, -Bt, Br], axis = -1))
        model, scale = train(boundary, n_steps = 1500, device = "cpu")
        alpha, grid = alpha_map(model, device = "cpu")

        _, kappa_star, _, _ = kappa_features(alpha, grid)
        values.append(kappa_star)

    return timestamps, values

print("Flaring AR")
ts, kf = kappa_series(3894, "2014-03-27", "2014-03-31")

for t, k in zip(ts, kf):
    print(f"{t} kappa = {k:.5f}")

print("\n Quiet AR")

ts_q, kq = kappa_series(843, "2011-09-03", "2011-09-07")

for t, k in zip(ts_q, kq):

    print(f"{t} kappa = {k:.5f}")


print(f"Flaring mean: {np.mean(kf):.5f} vs QUIET mean {np.mean(kq):.5f}")

