import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from reconstruction.pinn import train


CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
CUBE_FILE = os.path.join(CACHE, "harp3894_cube.npz")

if not os.path.exists(CUBE_FILE):
    from data.fetch import fetch_sharp_cube, stack_frames
    os.makedirs(CACHE, exist_ok=True)
    ts, frames = fetch_sharp_cube(3894, "2014-03-28", "2014-03-29", "diego.gaf28@gmail.com", cadence="6h")
    np.savez(CUBE_FILE, cube=stack_frames(frames), timestamps=np.array(ts))
    print(f"Cached cube to {CUBE_FILE}")

cube = np.load(CUBE_FILE)["cube"]
print("cube:", cube.shape)

frame = cube[0][:, ::4, ::4]
Br = frame[0]
Bt = frame[1]
Bp = frame[2]

B_boundary = np.stack([Bp, -Bt, Br], axis = -1)
B_boundary = np.nan_to_num(B_boundary)

model, B_scale = train(B_boundary, n_steps = 3000, device = "cpu", n_col = 1024)
print(f"done and B_scale = {B_scale:.2f}")