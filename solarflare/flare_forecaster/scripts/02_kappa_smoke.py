import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from reconstruction.pinn import train
from physics.kappa import alpha_map, polarity_inversion_line

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

cube = np.load(os.path.join(CACHE, "harp3894_cube.npz"))["cube"]

frame = cube[0][:, ::4, ::4]


Br = frame[0]
Bt = frame[1]
Bp = frame[2]

magnetic_boundary = np.nan_to_num(np.stack([Bp, -Bt, Br], axis = -1))

model, magnetic_scale = train(magnetic_boundary, n_steps = 1500, device = "cpu", n_col = 1024)

alpha, magnetic_grid = alpha_map(model, device = "cpu")

print("Alpha map: ", alpha.shape)
print(f"alpha min {alpha.min():.5f} max {alpha.max():.5f} |mean| {np.abs(alpha).mean():.5f}")

print(f"Strongest twist in bottom layer (at surface): |alpha| mean {np.abs(alpha[0]).mean():.5f}")

bz_surface = magnetic_grid[0][:, :, 2]
pil = polarity_inversion_line(bz_surface)

print("PIL pixels:", int(pil.sum()), "of", pil.size)

alpha_surface = np.abs(alpha[0])
print(f"|alpha| on PIL: {alpha_surface[pil].mean():.5f}")
print(f"|alpha| not on PIL: {alpha_surface[~pil].mean():.5f}")

