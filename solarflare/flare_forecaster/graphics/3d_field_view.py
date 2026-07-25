import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)
import pyvista as pv
from reconstruction.pinn import train

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

def sample_field(model, n=32, device = "cpu"):

    line = torch.linspace(0, 1, n)
    zz, yy, xx = torch.meshgrid(line, line, line, indexing = "ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1).to(device)

    with torch.no_grad():

        magnetic_field = model(points).cpu().numpy()
    
    return magnetic_field.reshape(-1, 3)

cube = np.load(os.path.join(CACHE, "harp3894_cube.npz"))["cube"]

frame = cube[0][:, ::4, ::4]
Br, Bt, Bp = frame[0], frame[1], frame[2]

boundary = np.nan_to_num(np.stack([Bp, -Bt, Br], axis = -1))

model, scale = train(boundary, n_steps = 1500, device = "cpu")
n =32
magnetic_field = sample_field(model, n = n)

