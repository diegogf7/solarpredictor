import torch
import numpy as np
from reconstruction.field import curl

from scipy.ndimage import binary_dilation

def alpha_map(model, nx = 48, ny = 48, nz = 24, device = "cpu"):

    zs, ys, xs = torch.meshgrid(
        torch.linspace(0, 1, nz),
        torch.linspace(0, 1, ny),
        torch.linspace(0, 1, nx),
        indexing = "ij"

    )

    points = torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], dim = 1).to(device) #need to change the input shape for our model


    points.requires_grad_(True)

    magnetic_field = model(points)
    magnetic_curl = curl(magnetic_field, points)

    numerator = (magnetic_curl * magnetic_field).sum(dim = 1)
    denominator = (magnetic_field * magnetic_field).sum(dim = 1).clamp(min = 1e-8)

    #need to determine the alpha value for our curl
    alpha = (numerator / denominator).reshape(nz, ny, nx)
    alpha = alpha.detach().cpu().numpy()
    #now for the magnetic grid
    magnetic_grid = magnetic_field.reshape(nz, ny, nx, 3)
    magnetic_grid = magnetic_grid.detach().cpu().numpy()

    return alpha, magnetic_grid


def polarity_inversion_line(bz, fraction = 0.05, width = 2):
    threshold = fraction * np.abs(bz).max()

    positive = bz > threshold
    negative = bz < -threshold

    negative_growing = binary_dilation(negative, iterations =width)
    positive_growing = binary_dilation(positive, iterations = width)

    return positive_growing & negative_growing

def kappa_features(alpha, magnetic_grid, fraction = 0.05, width = 2):
    bz = magnetic_grid[0][:, :, 2]

    polarity_inversion = polarity_inversion_line(bz, fraction, width)

    kappa_map = np.abs(alpha[0])

    if polarity_inversion.sum() == 0:
        return kappa_map, 0.0, polarity_inversion
    
    weights = np.abs(bz)[polarity_inversion]
    kappa_star = float(np.average(kappa_map[polarity_inversion], weights = weights))

    return kappa_map, kappa_star, polarity_inversion




