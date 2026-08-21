import torch

import torch.nn as nn
import numpy as np
from .field import curl, divergence

class Primary(nn.Module):

    def __init__(self, hidden = 256, n_layers = 5, omega = 30.0):

        super().__init__()

        dimension = [3] + [hidden] * n_layers + [3]

        self.layers = nn.ModuleList(
            [nn.Linear(dimension[i], dimension[i+1]) for i in range(len(dimension) - 1)]
        )

        self.omega = omega

        self._init_weights()

    def _init_weights(self):

        n = self.layers[0].in_features

        nn.init.uniform_(self.layers[0].weight, -1/n, 1/n)
        for layer in self.layers[1:]:

            bound = (6 / layer.in_features) ** 0.5/ self.omega

            nn.init.uniform_(layer.weight, -bound, bound)
        
    def forward(self, x):

        h = torch.sin(self.omega * self.layers[0](x))

        for layer in self.layers[1: -1]:

            h = torch.sin(self.omega * layer(h))
        
        return self.layers[-1](h)
    

def physics_loss(model, x_col):

    x_col.requires_grad_(True)

    B = model(x_col)

    curlB = curl(B, x_col)
    divB = divergence(B, x_col)

    #doing the forces 

    cross = torch.linalg.cross(curlB, B)
    norm_B = torch.norm(B, dim = 1, keepdim = True).clamp(min = 1e-8)
    loss_ff = torch.mean((cross/norm_B) ** 2)

    loss_div = torch.mean(divB ** 2)

    return loss_ff, loss_div
    
def boundary_loss(model, x_bc, B_bc):

    return torch.mean((model(x_bc) - B_bc) ** 2)


def train(
    B_boundary,
    n_steps = 6000,
    lr = 1e-4,
    w_ff = 1.0,
    w_div = 1.0,
    w_bc = 10.0,
    n_col = 2048,
    device = "cuda",
    model = None,
    optimizer = None,
    lr_final_fraction = 0.05,
    return_optimizer = False,
    verbose = True,
):
    """Reconstruct a force-free field from a photospheric boundary.

    Warm-starting a series: pass BOTH the model and the optimizer back in.
    Rebuilding AdamW per frame throws away the moment estimates and kicks an
    already-fitted SIREN -- v1 traced its alpha drift to exactly that, and the
    pilot showed the loss climbing between steps 2000 and 3000 because of it.

    Learning rate follows a cosine decay to `lr_final_fraction * lr` across
    this call, so a warm-started frame settles instead of bouncing.
    """
    if not np.isfinite(B_boundary).all():
        raise ValueError(
            "B_boundary contains non-finite values. SHARP cutouts carry NaN "
            "outside the patch; sanitize before calling (a single NaN makes "
            "B_scale NaN and every loss silently becomes nan)."
        )

    H, W, _ = B_boundary.shape
    B_scale = float(np.abs(B_boundary).max())
    if not np.isfinite(B_scale) or B_scale == 0.0:
        raise ValueError(f"unusable boundary: B_scale = {B_scale}")

    B_boundary = B_boundary / B_scale

    ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing = "ij")
    x_bc = torch.tensor(
        np.stack([xs.ravel(), ys.ravel(), np.zeros(H * W)], axis = 1), dtype = torch.float32
    ).to(device)

    B_bc = torch.tensor(B_boundary.reshape(-1, 3), dtype = torch.float32).to(device)

    if model is None:
        model = Primary().to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = lr

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = max(n_steps, 1), eta_min = lr * lr_final_fraction
    )

    for step in range(n_steps):
        optimizer.zero_grad()

        x_col = torch.rand(n_col, 3, device = device)
        loss_ff, loss_div = physics_loss(model, x_col)
        index = torch.randint(0, x_bc.shape[0], (n_col,), device = device)
        loss_bc = boundary_loss(model, x_bc[index], B_bc[index])

        loss = w_ff * loss_ff + w_div * loss_div + w_bc * loss_bc

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {step}: ff={loss_ff.item()} "
                f"div={loss_div.item()} bc={loss_bc.item()}"
            )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and step % 1000 == 0:
            print(f"step {step:5d} | ff = {loss_ff.item():.3e} div = {loss_div.item():.3e} bc = {loss_bc.item():.3e} lr = {scheduler.get_last_lr()[0]:.2e}")

    if return_optimizer:
        return model, B_scale, optimizer
    return model, B_scale