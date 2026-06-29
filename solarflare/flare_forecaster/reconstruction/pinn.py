import torch

import torch.nn as nn
import numpy as np
from .field import curl, div

class Primary(nn.Module):

    def __init__(self, hidden = 256, n_layers = 5, omega = 30.0):

        super().__init__()

        dimension = [3] + [hidden] * n_layers + [3]

        self.layers = nn.ModuleList(
            [nn.Linear(dimension[i], dimension[i+1]) for i in range(len(dimension) - 1)]
        )

        self.omega = omega