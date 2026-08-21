import torch
from torch import nn

class FusionMLP(nn.Module):

    def __init__(
            self, 
            d_surya: int,
            d_physics: int,
            d_hidden: int = 512,
            d_out: int = 256,
            dropout: float = 0.10,

    ):
        super().__init__()

        self.register_buffer("surya_mean", torch.zeros(d_surya))
        self.register_buffer("surya_std", torch.ones(d_surya))
        self.register_buffer("physics_mean", torch.zeros(d_physics))

        self.register_buffer("physics_std", torch.ones(d_physics))

        self.net = nn.Sequential(
            nn.Linear(d_surya + d_physics, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
    def load_stats(self, stats: dict[str, torch.Tensor]) -> None:
        self.surya_mean.copy_(stats["surya_mean"])
        self.surya_std.copy_(stats["surya_std"])

        self.physics_mean.copy_(stats["physics_mean"])
        self.physics_std.copy_(stats["physics_std"])


    def forward(
            self,
            z_surya: torch.Tensor,
            z_physics: torch.Tensor,
    ) -> torch.Tensor:
        assert z_surya.shape[:-1] == z_physics.shape[:-1]

        #now we need to standardize everything

        a = (z_surya.float() - self.surya_mean) / self.surya_std
        b = (z_physics.float() - self.physics_mean) / self.physics_std

        joint = torch.cat([a, b], dim = -1)
        fused = self.net(joint)

        assert fused.shape[-1] == self.net[-1].out_features
        assert torch.isfinite(fused).all()

        return fused