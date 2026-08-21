import torch
from torch import nn

class FourierTimeEmbedding(nn.Module):
    def __init__(self, width: int, max_period: int = 10_000):
        super().__init__()
        half = width // 2

        frequency = torch.exp(
            -torch.log(torch.tensor(float(max_period))) * torch.arange(half).float() / max(half - 1, 1)
        )
        self.register_buffer("frequency", frequency)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angle = t[:, None] * self.frequency[None, :] * 2 * torch.pi 
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim = -1)

def modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaLNZeroBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine = False)
        self.attn = nn.MultiheadAttention(
            width,
            heads,
            dropout = dropout,
            batch_first = True
        )

        self.norm2 = nn.LayerNorm(width, elementwise_affine = False)
        self.mlp = nn.Sequential(
            nn.Linear(width, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, width),
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 6 * width),
        )

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
            self,
            x: torch.Tensor,
            condition: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(condition).chunk(6, dim = -1)

        qkv = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(
            qkv,
            qkv,
            qkv,
            key_padding_mask = key_padding_mask,
            need_weights = False,
        )

        x = x + gate_attn[:, None, :] * attn_out

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x+ gate_mlp[:, None, :] * self.mlp(mlp_in)

        return x

class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        d_latent: int = 256,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
        ff: int = 1024,
        dropout: float = 0.10,
        max_history: int = 24
    ):
        super().__init__()
        self.max_history = max_history
        self.history_proj = nn.Linear(d_latent + 1, width)
        self.flow_state_proj = nn.Linear(d_latent, width)
        self.position = nn.Parameter(torch.zeros(1, max_history + 1, width))

        self.time_fourier = FourierTimeEmbedding(width)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(width, heads, ff, dropout)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.velocity_head = nn.Linear(width, d_latent)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        x_s: torch.Tensor,
        flow_time: torch.Tensor,
        history: torch.Tensor,
        delta_hours: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = history.shape
        assert length <= self.max_history

        history_input = torch.cat(
            [history, delta_hours.unsqueeze(-1)],
            dim=-1,
        )
        history_token = self.history_proj(history_input)
        query_token = self.flow_state_proj(x_s).unsqueeze(1)
        tokens = torch.cat([history_token, query_token], dim=1)
        tokens = tokens + self.position[:, : length + 1]

        query_valid = torch.ones(
            batch,
            1,
            dtype=torch.bool,
            device=history.device,
        )
        token_valid = torch.cat([history_valid, query_valid], dim=1)
        key_padding_mask = ~token_valid

        condition = self.time_mlp(self.time_fourier(flow_time))
        for block in self.blocks:
            tokens = block(tokens, condition, key_padding_mask)

        query = self.final_norm(tokens[:, -1])
        velocity = self.velocity_head(query)
        assert velocity.shape == x_s.shape
        assert torch.isfinite(velocity).all()
        return velocity
