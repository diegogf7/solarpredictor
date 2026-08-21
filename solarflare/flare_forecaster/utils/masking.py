import torch

def masked_mean(
        x: torch.Tensor,
        valid: torch.Tensor,
        dim: int = 1,
) -> torch.Tensor:
    weight = valid.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1.0)

def masked_max(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:

    filled = x.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    pooled = filled.amax(dim = dim)
    return torch.where(valid.any(dim = dim, keepdim = True), pooled, torch.zeros_like(pooled))


def safe_key_padding_mask( #had to have claude add some sort of padding 
    token_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    token_valid: [B,N] bool.
    Returns (key_padding_mask [B,N], empty_rows [B]).

    A row with no attendable token makes softmax return NaN. We force the
    final token (always the flow-state query) attendable and report which
    rows were degenerate so the caller can zero them downstream.
    """
    assert token_valid.dtype == torch.bool
    empty_rows = ~token_valid.any(dim=1)
    fixed = token_valid.clone()
    fixed[empty_rows, -1] = True
    return ~fixed, empty_rows 