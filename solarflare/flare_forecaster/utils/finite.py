import torch

def assert_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        n_nan = int(torch.isnan(tensor).sum())

        n_inf = int(torch.isinf(tensor).sum())
        raise FloatingPointError(
            f"{name}: {n_nan} NaN, {n_inf} Inf in shape {tuple(tensor.shape)}"
        )

    return tensor

def count_nonfinite(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum())

def sanitize(tensor: torch.Tensor, clip: float) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan = 0.0, posinf = clip, neginf = -clip).clamp(
        -clip, clip
    )