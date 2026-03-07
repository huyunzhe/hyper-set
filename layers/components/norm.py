"""Normalization layers used by energy-based and CRATE models.

Classes
-------
EnergyLayerNorm
    Learnable layer-norm with a *scalar* shared scale, designed for use in
    Energy Transformer (ET) where the energy gradient must remain well-defined
    without per-dimension scale degrees of freedom.
PreNorm
    Thin wrapper that applies ``nn.LayerNorm`` before any sub-module, used as
    the standard pre-norm pattern inside CRATE blocks.
"""

from typing import Union

import torch
import torch.nn as nn


class EnergyLayerNorm(nn.Module):
    """Layer normalization with a single shared scale and optional per-dim bias.

    Unlike ``nn.LayerNorm``, the scale parameter ``gamma`` is a *scalar*
    shared across all feature dimensions.  This keeps the norm differentiable
    w.r.t. the input without introducing per-dimension scale parameters that
    complicate energy-based gradient computation.

    Args:
        in_dim: Feature dimension to normalize.
        bias:   If ``True``, learn a per-dimension additive bias vector.
                Defaults to ``True``.
        eps:    Small constant for numerical stability.  Defaults to ``1e-5``.
    """

    def __init__(self, in_dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Shared scalar scale (not per-dimension)
        self.gamma = nn.Parameter(torch.ones(1))
        self.bias: Union[nn.Parameter, float] = nn.Parameter(torch.zeros(in_dim)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Zero-mean, unit-variance normalization along the last dimension
        xm = x - x.mean(-1, keepdim=True)
        x_norm = xm / torch.sqrt((xm**2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * x_norm + self.bias


class PreNorm(nn.Module):
    """Apply ``nn.LayerNorm`` before a sub-module.

    Commonly used in CRATE blocks to perform pre-normalised attention or
    feed-forward operations.

    Args:
        dim: Feature dimension for the internal ``nn.LayerNorm``.
        fn:  Sub-module to wrap; receives the normalised input.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):
        return self.fn(self.norm(x), **kwargs)
