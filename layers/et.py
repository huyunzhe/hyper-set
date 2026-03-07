"""Energy Transformer (ET) building blocks.

Reference: Hoover et al., "Energy Transformer" (NeurIPS 2023).

An ET block defines a scalar energy function over the token matrix ``g``::

    E(g) = E_attn(g) + E_hn(g)

where ``E_attn`` is a log-sum-exp Hopfield attention energy and ``E_hn`` is
a dense associative memory energy.  The token update is obtained by
differentiating w.r.t. ``g`` using ``torch.func.grad_and_value``.

Classes
-------
Hopfield
    Dense associative memory energy term (MLP-based scalar energy).
ETBlock
    Combined Hopfield + attention energy module.  ``forward`` returns a
    scalar energy so it can be used directly with autograd.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.components.attention import ETAttention


class Hopfield(nn.Module):
    """Dense Hopfield associative memory energy term.

    Computes ``fn(W · g)`` where ``W`` is a learned expansion matrix and
    ``fn`` is a scalar-valued energy function applied element-wise.

    Args:
        in_dim:     Input feature dimension.
        multiplier: Expansion factor for the hidden layer.  Defaults to ``4.0``.
        fn:         Scalar energy function applied to the projection output.
                    Defaults to ``-0.5 * sum(relu(z)^2)``.
        bias:       Whether to include a bias in the linear projection.
    """

    def __init__(
        self,
        in_dim: int,
        multiplier: float = 4.0,
        fn: Callable = lambda x: -0.5 * (F.relu(x) ** 2.0).sum(),
        bias: bool = False,
    ):
        super().__init__()
        self.fn = fn
        self.proj = nn.Linear(in_dim, int(in_dim * multiplier), bias=bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the Hopfield energy.

        Args:
            g: Token tensor of shape ``(..., in_dim)``.

        Returns:
            Scalar energy tensor.
        """
        return self.fn(self.proj(g))


class ETBlock(nn.Module):
    """Energy Transformer block: combined Hopfield + attention energy.

    ``forward`` returns the *scalar* total energy so that the gradient
    update can be computed externally via::

        dEdg, E = torch.func.grad_and_value(block)(norm(x))
        x = x - alpha * dEdg

    Args:
        in_dim:     Token feature dimension.
        qk_dim:     Per-head Q / K dimension for :class:`ETAttention`.
        nheads:     Number of attention heads.
        hn_mult:    Expansion multiplier for the :class:`Hopfield` sub-module.
                    Defaults to ``4.0``.
        attn_beta:  Inverse temperature for :class:`ETAttention`.  ``None``
                    defaults to ``1 / sqrt(qk_dim)``.
        attn_bias:  If ``True``, add learnable biases to Q and K.
        hn_bias:    If ``True``, add a bias to the Hopfield projection.
        hn_fn:      Energy function passed to :class:`Hopfield`.
    """

    def __init__(
        self,
        in_dim: int,
        qk_dim: int = 64,
        nheads: int = 12,
        hn_mult: float = 4.0,
        attn_beta: Optional[float] = None,
        attn_bias: bool = False,
        hn_bias: bool = False,
        hn_fn: Callable = lambda x: -0.5 * (F.relu(x) ** 2.0).sum(),
    ):
        super().__init__()
        assert qk_dim > 0 and in_dim > 0
        self.hn = Hopfield(in_dim, hn_mult, hn_fn, hn_bias)
        self.attn = ETAttention(in_dim, qk_dim, nheads, attn_beta, attn_bias)

    def energy(
        self,
        g: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the total energy ``E_attn(g) + E_hn(g)``."""
        return self.attn(g, mask) + self.hn(g)

    def forward(
        self,
        g: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for :meth:`energy`, required by ``torch.func.grad_and_value``."""
        return self.energy(g, mask)
