"""CRATE and CRATE-T transformer blocks.

Reference:
Yu et al., "White-Box Transformers via Sparse Rate Reduction" (NeurIPS 2023).
Hu et al., "An In-depth Investigation of Sparse Rate rReduction in Transformer-like Models" (NeurIPS 2024).

Classes
-------
CRATELayer
    Standard CRATE block: pre-norm attention followed by an ISTA feedforward.
CRATETLayer
    CRATE-T variant: uses transposed-weight attention instead of the default
    output-projected attention.
"""

import torch
import torch.nn as nn

from layers.components.attention import CRATEAttention, CRATETAttention
from layers.components.feedforward import CRATEFeedForward
from layers.components.norm import PreNorm


class CRATELayer(nn.Module):
    """CRATE transformer block.

    Implements one step of the Coding Rate Reduction Transformer::

        x = x + PreNorm(CRATEAttention)(x)   # gradient step on rate-reduction
        x = PreNorm(CRATEFeedForward)(x)      # gradient step on LASSO

    Note that the feedforward does *not* add a residual connection – the ISTA
    update already incorporates the residual through the ReLU threshold.

    Args:
        hidden:     Token embedding dimension.
        mlp_hidden: Forwarded to :class:`~layers.components.feedforward.CRATEFeedForward`
                    (unused by the ISTA feedforward, kept for a consistent API).
        dropout:    Dropout probability forwarded to the attention module.
        head:       Number of attention heads.
        ista:       ISTA gradient step size.  Defaults to ``0.1``.
    """

    def __init__(
        self,
        hidden: int,
        mlp_hidden: int,
        dropout: float,
        head: int,
        ista: float = 0.1,
    ):
        super().__init__()
        assert hidden % head == 0
        self.attn = PreNorm(
            hidden,
            CRATEAttention(hidden, heads=head, dim_head=hidden // head, dropout=dropout),
        )
        self.ff = PreNorm(
            hidden,
            CRATEFeedForward(hidden, mlp_hidden, dropout=dropout, step_size=ista),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rate-reduction gradient step (with residual) then sparse-coding step
        return self.ff(self.attn(x) + x)


class CRATETLayer(nn.Module):
    """CRATE-T transformer block.

    Same structure as :class:`CRATELayer` but uses
    :class:`~layers.components.attention.CRATETAttention`, whose output is
    re-projected through ``W`` (the transpose of the input weight matrix)
    instead of a separate learnable output matrix.

    Args:
        hidden:     Token embedding dimension.
        mlp_hidden: Forwarded to :class:`~layers.components.feedforward.CRATEFeedForward`.
        dropout:    Dropout probability.
        head:       Number of attention heads.
        ista:       ISTA gradient step size.  Defaults to ``0.1``.
    """

    def __init__(
        self,
        hidden: int,
        mlp_hidden: int,
        dropout: float,
        head: int,
        ista: float = 0.1,
    ):
        super().__init__()
        assert hidden % head == 0
        self.attn = PreNorm(
            hidden,
            CRATETAttention(hidden, heads=head, dim_head=hidden // head, dropout=dropout),
        )
        self.ff = PreNorm(
            hidden,
            CRATEFeedForward(hidden, mlp_hidden, dropout=dropout, step_size=ista),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.attn(x) + x)
