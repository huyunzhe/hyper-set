"""Standard transformer encoder building blocks.

Classes
-------
TransformerLayer
    Pre-LayerNorm transformer block with separate Q / K / V projections and a
    two-layer ReLU MLP.
TransformerLayerSharedQKV
    Same as :class:`TransformerLayer` but uses one shared QKV projection,
    halving the number of projection parameters.
"""

import torch
import torch.nn as nn

from layers.components.attention import MultiHeadSelfAttention, SharedQKVAttention


class TransformerLayer(nn.Module):
    """Pre-LayerNorm transformer encoder block.

    Applies::

        x = x + MHSA(LayerNorm(x))          # self-attention residual
        x = x + MLP(LayerNorm(x))            # feedforward residual

    where MHSA uses separate Q, K, V linear projections and the MLP is a
    two-layer network with ReLU activation and no bias.

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension of the MLP sub-block.
        head:       Number of attention heads.
        dropout:    Dropout probability applied inside the MLP.
    """

    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)  # pre-norm before attention
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)  # pre-norm before MLP
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(self.la1(x)) + x
        x = self.mlp(self.la2(x)) + x
        return x


class TransformerLayerSharedQKV(nn.Module):
    """Pre-LayerNorm transformer block with a single shared QKV projection.

    Identical to :class:`TransformerLayer` but uses
    :class:`~layers.components.attention.SharedQKVAttention` so that queries,
    keys, and values all come from the same linear map, reducing parameter count.

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension of the MLP sub-block.
        head:       Number of attention heads.
        dropout:    Dropout probability applied inside the MLP.
    """

    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)  # pre-norm before attention
        self.msa = SharedQKVAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)  # pre-norm before MLP
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(self.la1(x)) + x
        x = self.mlp(self.la2(x)) + x
        return x
