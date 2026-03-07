"""Feed-forward sub-modules shared across model variants.

Classes
-------
CRATEFeedForward
    ISTA-style sparse-coding feedforward used in CRATE.  Each forward pass
    performs one gradient descent step on a LASSO dictionary-learning objective.
HyperSETFeedForward
    Tied-weight MLP used inside HyperSET blocks.  Applies
    ``RMSNorm → activation → W^T`` with the same weight ``W`` used for the
    up-projection, ensuring the update is energy-consistent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CRATEFeedForward(nn.Module):
    """Iterative shrinkage-thresholding (ISTA) feedforward for CRATE.

    Each forward pass performs one proximal gradient step on the LASSO
    objective::

        min_z  0.5 * ||Dz - x||^2 + λ * ||z||_1

    where ``D`` is a learned dictionary matrix and ``λ`` is the sparsity
    penalty (``self.lambd``).

    Args:
        dim:       Input / output feature dimension.
        hidden_dim: Unused; kept so the constructor signature is compatible
                   with a standard two-layer MLP.
        dropout:   Unused; kept for API compatibility.
        step_size: Gradient step size ``η`` (``ista_step``).  Defaults to
                   ``0.1``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        step_size: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1  # Fixed sparsity penalty (λ in the LASSO)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # D^T D x  (reconstruction gradient, i.e. gradient of 0.5 ||Dz - x||^2)
        grad_recon = F.linear(F.linear(x, self.weight), self.weight.t())
        # D^T x    (representation gradient)
        grad_repr = F.linear(x, self.weight.t())
        # Negative gradient step with soft-thresholding for the L1 penalty
        grad_update = self.step_size * (grad_repr - grad_recon) - self.step_size * self.lambd
        return F.relu(x + grad_update)


class HyperSETFeedForward(nn.Module):
    """Tied-weight MLP used as the feedforward sub-block in HyperSET layers.

    Applies ``W x → RMSNorm → activation → W^T`` where the same weight
    matrix ``W`` is used for both the up-projection and the transposed
    down-projection.  This weight tying ensures the feedforward step is the
    gradient of a well-defined energy function.

    Args:
        in_dim:     Input / output feature dimension.
        hidden_dim: Expanded hidden dimension.
        activation: Activation function applied after the RMSNorm.
                    Defaults to ``torch.nn.functional.gelu``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        activation=F.gelu,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feedforward update ``W^T · act(norm(W · x))``."""
        return F.linear(self.act(self.norm(self.proj(x))), self.proj.weight.t())
