"""HyperSET transformer blocks for the image-classification (IC) task.

HyperSET replaces the standard additive residual update with a *learned
gradient descent* step.  Per-recurrence step sizes (``alpha`` for attention,
``beta`` for feedforward) are either pre-computed externally and passed in as
a conditioning vector ``c`` (base class), or derived *inside* the block from
a small inner time MLP (``Alter*`` variants).

Classes
-------
HyperSETLayer
    Core HyperSET block (accepts pre-computed step-size embedding ``c``).
HyperSETLayerLoRA
    HyperSET block with per-recurrence LoRA weight deltas for attention and
    feedforward.
HyperSETFixedStepSizeLayer
    Simplified HyperSET block with a single fixed scalar step size (no time
    conditioning).
HyperSETAlterFeedforwardLayer
    HyperSET block with a configurable feedforward activation.  Step sizes are
    computed *inside* the block via a small inner time MLP.
HyperSETAlterAttentionLayer
    HyperSET block with a configurable attention activation.  Step sizes are
    computed *inside* the block via a small inner time MLP.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.activations import d_phi, phi
from layers.components.attention import (
    HyperSETAlterAttention,
    HyperSETAttention,
    HyperSETLoRAAttention,
)


class HyperSETLayer(nn.Module):
    """Core HyperSET block for the image-classification (IC) task.

    Receives a pre-computed per-recurrence conditioning vector ``c`` of shape
    ``(B, N, 2 * feats)`` whose two halves are the per-token attention step
    ``alpha`` and feedforward step ``beta``::

        alpha, beta = split(c, dim=-1)
        x = x - alpha * attn(x)
        x = x + beta  * W^T · gelu(norm(W · x))

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension for the feedforward projection ``W``.
        head:       Number of attention heads.
        dropout:    Unused; kept for API compatibility.
    """

    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Linear(feats, mlp_hidden, bias=False)
        self.attn = HyperSETAttention(feats=feats, head=head)
        self.ln2 = nn.RMSNorm(mlp_hidden)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token tensor ``(B, N, feats)``.
            c: Step-size conditioning ``(B, N, 2 * feats)``; split into
               per-token attention step ``alpha`` and feedforward step ``beta``.
        """
        alpha, beta = c.chunk(2, dim=-1)
        x = x - alpha * self.attn(x)
        x = x + beta * F.linear(
            F.gelu(self.ln2(self.mlp(x))), self.mlp.weight.t()
        )  # gelu gives modest improvement over relu
        return x

    @torch.no_grad()
    def get_energy_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar attention energy for analysis / visualisation.

        Returns:
            Tensor of shape ``(B, 1)``.
        """

        B = x.shape[0]
        x_weight = self.attn.qkv(x)
        w = rearrange(x_weight, "b n (h d) -> b h n d", h=self.attn.head)
        w = F.rms_norm(w, (w.shape[-1],))
        attn = torch.matmul(w, w.transpose(-1, -2)) / self.attn.sqrt_d
        E_attn = attn.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True) * self.attn.sqrt_d

        return E_attn

    @torch.no_grad()
    def get_energy_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward energy ``-0.5 * ||relu(norm(W·x))||^2`` for analysis."""

        B = x.shape[0]
        ff = F.rms_norm(self.mlp(x), (self.mlp.out_features,))
        E_ff = -0.5 * F.relu(ff).view(B, -1).norm(dim=1, keepdim=True) ** 2

        return E_ff

    @torch.no_grad()
    def get_total_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Total energy (attention + feedforward) for analysis / visualisation."""
        return self.get_energy_attention(x) + self.get_energy_ff(x)


class HyperSETLayerLoRA(nn.Module):
    """HyperSET block augmented with per-recurrence LoRA weight deltas.

    At each recurrence step externally supplied low-rank matrices
    ``delta_w_attn`` and ``delta_w_ff`` are added to the attention and
    feedforward weights, allowing the effective weight to vary across
    iterations without storing a separate full matrix per step.

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension for the feedforward projection.
        head:       Number of attention heads.
        dropout:    Unused; kept for API compatibility.
    """

    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Linear(feats, mlp_hidden, bias=False)
        self.attn = HyperSETLoRAAttention(feats=feats, head=head)
        self.ln2 = nn.RMSNorm(mlp_hidden)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        delta_w_attn: torch.Tensor,
        delta_w_ff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:            Token tensor ``(B, N, feats)``.
            c:            Step-size conditioning ``(B, N, 2 * feats)``.
            delta_w_attn: LoRA delta for the attention weight ``(feats, feats)``.
            delta_w_ff:   LoRA delta for the feedforward weight
                          ``(mlp_hidden, feats)``.
        """
        alpha, beta = c.chunk(2, dim=-1)
        x = x - alpha * self.attn(x, delta_w_attn)
        eff_weight = self.mlp.weight + delta_w_ff
        x = x + beta * F.linear(
            F.gelu(self.ln2(F.linear(x, eff_weight))), eff_weight.t()
        )  # gelu gives modest improvement over relu
        return x


class HyperSETFixedStepSizeLayer(nn.Module):
    """Simplified HyperSET block with a single fixed scalar step size.

    Both the attention and feedforward sub-steps share the same scalar
    ``stepsize``::

        x = x - stepsize * attn(x)
        x = x + stepsize * W^T · gelu(norm(W · x))

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension for the feedforward projection.
        head:       Number of attention heads.
        dropout:    Unused; kept for API compatibility.
    """

    def __init__(self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Linear(feats, mlp_hidden, bias=False)
        self.attn = HyperSETAttention(feats=feats, head=head)
        self.ln2 = nn.RMSNorm(mlp_hidden)

    def forward(self, x: torch.Tensor, stepsize: float = 1.0) -> torch.Tensor:
        """
        Args:
            x:        Token tensor ``(B, N, feats)``.
            stepsize: Scalar step size applied to both sub-steps.
        """
        x = x - stepsize * self.attn(x)
        x = x + stepsize * F.linear(
            F.gelu(self.ln2(self.mlp(x))), self.mlp.weight.t()
        )  # gelu gives modest improvement over relu
        return x


class HyperSETAlterFeedforwardLayer(nn.Module):
    """HyperSET block with a configurable feedforward activation.

    Unlike :class:`HyperSETLayer`, the step sizes ``(alpha, beta)`` are
    computed *inside* the block by a small ``time_mlp_inner`` network applied
    to the sum of the external conditioning ``c`` and the current token
    representation.  This removes the need for a separate model-level time
    MLP that expands the conditioning to ``2 * feats``.

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension for the feedforward projection.
        head:       Number of attention heads.
        dropout:    Unused; kept for API compatibility.
        ff:         Feedforward activation name.  ``"relu"`` (default),
                    ``"softmax"``, or any mode from
                    :func:`layers.activations.phi`.
    """

    def __init__(
        self,
        feats: int,
        mlp_hidden: int,
        head: int = 8,
        dropout: float = 0.0,
        ff: str = "relu",
    ):
        super().__init__()
        self.mlp = nn.Linear(feats, mlp_hidden, bias=False)
        self.attn = HyperSETAttention(feats=feats, head=head)
        self.ln2 = nn.RMSNorm(mlp_hidden)

        self.time_mlp_inner = nn.Sequential(
            nn.RMSNorm(feats),
            nn.GELU(),
            nn.Linear(feats, 2 * feats),
        )

        nn.init.constant_(self.time_mlp_inner[2].weight, 0)
        nn.init.constant_(self.time_mlp_inner[2].bias, 0)
        self.ln_inp = nn.RMSNorm(feats)

        self.ff = ff
        if ff == "relu":
            self.act_func = F.relu
        elif ff == "softmax":
            self.act_func = partial(F.softmax, dim=-1)
        else:
            self.act_func = partial(phi, mode=ff)
            self.d_act_func = partial(d_phi, mode=ff)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token tensor ``(B, N, feats)``.
            c: External conditioning ``(B, N, feats)`` (e.g. recurrence embedding).
        """
        alpha, beta = self.time_mlp_inner(c + self.ln_inp(x)).chunk(2, dim=-1)
        x = x - alpha * self.attn(x)
        if self.ff in ("relu", "softmax"):
            x = x + beta * F.linear(self.act_func(self.ln2(self.mlp(x))), self.mlp.weight.t())
        else:
            ff_hidden = self.ln2(self.mlp(x))
            phi_ff_hidden = self.act_func(ff_hidden)
            d_phi_ff_hidden = self.d_act_func(ff_hidden)
            ff = phi_ff_hidden.sum(dim=-1, keepdim=True) * d_phi_ff_hidden
            x = x + beta * F.linear(ff, self.mlp.weight.t())
        return x

    @torch.no_grad()
    def get_energy_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar attention energy for analysis / visualisation.

        Returns:
            Tensor of shape ``(B, 1)``.
        """

        B = x.shape[0]
        x_weight = self.attn.qkv(x)
        w = rearrange(x_weight, "b n (h d) -> b h n d", h=self.attn.head)
        w = F.rms_norm(w, (w.shape[-1],))
        attn = torch.matmul(w, w.transpose(-1, -2)) / self.attn.sqrt_d
        E_attn = attn.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True) * self.attn.sqrt_d

        return E_attn

    @torch.no_grad()
    def get_energy_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Activation-dependent feedforward energy for analysis / visualisation.

        The formula varies with ``self.ff``:

        * ``"relu"``    : ``-0.5 * ||relu(norm(W·x))||^2``
        * ``"softmax"`` : ``-logsumexp(norm(W·x))``
        * other (phi)   : ``-0.5 * ||phi(norm(W·x))||^2``
        """

        B = x.shape[0]
        ff = self.mlp(x)
        ff = F.rms_norm(ff, (ff.shape[-1],))
        if self.ff == "relu":
            E_ff = -0.5 * (F.relu(ff)).view(B, -1).norm(dim=1, keepdim=True) ** 2
        elif self.ff == "softmax":
            E_ff = -ff.logsumexp(dim=-1).view(B, -1).sum(dim=1, keepdim=True)
        else:
            phi_ff = self.act_func(ff)
            E_ff = -0.5 * (phi_ff.norm(dim=-1) ** 2).view(B, -1).sum(dim=1, keepdim=True)

        return E_ff

    @torch.no_grad()
    def get_total_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Total energy (attention + feedforward) for analysis / visualisation."""
        return self.get_energy_attention(x) + self.get_energy_ff(x)


class HyperSETAlterAttentionLayer(nn.Module):
    """HyperSET block with a configurable attention activation.

    Like :class:`HyperSETAlterFeedforwardLayer`, step sizes come from an
    inner time MLP.  The attention module uses
    :class:`~layers.components.attention.HyperSETAlterAttention` so the
    score function can be swapped out.

    Args:
        feats:      Token embedding dimension.
        mlp_hidden: Hidden dimension for the feedforward projection.
        head:       Number of attention heads.
        dropout:    Unused; kept for API compatibility.
        attention:  Attention activation name passed to
                    :class:`~layers.components.attention.HyperSETAlterAttention`.
    """

    def __init__(
        self,
        feats: int,
        mlp_hidden: int,
        head: int = 8,
        dropout: float = 0.0,
        attention: str = "bisoftmax",
    ):
        super().__init__()
        self.attention = attention
        self.mlp = nn.Linear(feats, mlp_hidden, bias=False)
        self.attn = HyperSETAlterAttention(feats=feats, head=head, attention=attention)
        self.ln2 = nn.RMSNorm(mlp_hidden)

        self.time_mlp_inner = nn.Sequential(
            nn.RMSNorm(feats),
            nn.GELU(),
            nn.Linear(feats, 2 * feats),
        )
        nn.init.constant_(self.time_mlp_inner[2].weight, 0)
        nn.init.constant_(self.time_mlp_inner[2].bias, 0)
        self.ln_inp = nn.RMSNorm(feats)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token tensor ``(B, N, feats)``.
            c: External conditioning ``(B, N, feats)``.
        """
        alpha, beta = self.time_mlp_inner(c + self.ln_inp(x)).chunk(2, dim=-1)
        x = x - alpha * self.attn(x)
        x = x + beta * F.linear(
            F.gelu(self.ln2(self.mlp(x))), self.mlp.weight.t()
        )  # gelu gives modest improvement over relu
        return x

    @torch.no_grad()
    def get_energy_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-activation-dependent energy for analysis / visualisation.

        The formula varies with ``self.attention``:

        * ``"sigma"`` : ``sum(sigmoid(w·wᵀ/√d) · √d/2)``
        * ``"relu"``  : ``sum(0.25 · relu(w·wᵀ/√d)² · √d)``
        * other (phi) : ``sum(phi(w)·phi(w)ᵀ/√d)² · √d/4)``

        Returns:
            Tensor of shape ``(B, 1)``.
        """
        B = x.shape[0]
        x_weight = self.attn.qkv(x)
        w = rearrange(x_weight, "b n (h d) -> b h n d", h=self.attn.head)
        w = F.rms_norm(w, (w.shape[-1],))
        if self.attention == "sigma":
            att = torch.einsum("bhif,bhjf->bhij", w, w) / self.attn.sqrt_d
            E_attn = (torch.sigmoid(att) * self.attn.sqrt_d / 2).view(B, -1).sum(dim=-1, keepdim=True)
            return E_attn
        elif self.attention == "relu":
            att = torch.einsum("bhif,bhjf->bhij", w, w) / self.attn.sqrt_d
            E_attn = (0.25 * (F.relu(att)) ** 2).view(B, -1).sum(dim=1, keepdim=True) * self.attn.sqrt_d
            return E_attn
        else:
            phi_w = self.attn.act_func(w)
            att = torch.einsum("bhif,bhjf->bhij", phi_w, phi_w) / self.attn.sqrt_d
            E_attn = (att**2 * self.attn.sqrt_d / 4).view(B, -1).sum(dim=-1, keepdim=True)
            return E_attn

    @torch.no_grad()
    def get_energy_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward energy ``-0.5 * ||relu(norm(W·x))||^2`` for analysis."""

        B = x.shape[0]
        ff = F.rms_norm(self.mlp(x), (self.mlp.out_features,))
        E_ff = -0.5 * F.relu(ff).view(B, -1).norm(dim=1, keepdim=True) ** 2

        return E_ff

    @torch.no_grad()
    def get_total_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Total energy (attention + feedforward) for analysis / visualisation."""
        return self.get_energy_attention(x) + self.get_energy_ff(x)
