"""Attention modules shared across model variants.

Naming guide
------------
MultiHeadSelfAttention
    Standard MHSA with separate Q, K, V projection matrices.
SharedQKVAttention
    MHSA where a single projection produces the shared Q = K = V.
HyperSETAttention
    Energy-based symmetric attention at the core of the HyperSET model:
    row-softmax + column-softmax score, per-head RMSNorm on Q/K/V, and
    output projected back through ``W^T`` (tied input/output weights).
HyperSETLoRAAttention
    HyperSETAttention augmented with a per-iteration LoRA weight delta ``ΔW``.
HyperSETAlterAttention
    HyperSETAttention with a configurable score-activation function (relu,
    sigma, sigma_square, or any mode from :mod:`layers.activations`).
CRATEAttention
    CRATE self-attention: shared QKV projection, full learnable output
    projection.
CRATETAttention
    CRATE-T variant: output is re-projected through ``W`` (the *transpose* of
    the input weight), removing the separate output matrix.
ETAttention
    Hopfield-style energy-based attention that returns a *scalar* energy value
    (not token updates), intended for use with ``torch.func.grad_and_value``.
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.activations import d_phi, phi


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with separate Q, K, V projections.

    Args:
        feats:   Input / output feature dimension.
        head:    Number of attention heads.
        dropout: Dropout probability applied after the output projection.
    """

    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        assert feats % head == 0
        self.head = head
        self.feats = feats
        self.sqrt_d = (feats // head) ** 0.5

        self.q = nn.Linear(feats, feats, bias=False)
        self.k = nn.Linear(feats, feats, bias=False)
        self.v = nn.Linear(feats, feats, bias=False)
        self.o = nn.Linear(feats, feats, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()
        dh = self.feats // self.head
        q = self.q(x).view(b, n, self.head, dh).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, dh).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, dh).transpose(1, 2)
        # (b, h, n, n)
        score = F.softmax(torch.einsum("bhif,bhjf->bhij", q, k) / self.sqrt_d, dim=-1)
        attn = torch.einsum("bhij,bhjf->bihf", score, v)  # (b, n, h, dh)
        return self.dropout(self.o(attn.flatten(2)))


class SharedQKVAttention(nn.Module):
    """Multi-head self-attention with a single shared Q = K = V projection.

    The single weight matrix ``W`` is applied once per token; the same
    projected representation serves as query, key *and* value.

    Args:
        feats:   Input / output feature dimension.
        head:    Number of attention heads.
        dropout: Dropout probability applied after the output projection.
    """

    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        assert feats % head == 0
        self.head = head
        self.feats = feats
        self.sqrt_d = (feats // head) ** 0.5

        self.qkv = nn.Linear(feats, feats, bias=False)
        self.o = nn.Linear(feats, feats, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()
        dh = self.feats // self.head
        qkv = self.qkv(x).view(b, n, self.head, dh).transpose(1, 2)
        score = F.softmax(torch.einsum("bhif,bhjf->bhij", qkv, qkv) / self.sqrt_d, dim=-1)
        attn = torch.einsum("bhij,bhjf->bihf", score, qkv)
        return self.dropout(self.o(attn.flatten(2)))


class HyperSETAttention(nn.Module):
    """Energy-based symmetric attention at the core of HyperSET.

    Key design choices:
    - A single linear projection ``W`` produces Q, K, and V (tied weights).
    - Per-head ``RMSNorm`` is applied to Q, K, and V independently.
    - Attention scores use ``softmax(A, dim=-1) + softmax(A, dim=-2)``
      (row-softmax + column-softmax), which is the gradient of a log-sum-exp
      energy, making the token update a gradient descent step.
    - The output is re-projected through ``W^T`` to maintain energy consistency.

    Args:
        feats:   Input / output feature dimension.
        head:    Number of attention heads.
        dropout: Unused; kept for API compatibility with other attention classes.
    """

    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        assert feats % head == 0
        self.head = head
        self.feats = feats
        self.dim_head = feats // head
        self.sqrt_d = self.dim_head**0.5

        self.qkv = nn.Linear(feats, feats, bias=False)
        # Separate RMSNorm for queries (ln1_1), keys (ln1_2), values (ln1_3)
        self.ln1_1 = nn.RMSNorm(self.dim_head)
        self.ln1_2 = nn.RMSNorm(self.dim_head)
        self.ln1_3 = nn.RMSNorm(self.dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        w = self.qkv(x).view(B, T, self.head, self.dim_head).transpose(1, 2)
        # Symmetric attention: row-softmax + col-softmax
        att = torch.einsum("bhif,bhjf->bhij", self.ln1_1(w), self.ln1_2(w)) / self.sqrt_d
        score = F.softmax(att, dim=-1) + F.softmax(att, dim=-2)
        y = torch.einsum("bhij,bhjf->bihf", score, self.ln1_3(w))
        # Tied output projection: W^T
        return F.linear(y.flatten(2), self.qkv.weight.t())


class HyperSETLoRAAttention(nn.Module):
    """HyperSETAttention augmented with a per-iteration LoRA weight delta.

    At each recurrence step an external low-rank matrix ``ΔW`` is added to
    the QKV projection, allowing the effective weight to vary across iterations
    without requiring a separate weight per step.

    Args:
        feats:   Input / output feature dimension.
        head:    Number of attention heads.
        dropout: Unused; kept for API compatibility.
    """

    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        assert feats % head == 0
        self.head = head
        self.feats = feats
        self.dim_head = feats // head
        self.sqrt_d = self.dim_head**0.5

        self.qkv = nn.Linear(feats, feats, bias=False)
        # Separate RMSNorm for queries (ln1_1), keys (ln1_2), values (ln1_3)
        self.ln1_1 = nn.RMSNorm(self.dim_head)
        self.ln1_2 = nn.RMSNorm(self.dim_head)
        self.ln1_3 = nn.RMSNorm(self.dim_head)

    def forward(self, x: torch.Tensor, delta_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       Token tensor of shape ``(B, T, feats)``.
            delta_w: LoRA weight delta of shape ``(feats, feats)`` added to
                     the QKV projection for this iteration.
        """
        B, T, C = x.size()
        w = F.linear(x, self.qkv.weight + delta_w).view(B, T, self.head, self.dim_head).transpose(1, 2)
        att = torch.einsum("bhif,bhjf->bhij", self.ln1_1(w), self.ln1_2(w)) / self.sqrt_d
        score = F.softmax(att, dim=-1) + F.softmax(att, dim=-2)
        y = torch.einsum("bhij,bhjf->bihf", score, self.ln1_3(w))
        return F.linear(y.flatten(2), (self.qkv.weight + delta_w).t())


class HyperSETAlterAttention(nn.Module):
    """HyperSETAttention with a configurable score-activation function.

    The default ``"softmax"`` activation recovers :class:`HyperSETAttention`.
    Alternative activations (``"relu"``, ``"sigma"``, etc.) allow studying
    different energy landscapes.

    Args:
        feats:     Input / output feature dimension.
        head:      Number of attention heads.
        dropout:   Unused; kept for API compatibility.
        attention: Activation applied to the raw score matrix.  Accepts
                   ``"bisoftmax"`` (default; row-softmax + col-softmax),
                   ``"softmax"``, ``"relu"``, ``"sigma"``,
                   ``"sigma_square"``, or any mode from
                   :func:`layers.activations.phi`.
    """

    def __init__(
        self,
        feats: int,
        head: int = 8,
        dropout: float = 0.0,
        attention: str = "bisoftmax",
    ):
        super().__init__()
        assert feats % head == 0
        self.head = head
        self.feats = feats
        self.dim_head = feats // head
        self.sqrt_d = self.dim_head**0.5
        self.attention = attention

        self.qkv = nn.Linear(feats, feats, bias=False)
        # Separate RMSNorm for queries (ln1_1), keys (ln1_2), values (ln1_3)
        self.ln1_1 = nn.RMSNorm(self.dim_head)
        self.ln1_2 = nn.RMSNorm(self.dim_head)
        self.ln1_3 = nn.RMSNorm(self.dim_head)

        if attention == "relu":
            self.act_func = F.relu
        elif attention == "bisoftmax":
            # Symmetric row + column softmax
            self.act_func = lambda a: F.softmax(a, dim=-1) + F.softmax(a, dim=-2)
        elif attention == "sigma":
            self.act_func = lambda a: torch.sigmoid(a) * (1 - torch.sigmoid(a))
        elif attention == "sigma_square":
            self.act_func = lambda a: torch.sigmoid(a) ** 2 * (1 - torch.sigmoid(a))
        else:
            # General case: use phi/d_phi from layers.activations
            self.act_func = partial(phi, mode=attention)
            self.d_act_func = partial(d_phi, mode=attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        w = self.qkv(x).view(B, T, self.head, self.dim_head).transpose(1, 2)

        if self.attention in ("bisoftmax", "sigma", "sigma_square"):
            att = torch.einsum("bhif,bhjf->bhij", self.ln1_1(w), self.ln1_2(w)) / self.sqrt_d
            y = torch.einsum("bhij,bhjf->bihf", self.act_func(att), self.ln1_3(w))
            return F.linear(y.flatten(2), self.qkv.weight.t())

        elif self.attention == "relu":
            att = self.ln1_1(w) @ self.ln1_2(w).transpose(-1, -2)
            y = (self.act_func(att) @ self.ln1_3(w)).transpose(1, 2).contiguous().view(B, T, C)
            return F.linear(y, self.qkv.weight.t())

        else:
            # Phi-based linear attention
            att = (self.act_func(self.ln1_1(w)).transpose(-1, -2) @ self.act_func(self.ln1_2(w))) / self.sqrt_d
            y = self.act_func(self.ln1_3(w)) @ att
            # Chain-rule correction via the derivative of phi
            y = y * self.d_act_func(self.ln1_1(w))
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return F.linear(y, self.qkv.weight.t())


class CRATEAttention(nn.Module):
    """CRATE self-attention with shared Q = K = V and a separate output matrix.

    Reference: Yu et al., "White-Box Transformers via Sparse Rate Reduction"
    (NeurIPS 2023).

    Args:
        dim:      Input / output feature dimension.
        heads:    Number of attention heads.
        dim_head: Per-head projection dimension.
        dropout:  Dropout applied to attention weights and the output projection.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project and reshape to (B, H, N, dim_head)
        w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, w)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class CRATETAttention(nn.Module):
    """CRATE-T attention: weight-transposed output projection.

    Identical to :class:`CRATEAttention` except that the output is
    re-projected using the *transpose* of the input weight matrix (``W``),
    removing the need for a separate output matrix and enforcing symmetric
    weight coupling.

    Args:
        dim:      Input / output feature dimension.
        heads:    Number of attention heads.
        dim_head: Per-head projection dimension.
        dropout:  Dropout applied to attention weights.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, w)
        out = rearrange(out, "b h n d -> b n (h d)")
        # Re-project using W (transpose of the input weight)
        return F.linear(out, self.qkv.weight)


class ETAttention(nn.Module):
    """Hopfield-style energy-based attention that returns a scalar energy.

    Used inside :class:`layers.et.ETBlock` as the attention energy term.
    Because the forward pass produces a *scalar*, calling
    ``torch.func.grad_and_value(block)(g)`` yields both the gradient update
    and the energy value for the full block.

    Args:
        in_dim: Token feature dimension.
        qk_dim: Per-head Q / K dimension.
        nheads: Number of attention heads.
        beta:   Inverse temperature.  ``None`` defaults to ``1 / sqrt(qk_dim)``.
        bias:   If ``True``, add learnable biases to Q and K projections.
    """

    def __init__(
        self,
        in_dim: int,
        qk_dim: int = 64,
        nheads: int = 12,
        beta: Optional[float] = None,
        bias: bool = False,
    ):
        super().__init__()
        assert qk_dim > 0 and in_dim > 0
        self.h = nheads
        self.d = qk_dim
        self.beta = beta if beta is not None else 1.0 / qk_dim**0.5

        self.wq = nn.Parameter(torch.normal(0, 0.002, size=(nheads, qk_dim, in_dim)))
        self.wk = nn.Parameter(torch.normal(0, 0.002, size=(nheads, qk_dim, in_dim)))
        self.bq = nn.Parameter(torch.zeros(qk_dim)) if bias else None
        self.bk = nn.Parameter(torch.zeros(qk_dim)) if bias else None

    def forward(self, g: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Hopfield attention energy.

        Args:
            g:    Token tensor of shape ``(..., N, in_dim)``.
            mask: Optional attention mask applied multiplicatively before
                  the log-sum-exp.

        Returns:
            Scalar energy (0-d tensor).
        """
        q = torch.einsum("...kd,...hzd->...khz", g, self.wq)
        k = torch.einsum("...kd,...hzd->...khz", g, self.wk)
        if self.bq is not None:
            q = q + self.bq
            k = k + self.bk
        # (... H N N)
        A = torch.einsum("...qhz,...khz->...hqk", q, k)
        if mask is not None:
            A = A * mask
        return (-1.0 / self.beta) * torch.logsumexp(self.beta * A, dim=-1).sum()
