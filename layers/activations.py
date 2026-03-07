"""Configurable pointwise activation functions and their derivatives.

These helpers are shared across CRATE, ET, and HyperSET model variants to
support pluggable nonlinearities.  Both ``phi`` and ``d_phi`` accept the same
``mode`` string so callers can hold a single configuration value and obtain
the matching forward/backward pair.

Supported modes
---------------
sigmoid, relu, tanh, softplus, silu/swish, gelu, elu, elup1
"""

import math

import torch
import torch.nn.functional as F


def phi(Z: torch.Tensor, mode: str = "sigmoid") -> torch.Tensor:
    """Pointwise activation function.

    Args:
        Z:    Input tensor (any shape).
        mode: Activation name.  One of ``sigmoid``, ``relu``, ``tanh``,
              ``softplus``, ``silu``/``swish``, ``gelu``, ``elu``, ``elup1``.

    Returns:
        Activated tensor with the same shape as *Z*.

    Raises:
        NotImplementedError: If *mode* is not recognised.
    """
    if mode == "sigmoid":
        return torch.sigmoid(Z)
    elif mode == "relu":
        return F.relu(Z)
    elif mode == "tanh":
        return F.tanh(Z)
    elif mode == "softplus":
        return F.softplus(Z)
    elif mode in ("silu", "swish"):
        return F.silu(Z)
    elif mode == "gelu":
        return F.gelu(Z)
    elif mode == "elu":
        return F.elu(Z)
    elif mode == "elup1":
        return F.elu(Z) + 1
    else:
        raise NotImplementedError(f"Activation '{mode}' is not supported.")


def d_phi(Z: torch.Tensor, mode: str = "sigmoid") -> torch.Tensor:
    """Element-wise derivative of :func:`phi`.

    Args:
        Z:    Pre-activation tensor (same shape as the input to ``phi``).
        mode: Activation name (must match the corresponding :func:`phi` call).

    Returns:
        Derivative tensor with the same shape as *Z*.

    Raises:
        NotImplementedError: If *mode* is not recognised.
    """
    if mode == "sigmoid":
        s = torch.sigmoid(Z)
        return s * (1.0 - s)
    elif mode == "relu":
        return torch.where(Z >= 0, torch.ones_like(Z), torch.zeros_like(Z))
    elif mode == "tanh":
        return 1.0 - F.tanh(Z) ** 2
    elif mode == "softplus":
        return torch.sigmoid(Z)
    elif mode in ("silu", "swish"):
        s = torch.sigmoid(Z)
        return s + Z * s * (1.0 - s)
    elif mode == "gelu":
        return 0.5 * (1 + torch.erf(Z / 2**0.5)) + Z * (torch.exp(-(Z**2) / 2) / (2 * math.pi) ** 0.5)
    elif mode in ("elu", "elup1"):
        return torch.where(Z > 0, torch.ones_like(Z), torch.exp(Z))
    else:
        raise NotImplementedError(f"Activation derivative for '{mode}' is not supported.")
