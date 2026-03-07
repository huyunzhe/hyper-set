"""Positional embedding utilities shared across tasks.

Provides
--------
sinusoidal_positional_embedding
    NumPy-based 1-D sinusoidal PE (used by IC models for patch positions).
SinusoidalPosEmb
    ``nn.Module`` that maps integer indices to sinusoidal embeddings (used as
    a timestep / iteration-index embedder in HyperSET and MIM models).
TimestepEmbedder
    Two-layer MLP that maps a scalar timestep to a dense embedding, with an
    internal sinusoidal frequency encoding (used by MIM diffusion models).
get_2d_sincos_pos_embed
    2-D sin/cos positional embeddings for a square grid (Sudoku, MIM).
get_2d_sincos_pos_embed_from_grid
    Helper: combine two 1-D grids into a 2-D embedding.
get_1d_sincos_pos_embed_from_grid
    Helper: build a 1-D sinusoidal embedding from an array of positions.

References
----------
- Vaswani et al., "Attention Is All You Need" (2017)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022)
  https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
"""

import math

import numpy as np
import torch
import torch.nn as nn


def sinusoidal_positional_embedding(seq_len: int, d_model: int) -> torch.Tensor:
    """Create 1-D sinusoidal positional embeddings (Vaswani et al., 2017).

    Intended for embedding fixed patch positions before the transformer
    encoder in the IC models.

    Args:
        seq_len: Number of positions (e.g. number of image patches + CLS token).
        d_model: Embedding dimension.

    Returns:
        Float tensor of shape ``(seq_len, d_model)``.
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return torch.FloatTensor(pe)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for scalar indices (e.g. recurrence step or
    diffusion timestep).

    Maps a 1-D integer tensor of shape ``(L,)`` to embeddings of shape
    ``(L, dim)`` using half-dim sines and half-dim cosines.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        # Frequency scale factor
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = x[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps into dense vector representations.

    Used in MIM (masked image modelling) models that follow a diffusion
    schedule.  Internally applies a fixed sinusoidal frequency encoding
    followed by a small two-layer MLP with SiLU activation.

    Args:
        hidden_size:             Output embedding dimension.
        frequency_embedding_size: Dimension of the intermediate sinusoidal
                                  frequency encoding.  Defaults to ``256``.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Adapted from https://github.com/openai/glide-text2im.

        Args:
            t:          1-D tensor of ``N`` timestep indices (may be fractional).
            dim:        Output embedding dimension.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            Tensor of shape ``(N, dim)``.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# 2-D sin/cos positional embeddings (MAE / MIM style)
# Reference: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# ---------------------------------------------------------------------------


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """Build 2-D sin/cos positional embeddings for a square spatial grid.

    Args:
        embed_dim:    Embedding dimension.
        grid_size:    Height (and width) of the grid.
        cls_token:    If ``True``, prepend ``extra_tokens`` zero-embedding rows
                      for CLS (or other prefix) tokens.
        extra_tokens: Number of prefix zero-embedding rows when
                      ``cls_token=True``.

    Returns:
        Array of shape ``(grid_size * grid_size, embed_dim)`` or
        ``(extra_tokens + grid_size * grid_size, embed_dim)`` when
        ``cls_token=True``.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # w goes first to match the convention in the MAE reference implementation
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Combine per-axis 1-D embeddings into a 2-D grid embedding.

    Args:
        embed_dim: Total embedding dimension (must be even).
        grid:      Grid coordinates of shape ``(2, 1, H, W)``.

    Returns:
        Array of shape ``(H * W, embed_dim)``.
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Build 1-D sinusoidal positional embeddings from a position array.

    Args:
        embed_dim: Output dimension per position (must be even).
        pos:       Position values of any shape; will be flattened to ``(M,)``.

    Returns:
        Array of shape ``(M, embed_dim)``.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)
