"""Geometry and rank metrics for token representations.

All functions are decorated with ``@torch.no_grad()`` and operate on batched
tensor inputs produced by the attention heads of a transformer model.

Functions
---------
compute_average_angle
    Average pairwise angle (in degrees) between token vectors across a batch.
compute_effective_rank
    Effective rank via the entropy of the normalised singular value spectrum.
compute_rank
    Numerical matrix rank computed via ``torch.linalg.matrix_rank``.
"""

import torch


@torch.no_grad()
def compute_average_angle(vectors: torch.Tensor) -> torch.Tensor:
    """Compute the average pairwise angle between row vectors.

    Args:
        vectors: Tensor of shape ``(B, H, N, d)`` where *B* is batch size,
                 *H* is number of heads, *N* is sequence length, and *d* is
                 head dimension.

    Returns:
        Average pairwise angle in **degrees**, shape ``(B, H, 1)``.
    """
    # Normalize the vectors to unit length
    normalized_vectors = torch.nn.functional.normalize(vectors, dim=-1)

    # Compute the cosine similarity (alignment) using matrix multiplication
    alignment = torch.matmul(normalized_vectors, normalized_vectors.transpose(-1, -2))

    alignment.clamp_(-1.0, 1.0)

    # Compute the angle in radians using arccos
    angles_rad = torch.acos(alignment)

    # Exclude the diagonal (angle between a vector and itself is 0)
    angles_rad_mask = angles_rad[
        ~torch.eye(angles_rad.shape[-1], dtype=torch.bool).repeat(angles_rad.shape[0], angles_rad.shape[1], 1, 1)
    ].view(angles_rad.shape[0], angles_rad.shape[1], -1)

    # Compute the average angle in radians then convert to degrees
    average_angle_rad = angles_rad_mask.mean(dim=-1, keepdim=True)
    average_angle_deg = torch.rad2deg(average_angle_rad)

    return average_angle_deg


@torch.no_grad()
def compute_effective_rank(x: torch.Tensor) -> torch.Tensor:
    """Compute the effective rank via singular-value entropy.

    Effective rank is defined as ``exp(H(σ))`` where ``H`` is the Shannon
    entropy of the normalised singular value distribution.

    Args:
        x: Tensor of shape ``(B, H, N, d)``.

    Returns:
        Effective rank tensor of shape ``(B, H, 1)``.
    """
    s = torch.linalg.svdvals(x)
    normalized_s = s / s.sum(dim=-1, keepdim=True)
    return (-normalized_s * normalized_s.log()).sum(dim=-1, keepdim=True).exp()


@torch.no_grad()
def compute_rank(x: torch.Tensor) -> torch.Tensor:
    """Compute the numerical matrix rank.

    Args:
        x: Tensor of shape ``(B, H, N, d)``.

    Returns:
        Integer rank as a float tensor of shape ``(B, H, 1)``.
    """
    return torch.linalg.matrix_rank(x).float().unsqueeze(-1)
