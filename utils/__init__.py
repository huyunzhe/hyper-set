"""Shared utility helpers for the hyper-set project.

Sub-modules
-----------
metric_utils
    Rank and angle diagnostics for analysing representation geometry.
training_utils
    Device setup, random seeding, and Weights & Biases initialisation.
"""

from utils.metric_utils import (
    compute_average_angle,
    compute_effective_rank,
    compute_rank,
)
from utils.training_utils import setup_device, setup_seed, setup_wandb

__all__ = [
    # metric_utils
    "compute_average_angle",
    "compute_effective_rank",
    "compute_rank",
    # training_utils
    "setup_device",
    "setup_seed",
    "setup_wandb",
]
