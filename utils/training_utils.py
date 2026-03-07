import random

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_device(device: str = "auto") -> torch.device:
    """Set up the compute device. Options: 'auto', 'cuda', or 'cpu'."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    return device


def setup_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_wandb(cfg: DictConfig):
    """Initialize Weights & Biases logging if enabled."""

    if not cfg.wandb:
        return None

    try:
        import wandb

        project_name = f"Hyper-SET-{cfg.task_name}"

        run_name = (
            f"{cfg.model.type}_layer{cfg.model.n_layer}_recur{cfg.model.n_recur}_dim{cfg.model.n_embd}_seed{cfg.seed}"
        )

        run_id = None  # Optionally set a unique run ID for resuming runs

        run_dir = HydraConfig.get().runtime.output_dir

        cfg.run_dir = run_dir

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=run_dir,
            notes=cfg.experiment.comment,
            group=f"{cfg.model.type}_layer{cfg.model.n_layer}_recur{cfg.model.n_recur}_dim{cfg.model.n_embd}",
            resume="allow" if run_id else None,
            id=run_id,  # NOTE: Generate unique ID for each run if needed
        )

        logger.info(f"Weights & Biases initialized for project: {project_name}")
        return wandb

    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        return None
