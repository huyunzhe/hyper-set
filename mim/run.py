"""
MIM (Masked Image Modeling) task entry point.

This module provides the `run_task` function called by the root `main.py`.
It orchestrates dataset loading, model creation, VQGAN loading, trainer setup,
and dispatches to training, evaluation, or reconstruction evaluation modes.

Supports multi-GPU DDP training via torchrun. When launched with torchrun,
this module automatically initializes the NCCL process group and cleans up
after the task completes.
"""

import os

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig

from mim.dataset import create_dataloaders, load_datasets
from mim.trainer import MIMTrainer
from mim.utils import create_model, load_vqgan


def _setup_ddp() -> None:
    """Initialize DDP process group and set CUDA device for the local rank.

    Should only be called when WORLD_SIZE > 1 (i.e., launched via torchrun).
    """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logger.info(
        f"DDP initialized: rank {dist.get_rank()}/{dist.get_world_size()}, local_rank {os.environ['LOCAL_RANK']}"
    )


def _teardown_ddp() -> None:
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed.")


def run_task(cfg: DictConfig, device: torch.device, wandb_run=None) -> None:
    """Main entry point for the MIM task.

    Args:
        cfg: Hydra configuration object.
        device: torch.device for computation.
        wandb_run: Optional wandb run object for logging.
    """
    # Detect multi-GPU environment (set by torchrun)
    is_multi_gpus = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0

    # Initialize DDP if multi-GPU
    if is_multi_gpus:
        _setup_ddp()
        device = torch.device(f"cuda:{local_rank}")

    if is_master:
        logger.info("=" * 80)
        logger.info("MIM Task - MaskGIT Masked Image Modeling")
        logger.info("=" * 80)
        if is_multi_gpus:
            logger.info(f"Multi-GPU mode: rank {local_rank}, device {device}")

    # 1. Load datasets
    train_dataset, test_dataset, nclass = load_datasets(cfg, is_master)

    # 2. Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        is_multi_gpus=is_multi_gpus,
        is_master=is_master,
    )

    # 3. Load VQGAN autoencoder
    ae = load_vqgan(cfg, device, is_master)
    codebook_size = ae.n_embed

    # 4. Create MaskTransformer model
    vit = create_model(cfg, codebook_size=codebook_size, mode=cfg.mode, nclass=nclass, is_master=is_master)

    # Watch model with wandb if enabled (master only)
    if wandb_run and is_master and hasattr(wandb_run, "watch"):
        wandb_run.watch(vit, log_freq=100)

    # 5. Create trainer
    trainer = MIMTrainer(
        vit=vit,
        ae=ae,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        wandb_run=wandb_run,
        is_multi_gpus=is_multi_gpus,
        is_master=is_master,
    )

    # 6. Dispatch based on mode
    if cfg.mode == "train":
        if is_master:
            logger.info("Mode: Training")
        trainer.train()
    elif cfg.mode == "eval":
        if is_master:
            logger.info("Mode: Evaluation (FID)")
        metrics = trainer.eval()
        if is_master:
            logger.info(f"Evaluation metrics: {metrics}")
    elif cfg.mode == "eval_recon":
        if is_master:
            logger.info("Mode: Reconstruction Evaluation")
        trainer.eval_recon()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Choose from: train, eval, eval_recon")

    if is_master:
        logger.info("MIM task completed!")

    # Always clean up DDP
    if is_multi_gpus:
        _teardown_ddp()
