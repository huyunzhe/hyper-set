"""
Image Classification (IC) task entry point.

This module provides :func:`run_task`, called by the root ``main.py`` via
Hydra.

Supported model types (``model.type`` in Hydra config):

* ``hyper-set``          – :class:`ic.model.Universal_Ours_Trade_Off_RM_RMS`
* ``hyper-set-lora``     – :class:`ic.model.Universal_Ours_Trade_Off_RM_RMS_LORA`
* ``hyper-set-basic``    – :class:`ic.model.Universal_Ours_Trade_Off`
* ``hyper-set-alt-attn`` – :class:`ic.model.Universal_Ours_Alternative_ATTENTION`
* ``hyper-set-alt-ff``   – :class:`ic.model.Universal_Ours_Alternative_FF`
* ``hyper-set-ss``       – :class:`ic.model.Universal_Ours_Stepsize`
"""

import os

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig

from ic.dataset import create_dataloaders, load_datasets
from ic.trainer import ICTrainer
from ic.utils import create_criterion, create_model

# ── DDP helpers ───────────────────────────────────────────────────────────────


def _setup_ddp() -> None:
    """Initialise the NCCL process group and pin the current process to its GPU."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _teardown_ddp() -> None:
    """Destroy the process group (no-op when DDP was never initialised)."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Task entry point ──────────────────────────────────────────────────────────


def run_task(cfg: DictConfig, device: torch.device, wandb_run=None) -> None:
    """Main entry point for the Image Classification task.

    Called by ``main.py``.  Reads the full Hydra configuration and
    dispatches to training or evaluation depending on ``cfg.mode``.

    When launched via ``torchrun --standalone --nproc_per_node=gpu``, the
    ``WORLD_SIZE`` and ``LOCAL_RANK`` environment variables are set
    automatically and DDP is enabled.

    Args:
        cfg: Hydra configuration object (merged base + task overrides).
        device: ``torch.device`` selected by ``main.py`` (overridden per-rank
                in multi-GPU mode).
        wandb_run: Optional ``wandb`` run initialised in ``main.py``
                   (``None`` when ``cfg.wandb`` is ``false``).
    """
    # ── DDP detection ─────────────────────────────────────────────────────────
    is_multi_gpus = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0

    if is_multi_gpus:
        _setup_ddp()
        device = torch.device(f"cuda:{local_rank}")

    if is_master:
        logger.info("=" * 80)
        logger.info("IC Task – Image Classification")
        if is_multi_gpus:
            logger.info(f"Distributed training: WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={local_rank}")
        logger.info("=" * 80)

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    train_dataset, test_dataset, meta = load_datasets(cfg, is_master)

    # Inject resolved image size and num_classes into cfg
    cfg.data.size = meta["size"]
    cfg.data.num_classes = meta["num_classes"]
    cfg.data.in_c = meta["in_c"]

    # ── 2. Create DataLoaders ─────────────────────────────────────────────────
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, cfg, is_multi_gpus=is_multi_gpus)

    # ── 3. Create model ───────────────────────────────────────────────────────
    model = create_model(cfg, is_master=is_master)

    # ── 4. Create loss criterion ──────────────────────────────────────────────
    criterion = create_criterion(cfg)

    # ── 5. Create trainer ─────────────────────────────────────────────────────
    trainer = ICTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        cfg=cfg,
        device=device,
        wandb_run=wandb_run,
        is_multi_gpus=is_multi_gpus,
        is_master=is_master,
    )

    # ── 6. Dispatch ───────────────────────────────────────────────────────────
    if cfg.mode == "train":
        if is_master:
            logger.info("=" * 80)
            logger.info("Starting training ...")
            logger.info("=" * 80)
        trainer.train()
        if is_master:
            logger.info("=" * 80)
            logger.info("Training complete.")
            logger.info("=" * 80)

    elif cfg.mode == "eval":
        if is_master:
            logger.info("=" * 80)
            logger.info("Running evaluation ...")
            logger.info("=" * 80)
        val_loss, val_acc = trainer.evaluate()
        if is_master:
            logger.info("=" * 80)
            logger.info(f"Results  |  loss={val_loss:.4f}  |  top-1 acc={val_acc * 100:.2f}%")
            logger.info("=" * 80)
    else:
        raise ValueError(f"Unknown mode '{cfg.mode}'. Expected 'train' or 'eval'.")

    # ── 7. DDP teardown ───────────────────────────────────────────────────────
    if is_multi_gpus:
        _teardown_ddp()
