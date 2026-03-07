"""
Model and VQGAN factory utilities for the MIM task.

Provides functions to create the MaskTransformer model and load the
pretrained VQGAN tokenizer from configuration.
"""

import os

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from mim.models import MaskTransformerUncond
from mim.Taming.models.vqgan import VQModel


def load_vqgan(cfg: DictConfig, device: torch.device, is_master: bool) -> VQModel:
    """Load the pretrained VQGAN autoencoder for image tokenization.

    Args:
        cfg: Hydra configuration with vqgan.folder path.
        device: Device to load the model onto.

    Returns:
        Pretrained VQModel in eval mode.
    """
    vqgan_folder = cfg.vqgan.folder
    config_path = os.path.join(vqgan_folder, "model.yaml")
    ckpt_path = os.path.join(vqgan_folder, "last.ckpt")

    if is_master:
        logger.info(f"Loading VQGAN from: {vqgan_folder}")

    # Load VQGAN config and weights
    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model = model.eval()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if is_master:
        logger.info(f"VQGAN loaded: {n_params / 1e6:.3f}M parameters, codebook_size={model.n_embed}")

    return model


def create_model(
    cfg: DictConfig, codebook_size: int, mode: str = "train", nclass: int = 0, is_master: bool = True
) -> nn.Module:
    """Create the MaskTransformer model from configuration.

    Args:
        cfg: Hydra configuration with model settings.
        codebook_size: Size of the VQGAN codebook (needed for output layer).
        mode: 'train' for training architecture, 'eval' or 'eval_recon' for evaluation architecture.
        nclass: Number of classes in the dataset.

    Returns:
        Instantiated MaskTransformerUncond model.
    """
    if is_master:
        logger.info(f"Creating model ({mode})...")

    model = MaskTransformerUncond(
        model=cfg.model.type,
        img_size=cfg.data.img_size,
        codebook_size=codebook_size,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_recur=cfg.model.n_recur,
        head=cfg.model.n_head,
        multiplier=cfg.model.multiplier,
        dropout=cfg.training.dropout,
        nclass=nclass,
        phi_func=cfg.model.get("phi_func", None),
        adaptive_mode=cfg.model.adaptive_mode,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_master:
        logger.info(f"Model created: {n_params / 1e6:.3f}M trainable parameters")
        logger.info(
            f"Architecture: {cfg.model.type}, {cfg.model.n_layer} layers, "
            f"{cfg.model.n_head} heads, {cfg.model.n_embd} dim, "
            f"{cfg.model.n_recur} recurrence, multiplier={cfg.model.multiplier}"
        )

    return model
