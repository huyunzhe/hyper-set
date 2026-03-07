"""
Model factory and criterion utilities for image classification.

Provides :func:`create_model` (unified factory for all ViT variants),
:func:`create_criterion`.
"""

import importlib

import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from ic.criterions import LabelSmoothingCrossEntropyLoss

# ── Criterion ─────────────────────────────────────────────────────────────────


def create_criterion(cfg: DictConfig) -> nn.Module:
    """Build the classification loss function.

    Supports standard cross-entropy and cross-entropy with label smoothing.

    Args:
        cfg: Hydra configuration object.

    Returns:
        A :class:`torch.nn.Module` loss function.
    """
    if cfg.training.criterion == "ce":
        if cfg.training.label_smoothing:
            return LabelSmoothingCrossEntropyLoss(cfg.data.num_classes, smoothing=cfg.training.smoothing)
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown criterion '{cfg.training.criterion}'.")


# ── Model registry ────────────────────────────────────────────────────────────

# Maps model.type → ViT class name inside ic/models.py
_MODEL_REGISTRY = {
    "hyper-set": "HyperSET",  # default Hyper-SET
    "hyper-set-lora": "HyperSETLoRA",  # + depth-wise LoRA
    "hyper-set-basic": "HyperSETBasic",  # one extra RMSNorms in time_mlp, for training from scratch on CIFAR-10/100
    "hyper-set-alt-attn": "HyperSETAlternativeAttention",  # alternative attention
    "hyper-set-alt-ff": "HyperSETAlternativeFeedforward",  # alternative feed-forward
    "hyper-set-ss": "HyperSETFixedStepSize",  # fixed step-size
    "transformer": "ViT",  # standard ViT
    "crate": "CRATE",  # CRATE
    "crate-T": "CRATET",  # CRATE-T
    "et": "ET",  # Energy Transformer
}


def create_model(cfg: DictConfig, is_master: bool = True) -> nn.Module:
    """Instantiate the correct ViT variant from configuration.

    The ``model.type`` key selects the architecture (see :data:`_MODEL_REGISTRY`).

    Args:
        cfg: Hydra configuration object.
        is_master: Whether this is the master process (controls logging).

    Returns:
        An instantiated :class:`torch.nn.Module`.

    Raises:
        KeyError: If ``model.type`` is not in :data:`_MODEL_REGISTRY`.
    """
    model_type = cfg.model.type

    if model_type not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model type '{model_type}'. Valid options: {list(_MODEL_REGISTRY.keys())}")

    ViT = getattr(importlib.import_module("ic.models"), _MODEL_REGISTRY[model_type])

    # Keyword arguments shared by all model variants
    kwargs = dict(
        num_classes=cfg.data.num_classes,
        img_size=cfg.data.size,
        patch=cfg.model.patch,
        dropout=cfg.model.dropout,
        mlp_hidden=cfg.model.mlp_hidden,
        n_layer=cfg.model.n_layer,
        n_recur=cfg.model.n_recur,
        n_embd=cfg.model.n_embd,
        head=cfg.model.n_head,
        use_cls_token=cfg.model.use_cls_token,
        input_cond=cfg.model.input_cond,
        time_embed=cfg.model.time_emb,
    )

    # Model-type-specific additional arguments
    if model_type in ("hyper-set", "hyper-set-basic", "hyper-set-lora"):
        kwargs["n_recur"] = cfg.model.n_recur

    if model_type == "hyper-set-lora":
        kwargs["r"] = cfg.model.lora_r
        kwargs["alpha"] = cfg.model.lora_alpha

    if model_type == "hyper-set-alt-attn":
        kwargs["attention"] = cfg.model.attention

    if model_type == "hyper-set-alt-ff":
        kwargs["ff"] = cfg.model.ff

    if model_type in ("hyper-set-ss", "transformer", "crate", "crate-T", "et"):
        if model_type == "hyper-set-ss":
            kwargs["stepsize"] = cfg.model.step_size
        if model_type == "et":
            del kwargs["mlp_hidden"]  # not used in ET
        del kwargs["time_embed"]  # not used in this variant
        del kwargs["input_cond"]  # not used in this variant

    model = ViT(**kwargs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_master:
        logger.info(
            f"Model '{model_type}' ({_MODEL_REGISTRY[model_type]}) created: "
            f"{n_params / 1e6:.3f}M trainable parameters | "
            f"n_layer={cfg.model.n_layer}, n_recur={cfg.model.n_recur}, "
            f"n_embd={cfg.model.n_embd}, n_head={cfg.model.n_head},"
            f"mlp_hidden={cfg.model.mlp_hidden}, time_emb={cfg.model.time_emb}."
        )
    return model
