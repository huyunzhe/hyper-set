"""
Dataset loading and data augmentation for image classification.

Supports CIFAR-10, CIFAR-100, SVHN, ImageNet-100, and ImageNet-1k.
Transforms are built from the Hydra config, matching the original
per-dataset defaults used in the argparse-based training scripts.
"""

import os
from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ic.augmentation import CIFAR10Policy, ImageNetPolicy, RandomCropPaste, SVHNPolicy

# ── Per-dataset defaults ──────────────────────────────────────────────────────

# image size, channel count, number of classes, normalization stats
_DATASET_META = {
    "c10": {
        "size": 32,
        "in_c": 3,
        "num_classes": 10,
        "padding": 4,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616],
    },
    "c100": {
        "size": 32,
        "in_c": 3,
        "num_classes": 100,
        "padding": 4,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
    },
    "svhn": {
        "size": 32,
        "in_c": 3,
        "num_classes": 10,
        "padding": 4,
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
    },
    "in100": {
        "size": 224,
        "in_c": 3,
        "num_classes": 100,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "in1k": {
        "size": 224,
        "in_c": 3,
        "num_classes": 1000,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

# When fine-tuning small datasets with a pretrained ImageNet model,
# resize to 224 and use ImageNet normalisation stats.
_FINETUNE_OVERRIDE = {
    "size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def get_dataset_meta(cfg: DictConfig) -> dict:
    """Return dataset-level metadata (size, channels, classes, stats).

    When ``experiment.ckpt_path`` is set for a small dataset (c10/c100),
    the image size and normalization are overridden to match ImageNet defaults
    so that a pretrained backbone can be used directly.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Dictionary with keys: size, in_c, num_classes, mean, std,
        and optionally padding (for CIFAR / SVHN random crop).
    """
    dataset_name = cfg.data.dataset
    if dataset_name not in _DATASET_META:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")
    meta = dict(_DATASET_META[dataset_name])

    # Override for fine-tuning small datasets with a large pretrained model
    is_finetune = cfg.experiment.get("ckpt_path") is not None
    if is_finetune and dataset_name in ("c10", "c100"):
        meta.update(_FINETUNE_OVERRIDE)

    return meta


def build_transforms(
    cfg: DictConfig,
    meta: dict,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and test torchvision transform pipelines.

    Args:
        cfg: Hydra configuration object.
        meta: Dataset metadata returned by :func:`get_dataset_meta`.

    Returns:
        A tuple ``(train_transform, test_transform)``.
    """
    dataset = cfg.data.dataset
    size = meta["size"]
    mean = meta["mean"]
    std = meta["std"]
    is_imagenet = dataset in ("in100", "in1k")
    is_finetune = cfg.experiment.get("ckpt_path") is not None

    train_tfm: list = []
    test_tfm: list = []

    # ── Spatial transforms ────────────────────────────────────────────────────
    if is_imagenet or (is_finetune and dataset in ("c10", "c100")):
        # ImageNet-style: random resized crop + horizontal flip
        train_tfm += [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
        ]
        test_tfm += [transforms.Resize((size, size))]
    else:
        # CIFAR / SVHN style: random crop with padding
        padding = meta.get("padding", 4)
        train_tfm += [transforms.RandomCrop(size=size, padding=padding)]

    # Horizontal flip (all except SVHN)
    if dataset != "svhn":
        train_tfm += [transforms.RandomHorizontalFlip()]

    # ── AutoAugment ───────────────────────────────────────────────────────────
    if cfg.data.autoaugment:
        if dataset in ("c10", "c100"):
            train_tfm.append(CIFAR10Policy())
        elif dataset == "svhn":
            train_tfm.append(SVHNPolicy())
        elif is_imagenet:
            train_tfm.append(ImageNetPolicy())
        else:
            logger.warning(f"No AutoAugment policy defined for dataset '{dataset}'.")

    # ── Tensor + normalization ────────────────────────────────────────────────
    normalize = transforms.Normalize(mean=mean, std=std)
    train_tfm += [transforms.ToTensor(), normalize]
    test_tfm += [transforms.ToTensor(), normalize]

    # ── RandomCropPaste ───────────────────────────────────────────────────────
    if cfg.data.rcpaste:
        train_tfm += [RandomCropPaste(size=size)]

    return transforms.Compose(train_tfm), transforms.Compose(test_tfm)


class HFImageFolder(Dataset):
    """HuggingFace dataset wrapper with same interface as torchvision.datasets.ImageFolder"""

    def __init__(self, repo_id, split="train", transform=None, target_transform=None):
        assert split in ["train", "val"], "Split must be 'train' or 'val'"
        self.dataset = load_dataset(repo_id, split=split)
        self.transform = transform
        self.target_transform = target_transform

        # Same attributes as ImageFolder
        self.classes = self.dataset.features["label"].names
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.targets = self.dataset["label"]
        self.imgs = list(zip(["hf"] * len(self.dataset), self.targets))  # mimics .imgs
        self.samples = self.imgs  # mimics .samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label  # same as ImageFolder: (tensor, int)


def load_datasets(cfg: DictConfig, is_master: bool) -> Tuple[Dataset, Dataset, dict]:
    """Load train and test datasets for image classification.

    Args:
        cfg: Hydra configuration object.

    Returns:
        A tuple ``(train_dataset, test_dataset, meta)`` where *meta* is the
        dataset metadata dictionary (size, in_c, num_classes, mean, std).
    """
    dataset = cfg.data.dataset
    root = cfg.data.data_root
    meta = get_dataset_meta(cfg)
    train_tfm, test_tfm = build_transforms(cfg, meta)

    if is_master:
        logger.info(f"Loading dataset '{dataset}'")

    if dataset == "c10":
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_tfm, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_tfm, download=True)

    elif dataset == "c100":
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_tfm, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_tfm, download=True)

    elif dataset == "svhn":
        train_ds = torchvision.datasets.SVHN(root, split="train", transform=train_tfm, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_tfm, download=True)

    elif dataset == "in100":
        in100_path = cfg.data.in100_path
        in100_train_path = os.path.join(in100_path, "train")
        in100_val_path = os.path.join(in100_path, "val")
        try:
            train_ds = torchvision.datasets.ImageFolder(in100_train_path, transform=train_tfm)
            test_ds = torchvision.datasets.ImageFolder(in100_val_path, transform=test_tfm)
        except Exception as e:
            if is_master:
                logger.warning(f"Failed to load ImageNet-100 from disk: {e}. Falling back to HuggingFace dataset.")
            train_ds = HFImageFolder("Yunzhe/ImageNet-100", split="train", transform=train_tfm)
            test_ds = HFImageFolder("Yunzhe/ImageNet-100", split="val", transform=test_tfm)

    elif dataset == "in1k":
        in1k_path = cfg.data.in1k_path
        in1k_train_path = os.path.join(in1k_path, "train")
        in1k_val_path = os.path.join(in1k_path, "val")
        try:
            train_ds = torchvision.datasets.ImageFolder(in1k_train_path, transform=train_tfm)
            test_ds = torchvision.datasets.ImageFolder(in1k_val_path, transform=test_tfm)
        except Exception as e:
            if is_master:
                logger.warning(f"Failed to load ImageNet-1k from disk: {e}. Falling back to HuggingFace dataset.")
            train_ds = HFImageFolder("Yunzhe/ImageNet-1k", split="train", transform=train_tfm)
            test_ds = HFImageFolder("Yunzhe/ImageNet-1k", split="val", transform=test_tfm)

    else:
        raise NotImplementedError(f"Dataset '{dataset}' is not implemented.")

    if is_master:
        logger.info(f"Dataset loaded: {len(train_ds)} train samples, {len(test_ds)} test samples")
    return train_ds, test_ds, meta


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: DictConfig,
    is_multi_gpus: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders.

    When *is_multi_gpus* is ``True``, wraps both datasets in a
    :class:`~torch.utils.data.distributed.DistributedSampler` so that each
    rank receives a non-overlapping shard of the data.  The caller must
    invoke ``train_loader.sampler.set_epoch(epoch)`` at the start of each
    training epoch to ensure different shuffles across epochs.

    Args:
        train_dataset: Training dataset.
        test_dataset: Evaluation / test dataset.
        cfg: Hydra configuration object.
        is_multi_gpus: Whether Distributed Data Parallel is active.

    Returns:
        A tuple ``(train_loader, test_loader)``.
    """
    num_workers = cfg.training.num_workers

    if is_multi_gpus:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.eval_batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return train_loader, test_loader
