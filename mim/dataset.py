"""
Dataset loading utilities for the MIM (Masked Image Modeling) task.

Provides dataset factory functions for loading and preprocessing image datasets
(MNIST, CIFAR-10, STL-10, ImageNet-100, ImageNet, MS-COCO) with appropriate
transforms for VQGAN-based masked image modeling.
"""

import os
from typing import Tuple

from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, STL10, ImageFolder
from torchvision.datasets.coco import CocoCaptions


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


# Default data paths per dataset (Change for your own)
_DEFAULT_DATA_PATHS = {
    "mscoco_train": "/datasets_master/COCO/images/train2017/",
    "mscoco_train_ann": "/datasets_master/COCO/annotations/captions_train2017.json",
    "mscoco_val": "/datasets_master/COCO/images/val2017/",
    "mscoco_val_ann": "/datasets_master/COCO/annotations/captions_val2017.json",
}


def _get_transforms(img_size: int, split: str = "train") -> transforms.Compose:
    """Create image transforms for the given split.

    Args:
        img_size: Target image size.
        split: Data split ('train' or 'val'/'test').

    Returns:
        Composed transforms pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )


def load_datasets(cfg: DictConfig, is_master: bool) -> Tuple[Dataset, Dataset, int]:
    """Load train and test datasets based on configuration.

    Args:
        cfg: Hydra configuration object with data settings.
        is_master: Whether the current process is the master process.
    Returns:
        Tuple of (train_dataset, test_dataset, nclass) where nclass is the
        number of classes in the dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    dataset_name = cfg.data.dataset
    img_size = cfg.data.img_size

    if is_master:
        logger.info(f"Loading dataset: {dataset_name} (img_size={img_size})")

    if dataset_name == "mnist":
        t = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        data_train = MNIST("./data/mnist", download=True, transform=t)
        data_test = data_train  # MNIST uses same set for simplicity
        nclass = 10

    elif dataset_name == "cifar10":
        t_train = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        t_test = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        data_train = CIFAR10("./data/CIFAR10/", train=True, download=True, transform=t_train)
        data_test = CIFAR10("./data/CIFAR10/", train=False, download=False, transform=t_test)
        nclass = 10

    elif dataset_name == "stl10":
        t_train = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        t_test = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        data_train = STL10("./data/stl10", split="train+unlabeled", transform=t_train)
        data_test = STL10("./data/stl10", split="test", transform=t_test)
        nclass = 10

    elif dataset_name in ("in100", "in1k"):
        t_train = _get_transforms(img_size, "train")
        t_test = _get_transforms(img_size, "val")
        data_folder = cfg.data.in100_path if dataset_name == "in100" else cfg.data.in1k_path
        try:
            data_train = ImageFolder(os.path.join(data_folder, "train"), transform=t_train)
            data_test = ImageFolder(os.path.join(data_folder, "val"), transform=t_test)
        except Exception as e:
            if is_master:
                logger.warning(f"Failed to load ImageFolder for {dataset_name} at {data_folder}: {e}")
                logger.info("Falling back to HuggingFace dataset loading...")
            repo_id = "Yunzhe/ImageNet-100" if dataset_name == "in100" else "Yunzhe/ImageNet-1k"
            data_train = HFImageFolder(repo_id, split="train", transform=t_train)
            data_test = HFImageFolder(repo_id, split="val", transform=t_test)
        nclass = 100 if dataset_name == "in100" else 1000

    elif dataset_name == "mscoco":

        def _cap_lambda(x):
            return x[:5]

        t_train = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        t_test = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        data_train = CocoCaptions(
            root=_DEFAULT_DATA_PATHS["mscoco_train"],
            annFile=_DEFAULT_DATA_PATHS["mscoco_train_ann"],
            transform=t_train,
            target_transform=_cap_lambda,
        )
        data_test = CocoCaptions(
            root=_DEFAULT_DATA_PATHS["mscoco_val"],
            annFile=_DEFAULT_DATA_PATHS["mscoco_val_ann"],
            transform=t_test,
            target_transform=_cap_lambda,
        )
        nclass = 0  # COCO captions have no class labels

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: mnist, cifar10, stl10, in100, in1k, mscoco")

    if is_master:
        logger.info(f"Dataset loaded: {len(data_train)} train, {len(data_test)} test, {nclass} classes")
    return data_train, data_test, nclass


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    is_multi_gpus: bool = False,
    is_master: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders with optional distributed sampling.

    Args:
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        is_multi_gpus: Whether to use DistributedSampler for multi-GPU training.
        is_master: Whether the current process is the master process for logging purposes.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_multi_gpus else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_multi_gpus else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not is_multi_gpus),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    if is_master:
        logger.info(f"DataLoaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")
    return train_loader, test_loader
