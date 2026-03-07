import math
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SudokuTrainer:
    """
    Trainer for Sudoku-solving models using the Hyper-SET framework.

    Args:
        model: The neural network model to train
        train_dataset: Labeled training dataset
        test_dataset: Test dataset for evaluation
        cfg: Hydra configuration object
        device: torch.device for training (CPU or CUDA)
        wandb_run: Optional wandb run object for logging
        train_dataset_ulb: Optional unlabeled training dataset for semi-supervised learning
        eval_func: Evaluation functions to run during training
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        cfg: DictConfig,
        device: torch.device,
        wandb_run=None,
        train_dataset_ulb: Optional[Dataset] = None,
        eval_func: Optional[Callable] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.train_dataset_ulb = train_dataset_ulb
        self.test_dataset = test_dataset
        self.cfg = cfg
        self.device = device
        self.wandb = wandb_run

        # Extract configuration
        self.batch_size = cfg.training.batch_size
        self.label_size = cfg.data.label_size
        self.max_epochs = cfg.training.epochs
        self.lr = cfg.training.lr
        self.lr_decay = cfg.training.lr_decay
        self.weight_decay = cfg.training.weight_decay
        self.eval_interval = cfg.training.eval_interval
        self.num_workers = cfg.training.num_workers

        # Experiment tracking
        self.ckpt_name = cfg.experiment.ckpt_name
        self.eval_func = eval_func if eval_func else None
        self.result = {self.eval_func.__name__: []} if self.eval_func else None

        # Set up data loaders
        self._setup_dataloaders()

        # Move model to device
        self.model.to(self.device)

        # Set up optimizer
        self._setup_optimizer()

        logger.info(f"Trainer initialized with {len(self.train_dataset)} training samples")
        if self.train_dataset_ulb:
            logger.info(f"Using semi-supervised learning with {len(self.train_dataset_ulb)} unlabeled samples")

    def _setup_dataloaders(self):
        """
        Set up PyTorch DataLoaders for training and evaluation.

        Creates separate loaders for:
        - Training (labeled and unlabeled data if semi-supervised)
        - Validation (subset of training data)
        - Testing
        """
        # Create evaluation dataloader (subset of train data)
        train_eval_size = min(1000, len(self.train_dataset))
        train_dataset_eval = torch.utils.data.Subset(self.train_dataset, list(range(train_eval_size)))
        self.train_eval_dataloader = DataLoader(
            train_dataset_eval,
            batch_size=self.batch_size,
        )

        # Create test dataloader
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )

        # Create training dataloaders
        if self.label_size < self.batch_size and self.train_dataset_ulb:
            # Semi-supervised learning: separate loaders for labeled and unlabeled
            logger.info("Setting up semi-supervised learning dataloaders")
            self.loader_lb = DataLoader(
                self.train_dataset,
                batch_size=self.label_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
            )
            self.loader_ulb = DataLoader(
                self.train_dataset_ulb,
                batch_size=self.batch_size - self.label_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        else:
            # Fully supervised learning
            self.loader_lb = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                persistent_workers=True,  # set True
            )
            self.loader_ulb = None

    def _save_checkpoint(self, suffix: str = "") -> None:
        """
        Save model checkpoint to disk.

        Args:
            suffix: Optional suffix to add to checkpoint filename
        """
        # Get the raw model (unwrap DataParallel if needed)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # Create checkpoint directory
        ckpt_dir = f"{self.cfg.run_dir}/checkpoint/"
        os.makedirs(ckpt_dir, exist_ok=True)

        # Determine filename
        if suffix:
            filename = f"{self.ckpt_name}_{suffix}.pth"
        else:
            filename = f"{self.ckpt_name}.pth"

        save_path = os.path.join(ckpt_dir, filename)

        # Save model state dict
        torch.save(raw_model.state_dict(), save_path)
        logger.info(f"Saved checkpoint: {save_path}")

    def _setup_optimizer(self):
        """
        Set up the optimizer for training.

        Uses AdamW with parameters from configuration.
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2),
            weight_decay=self.weight_decay,
        )

    def _run_epoch(self, epoch: int, split: str) -> float:
        """
        Run one epoch of training or evaluation.

        Args:
            epoch: Current epoch number (0-indexed)
            split: Either 'train' or 'test'

        Returns:
            Average loss for the epoch
        """
        is_train = split == "train"
        self.model.train(is_train)
        losses = []

        dataloader = self.loader_lb if is_train else self.test_dataloader
        desc = f"{'Train' if is_train else 'Test'} Epoch {epoch + 1}/{self.max_epochs}"

        if is_train:
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
        else:
            pbar = enumerate(dataloader)

        for it, (x, y) in pbar:
            # Move data to device
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            with torch.set_grad_enabled(is_train):
                logits, loss = self.model(x, y)
                loss = loss.mean()
                losses.append(loss.item())

            # Backward pass (training only)
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update learning rate
                lr = self._get_lr(epoch, self.optimizer)

                # Update progress bar
                if isinstance(pbar, tqdm):
                    pbar.set_postfix({"loss": f"{loss.item():.5f}", "lr": f"{lr:e}"})

        return float(np.mean(losses))

    def _get_lr(self, epoch: int, optimizer) -> float:
        """
        Get and apply learning rate for current epoch.

        Uses cosine annealing if lr_decay is enabled.

        Args:
            epoch: Current epoch number
            optimizer: PyTorch optimizer

        Returns:
            Current learning rate
        """
        if self.lr_decay:
            # Cosine annealing from lr to 1e-7
            progress = float(epoch / self.max_epochs)
            lr = 1e-7 + 0.5 * (self.lr - 1e-7) * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = self.lr

        return lr

    def _evaluate(self, epoch: int) -> None:
        """
        Run evaluation functions and log results.

        Args:
            epoch: Current epoch number
        """
        eval_func = self.eval_func if self.eval_func else None
        if eval_func is None:
            logger.warning("No evaluation function provided, skipping evaluation")
            return
        # Evaluate on test set
        result = eval_func(self.model, self.test_dataloader, return_stats=False)
        self.result[eval_func.__name__].append(result)

        if self.wandb:
            # Extract accuracy metrics
            correct, total, single_correct, single_total = result
            board_acc = 100 * correct / total
            cell_acc = 100 * single_correct / single_total

            # Log test accuracies
            self.wandb.log(
                {
                    f"test_acc/{eval_func.__name__}_board": board_acc,
                    f"test_acc/{eval_func.__name__}_cell": cell_acc,
                },
                step=epoch + 1,
            )

            # Evaluate with different recurrence steps (for single-layer models)
            if self.cfg.model.n_layer == 1:
                prev_n_recur = self.model.n_recur

                for n_recur in [16, 32, 64]:
                    self.model.n_recur = n_recur
                    correct, total, single_correct, single_total = eval_func(
                        self.model, self.test_dataloader, return_stats=False
                    )

                    board_acc = 100 * correct / total
                    cell_acc = 100 * single_correct / single_total

                    self.wandb.log(
                        {
                            f"test_acc/{eval_func.__name__}_board[{n_recur}]": board_acc,
                            f"test_acc/{eval_func.__name__}_cell[{n_recur}]": cell_acc,
                        },
                        step=epoch + 1,
                    )

                # Restore original n_recur
                self.model.n_recur = prev_n_recur

            # Evaluate on training set
            correct, total, single_correct, single_total = eval_func(
                self.model, self.train_eval_dataloader, return_stats=False
            )

            train_board_acc = 100 * correct / total
            train_cell_acc = 100 * single_correct / single_total

            self.wandb.log(
                {
                    f"train_acc/{eval_func.__name__}_board": train_board_acc,
                    f"train_acc/{eval_func.__name__}_cell": train_cell_acc,
                },
                step=epoch + 1,
            )

            logger.info(
                f"Epoch {epoch + 1}: Test Board Acc={board_acc:.2f}%, "
                f"Test Cell Acc={cell_acc:.2f}%, "
                f"Train Board Acc={train_board_acc:.2f}%",
                f"Train Cell Acc={train_cell_acc:.2f}%",
            )

    def train(self) -> Dict:
        """
        Main training loop.

        Runs training for the specified number of epochs, evaluating at intervals,
        saving checkpoints, and optionally visualizing attention.

        Returns:
            Dictionary containing evaluation results
        """

        best_loss = float("inf")

        for epoch in range(self.max_epochs):
            # Training epoch
            train_loss = self._run_epoch(epoch, "train")

            # Log training loss
            if self.wandb:
                self.wandb.log({"loss/train_loss": train_loss}, step=epoch + 1)

            # Validation epoch
            if self.test_dataset is not None:
                test_loss = self._run_epoch(epoch, "test")

                if self.wandb:
                    self.wandb.log({"loss/test_loss": test_loss}, step=epoch + 1)

                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs} - Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}"
                )

                # Save best checkpoint
                if test_loss < best_loss:
                    best_loss = test_loss
                    if self.ckpt_name:
                        logger.info(f"New best model! Test loss: {test_loss}")
                        self._save_checkpoint("best")
            else:
                logger.info(f"Epoch {epoch + 1}/{self.max_epochs} - Train Loss: {train_loss:.5f}")

            # Evaluation
            if (epoch + 1) % self.eval_interval == 0:
                logger.info(f"Running evaluation at epoch {epoch + 1}")
                self._evaluate(epoch)

        # Save final checkpoint
        if self.ckpt_name:
            self._save_checkpoint("last")

        return self.result

    def evaluate(
        self,
    ):
        """
        Run evaluation only (load checkpoint and evaluate).

        Args:
            trainer: SudokuTrainer instance
        """

        # Load checkpoint
        ckpt_path = f"{self.cfg.experiment.ckpt_path}"
        if ckpt_path == "None":
            raise NotImplementedError("No checkpoint path provided for evaluation")

        if os.path.isdir(ckpt_path):
            ckpt_path = f"{ckpt_path}/{self.ckpt_name}_best.pth"
            assert os.path.isfile(ckpt_path), f"No checkpoint found at {ckpt_path}"

        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return

        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

        # Run evaluation
        if self.eval_func is None:
            logger.error("No evaluation function provided, skipping evaluation")
            return 0, 0, 0, 0, 0, 0

        if self.cfg.mode == "vis":
            assert self.cfg.model.type == "hyper-set" and self.cfg.model.n_layer == 1, (
                "Visualization only supported for single-layer Hyper-SET."
            )
            (
                correct,
                total,
                single_correct,
                single_total,
                energy_traj,
                energy_attn_traj,
                energy_ff_traj,
                effective_rank_traj,
                average_angle_traj,
                rank_traj,
            ) = self.eval_func(self.model, self.test_dataloader, return_stats=True)
            saved_stats = {
                self.model.n_recur: {
                    "energy_attn_traj": energy_attn_traj,
                    "energy_ff_traj": energy_ff_traj,
                    "effective_rank": effective_rank_traj,
                    "average_angle": average_angle_traj,
                    "rank": rank_traj,
                }
            }
            logger.info("Saving evaluation stats for visualization...")
            torch.save(saved_stats, "sudoku_stats.pth")
        else:
            correct, total, single_correct, single_total = self.eval_func(
                self.model, self.test_dataloader, return_stats=False
            )

        board_acc = 100 * correct / total
        cell_acc = 100 * single_correct / single_total

        return board_acc, cell_acc, correct, total, single_correct, single_total
