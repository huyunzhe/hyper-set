"""
Image Classification Trainer for the Hyper-SET framework.

This trainer handles:
- Training ViT-based models with optional CutMix / MixUp augmentation
- Cosine-annealing LR schedule with linear warmup (matching original warmup_scheduler)
- Mixed-precision training via ``torch.amp``
- Multi-GPU training via ``torch.nn.parallel.DistributedDataParallel``
- Top-1 accuracy evaluation
- Checkpoint saving (best + last, full state dict) and optional resume
- Logging via Weights & Biases (master process only)
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from ic.augmentation import CutMix, MixUp


class ICTrainer:
    """Trainer for ViT image-classification models.

    Args:
        model: ViT model.
        train_loader: Training DataLoader.
        test_loader: Test / validation DataLoader.
        criterion: Loss function (cross-entropy or label-smoothing CE).
        cfg: Hydra configuration object.
        device: ``torch.device`` for computation.
        wandb_run: Optional wandb run object for metric logging.
        is_multi_gpus: Whether DDP multi-GPU training is active.
        is_master: Whether this process is the master (rank 0).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        cfg: DictConfig,
        device: torch.device,
        wandb_run=None,
        is_multi_gpus: bool = False,
        is_master: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.wandb = wandb_run
        self.is_multi_gpus = is_multi_gpus
        self.is_master = is_master

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_data = train_loader
        self.test_data = test_loader

        # ── Model ─────────────────────────────────────────────────────────────
        model = model.to(device)
        if is_multi_gpus:
            if is_master:
                logger.info("Wrapping model with DistributedDataParallel.")
            self.model = DDP(model, device_ids=[device])
        else:
            self.model = model

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = criterion.to(device)

        # ── Training hyper-parameters ─────────────────────────────────────────
        self.max_epochs = cfg.training.epochs
        self.lr = cfg.training.lr
        self.min_lr = cfg.training.min_lr
        self.beta1 = cfg.training.beta1
        self.beta2 = cfg.training.beta2
        self.weight_decay = cfg.training.weight_decay
        self.warmup_epoch = cfg.training.warmup_epoch
        self.grad_clip = cfg.training.get("grad_clip", 0.0)
        self.use_amp = cfg.training.amp and torch.cuda.is_available()

        # ── Data augmentation mixers ───────────────────────────────────────────
        self.cutmix = None
        self.mixup = None
        if cfg.data.cutmix and cfg.data.size is not None:
            self.cutmix = CutMix(cfg.data.size, beta=1.0)
        if cfg.data.mixup:
            self.mixup = MixUp(alpha=1.0)

        # ── Experiment tracking ───────────────────────────────────────────────
        self.ckpt_name = cfg.experiment.ckpt_name
        self.run_dir = cfg.run_dir

        # ── Optimizer + schedulers ────────────────────────────────────────────
        self._setup_optimizer()

        # ── AMP scaler ────────────────────────────────────────────────────────
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp and is_master:
            logger.info("Mixed-precision (AMP) training enabled.")

        if is_master:
            logger.info(
                f"ICTrainer ready | epochs={self.max_epochs}, lr={self.lr}, "
                f"warmup={self.warmup_epoch}, amp={self.use_amp}, "
                f"grad_clip={self.grad_clip}, ddp={is_multi_gpus}"
            )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _setup_optimizer(self) -> None:
        """Initialise Adam optimizer and cosine-annealing + warmup schedulers."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        # Cosine annealing after warmup
        if self.cfg.training.lr_decay:
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=self.min_lr,
            )

    def _get_lr(self, epoch: int) -> float:
        """Return the current learning rate using linear warmup + cosine decay.

        During the first ``warmup_epoch`` epochs the LR is scaled linearly from
        ``lr / warmup_epoch`` to ``lr``.  After warmup, the cosine-annealing
        scheduler takes over.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Learning rate for the current epoch.
        """
        if epoch < self.warmup_epoch:
            warmup_factor = (epoch + 1) / max(1, self.warmup_epoch)
            lr = self.lr * warmup_factor
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            if self.cfg.training.lr_decay:
                if epoch == self.warmup_epoch:
                    self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.max_epochs - self.warmup_epoch,
                        eta_min=self.min_lr,
                    )
                self.base_scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
        return lr

    @staticmethod
    def _all_gather(obj: float, gpus: int) -> float:
        """Gather a scalar loss from all DDP ranks and return the mean.

        Args:
            obj: Scalar value on this rank.
            gpus: Total number of GPUs / ranks.

        Returns:
            Mean of *obj* across all ranks.
        """
        tensor_list = [torch.zeros(1) for _ in range(gpus)]
        dist.all_gather_object(tensor_list, obj)
        return torch.FloatTensor(tensor_list).mean().item()

    def _apply_mixup(
        self,
        img: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix or MixUp augmentation to a batch.

        Args:
            img: Image batch of shape ``(B, C, H, W)``.
            label: Integer label tensor of shape ``(B,)``.

        Returns:
            Tuple ``(img, label, rand_label, lambda_)``.
        """
        rand_label = torch.zeros_like(label)
        lambda_ = 1.0

        if self.cutmix is not None:
            img, label, rand_label, lambda_ = self.cutmix((img, label))
        elif self.mixup is not None:
            if np.random.rand() <= 0.8:
                img, label, rand_label, lambda_ = self.mixup((img, label))

        return img, label, rand_label, lambda_

    def _raw_model(self) -> nn.Module:
        """Return the underlying model, unwrapping DDP if necessary."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _save_checkpoint(self, epoch: int, suffix: str = "") -> None:
        """Persist full training state to ``<run_dir>/checkpoint/<ckpt_name>[_suffix].pth``.

        Only the master process writes to disk.  The saved dict contains:
        ``epoch``, ``model_state_dict``, ``optimizer_state_dict``, and
        ``scaler_state_dict``.

        Args:
            epoch: Current epoch number.
            suffix: Optional tag appended to the filename (e.g. ``"best"``).
        """
        if not self.is_master:
            return

        ckpt_dir = os.path.join(self.run_dir, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)

        fname = f"{self.ckpt_name}{'_' + suffix if suffix else ''}.pth"
        save_path = os.path.join(ckpt_dir, fname)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._raw_model().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            },
            save_path,
        )
        if self.is_master:
            logger.info(f"Checkpoint saved → {save_path}")

    def _load_state_dict(self, ckpt_path: str, finetune: bool = False) -> int:
        """Load model weights from a checkpoint file, handling both native and
        PyTorch Lightning checkpoint formats.

        PyTorch Lightning wraps the model state dict under a nested ``state_dict``
        key and prefixes every parameter name with ``"model."``.  This method
        detects that format and strips the prefix automatically.

        Args:
            ckpt_path: Path to the checkpoint file.
            finetune: If ``True``, skip the classification head (``fc.*``) to
                allow loading a checkpoint trained on a different number of
                classes (e.g. ImageNet-1k → CIFAR-100).

        Returns:
            The epoch number stored in the checkpoint (0 if not present).
        """
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # ── PyTorch Lightning checkpoint ──────────────────────────────────────
        if "pytorch-lightning_version" in state or ("state_dict" in state and "model_state_dict" not in state):
            if self.is_master:
                logger.info("Detected PyTorch Lightning checkpoint — unwrapping state_dict.")
                logger.info("State keys: " + ", ".join(state.keys()))
                logger.info("State dict keys: " + ", ".join(state["state_dict"].keys()) + "...")
            pl_sd = state["state_dict"]
            model_sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in pl_sd.items()}
            if not finetune:
                # Only restore optimizer/scaler when resuming, not when fine-tuning
                self.optimizer.load_state_dict(state["optimizer_states"][0])
                if self.scaler is not None and state.get("scaler_states") is not None:
                    self.scaler.load_state_dict(state["scaler_states"][0])
            start_epoch = 0 if finetune else state["epoch"]

        # ── Native full checkpoint ─────────────────────────────────────────────
        elif "model_state_dict" in state:
            model_sd = state["model_state_dict"]
            if not finetune:
                # Only restore optimizer/scaler when resuming, not when fine-tuning
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                if self.scaler is not None and state.get("scaler_state_dict") is not None:
                    self.scaler.load_state_dict(state["scaler_state_dict"])
            start_epoch = 0 if finetune else state.get("epoch", 0)

        # ── Bare model state dict ─────────────────────────────────────
        else:
            model_sd = state
            start_epoch = 0

        # ── Strip classification head for fine-tuning ─────────────────────────
        if finetune:
            head_keys = [k for k in model_sd if k.startswith("fc.")]
            for k in head_keys:
                del model_sd[k]
            if self.is_master and head_keys:
                logger.info(
                    f"Fine-tuning mode: skipped {len(head_keys)} classification-head "
                    f"key(s) {head_keys} — head will be randomly initialised."
                )

        missing, unexpected = self._raw_model().load_state_dict(model_sd, strict=False)
        if self.is_master:
            if missing:
                logger.warning(f"Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                logger.warning(
                    f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
                )

        return start_epoch

    def _resume_checkpoint(self) -> int:
        """Load a checkpoint and return the epoch to resume from.

        Reads ``cfg.experiment.ckpt_path``.  If the path is a directory,
        the ``<ckpt_name>_last.pth`` file inside it is used.

        Returns:
            The epoch number to resume from (0 if no checkpoint is loaded).
        """
        ckpt_path = self.cfg.experiment.ckpt_path
        if not ckpt_path:
            return 0

        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, f"{self.ckpt_name}_last.pth")

        if not os.path.isfile(ckpt_path):
            if self.is_master:
                logger.warning(f"Checkpoint not found at '{ckpt_path}', starting from scratch.")
            return 0

        # state = torch.load(ckpt_path, map_location=self.device)

        # Support both full state dicts (new) and bare model weights (legacy)
        # if "model_state_dict" in state:
        #     self._raw_model().load_state_dict(state["model_state_dict"])
        #     self.optimizer.load_state_dict(state["optimizer_state_dict"])
        #     if self.scaler is not None and state.get("scaler_state_dict") is not None:
        #         self.scaler.load_state_dict(state["scaler_state_dict"])
        #     start_epoch = state.get("epoch", 0)
        # else:
        #     # Legacy: bare model state dict
        #     self._raw_model().load_state_dict(state)
        #     start_epoch = 0

        start_epoch = self._load_state_dict(ckpt_path, finetune=self.cfg.experiment.finetune)

        if self.is_master:
            logger.info(f"Resumed from checkpoint: {ckpt_path} (epoch {start_epoch})")
        return start_epoch

    # ── Training ───────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch index (0-indexed).

        Returns:
            Tuple ``(avg_loss, top1_accuracy)`` for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loader = self.train_data
        if self.is_master:
            loader = tqdm(
                loader,
                desc=f"Train [{epoch + 1}/{self.max_epochs}]",
                leave=False,
            )

        for img, label in loader:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            use_mix = self.cutmix is not None or self.mixup is not None
            if use_mix:
                img, label, rand_label, lambda_ = self._apply_mixup(img, label)
                rand_label = rand_label.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(img)
                if use_mix:
                    loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1.0 - lambda_)
                else:
                    loss = self.criterion(out, label)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            with torch.no_grad():
                orig_label = label
                total_loss += loss.item()
                correct += torch.eq(out.argmax(-1), orig_label).sum().item()
                total += orig_label.size(0)

            if self.is_master:
                loader.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_data)
        top1 = correct / total
        return avg_loss, top1

    @torch.no_grad()
    def _eval_epoch(self) -> Tuple[float, float]:
        """Run one evaluation pass over the test loader.

        Returns:
            Tuple ``(avg_loss, top1_accuracy)``.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for img, label in self.test_data:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(img)
                loss = self.criterion(out, label)

            total_loss += loss.item()
            correct += torch.eq(out.argmax(-1), label).sum().item()
            total += label.size(0)

        avg_loss = total_loss / len(self.test_data)
        top1 = correct / total
        return avg_loss, top1

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop.

        Optionally resumes from a checkpoint (``cfg.experiment.ckpt_path``).
        Saves the best and last checkpoints to ``<run_dir>/checkpoint/``.
        Logs per-epoch metrics to W&B when ``cfg.wandb`` is enabled.
        Only the master process logs and checkpoints.
        """
        start_epoch = self._resume_checkpoint()
        best_val_acc = 0.0
        n_gpus = torch.cuda.device_count() if self.is_multi_gpus else 1

        for epoch in range(start_epoch, self.max_epochs):
            # ── DistributedSampler epoch shuffle ──────────────────────────────
            if self.is_multi_gpus:
                self.train_data.sampler.set_epoch(epoch)

            # ── Learning rate update ───────────────────────────────────────────
            lr = self._get_lr(epoch)

            # ── Training ──────────────────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(epoch)

            # Synchronise losses across ranks for accurate reporting
            if self.is_multi_gpus:
                train_loss = self._all_gather(train_loss, n_gpus)
                train_acc = self._all_gather(train_acc, n_gpus)

            # ── Evaluation (all ranks, but only master reports) ───────────────
            val_loss, val_acc = self._eval_epoch()

            if self.is_multi_gpus:
                val_loss = self._all_gather(val_loss, n_gpus)
                val_acc = self._all_gather(val_acc, n_gpus)

            if self.is_master:
                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs} | "
                    f"lr={lr:.2e} | "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc * 100:.2f}% | "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}%"
                )

                # ── W&B logging ────────────────────────────────────────────────
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/loss": val_loss,
                            "val/acc": val_acc,
                            "lr": lr,
                        },
                        step=epoch + 1,
                    )

                # ── Save best checkpoint ───────────────────────────────────────
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch + 1, "best")
                    logger.info(f"New best val acc: {val_acc * 100:.2f}%")

        # ── Save final checkpoint ──────────────────────────────────────────────
        self._save_checkpoint(self.max_epochs, "last")
        if self.is_master:
            logger.info(f"Training complete. Best val acc: {best_val_acc * 100:.2f}%")

    def evaluate(self) -> Tuple[float, float]:
        """Load a checkpoint and run a single evaluation pass.

        The checkpoint is read from ``cfg.experiment.ckpt_path``.

        Returns:
            Tuple ``(avg_loss, top1_accuracy)``.
        """
        ckpt_path = self.cfg.experiment.ckpt_path
        if not ckpt_path:
            raise ValueError("cfg.experiment.ckpt_path must be set for evaluation mode.")

        if os.path.isdir(ckpt_path):
            candidate = os.path.join(ckpt_path, f"{self.ckpt_name}_best.pth")
            ckpt_path = candidate if os.path.isfile(candidate) else ckpt_path

        if self.is_master:
            logger.info(f"Loading checkpoint for evaluation: {ckpt_path}")

        self._load_state_dict(ckpt_path)

        val_loss, val_acc = self._eval_epoch()

        if self.is_master:
            logger.info(f"Evaluation results | loss={val_loss:.4f}, top-1 acc={val_acc * 100:.2f}%")
        return val_loss, val_acc
