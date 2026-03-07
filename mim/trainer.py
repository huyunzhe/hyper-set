"""
MIM Trainer: Training and evaluation for MaskGIT-based masked image modeling.
"""

import math
import os
import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from loguru import logger
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the total L2 norm of all gradients in the model.

    Args:
        model: PyTorch model.

    Returns:
        Total gradient norm as a float.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm**0.5


class MIMTrainer:
    """Trainer for MaskGIT-based masked image modeling.

    Follows the same pattern as SudokuTrainer: clean __init__, train/eval methods,
    wandb logging, and checkpoint management.

    Args:
        vit: The MaskTransformer model to train.
        ae: Pretrained VQGAN autoencoder (frozen).
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        cfg: Hydra configuration object.
        device: torch.device for computation.
        wandb_run: Optional wandb run object for logging.
        is_multi_gpus: Whether running in multi-GPU DDP mode.
        is_master: Whether this is the master process.
    """

    def __init__(
        self,
        vit: nn.Module,
        ae: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
        wandb_run=None,
        is_multi_gpus: bool = False,
        is_master: bool = True,
    ):
        self.cfg = cfg
        self.device = device
        self.wandb = wandb_run
        self.is_multi_gpus = is_multi_gpus
        self.is_master = is_master

        # Data
        self.train_data = train_loader
        self.test_data = test_loader

        # Models
        self.ae = ae
        self.codebook_size = ae.n_embed
        self.patch_size = cfg.data.img_size // 2 ** (ae.encoder.num_resolutions - 1)

        # Training config
        self.max_epochs = cfg.training.epochs
        self.lr = cfg.training.lr
        self.grad_cum = cfg.training.grad_cum
        self.grad_clip = cfg.training.grad_clip
        self.log_iter = cfg.training.log_iter
        self.eval_interval = cfg.training.eval_interval
        self.mask_value = cfg.model.mask_value
        self.n_recur = cfg.model.n_recur
        self.img_size = cfg.data.img_size

        # Sampling config
        self.drop_label = cfg.sampling.drop_label
        self.cfg_w = cfg.sampling.cfg_w
        self.sm_temp = cfg.sampling.sm_temp
        self.r_temp = cfg.sampling.r_temp
        self.sched_mode = cfg.sampling.sched_mode
        self.step = cfg.sampling.step

        # Iteration counters
        self.iter = 0
        self.global_epoch = 0

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda")

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            vit.parameters(),
            lr=self.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )

        # Resume from checkpoint
        if cfg.experiment.ckpt_path:
            self._load_checkpoint(cfg.experiment.ckpt_path, vit)

        # Move model to device and wrap in DDP
        vit = vit.to(device)
        if is_multi_gpus:
            self.vit = DDP(vit, device_ids=[device])
        else:
            self.vit = vit

        # Checkpoint directory
        if cfg.get("run_dir"):
            self.ckpt_path = os.path.join(cfg.run_dir, "checkpoint")
        else:
            self.ckpt_path = os.path.join("outputs", "mim_checkpoint")
        if is_master:
            os.makedirs(self.ckpt_path, exist_ok=True)

        if is_master:
            logger.info(f"MIMTrainer initialized: codebook_size={self.codebook_size}, patch_size={self.patch_size}")
            logger.info(f"Training: {self.max_epochs} epochs, lr={self.lr}, batch_size={cfg.training.batch_size}")

    def _load_checkpoint(self, ckpt_path: str, model: nn.Module) -> None:
        """Load model, optimizer, and scaler state from a checkpoint.

        Args:
            ckpt_path: Path to checkpoint file or directory.
            model: The model to load weights into.
        """
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "current.pth")
            assert os.path.isfile(ckpt_path), f"No checkpoint found at {ckpt_path}"

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if self.is_master:
            logger.info(f"Loading checkpoint from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.iter = checkpoint.get("iter", 0)
        self.global_epoch = checkpoint.get("global_epoch", 0)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.is_master:
            logger.info(f"Resumed from epoch {self.global_epoch}, iter {self.iter}")

    def _save_checkpoint(self, filename: str = "current") -> None:
        """Save model, optimizer, and scaler state to checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        if not self.is_master:
            return

        model_state = self.vit.module.state_dict() if self.is_multi_gpus else self.vit.state_dict()
        if filename:
            save_path = os.path.join(self.ckpt_path, f"{self.cfg.experiment.ckpt_name}_{filename}.pth")
        else:
            save_path = os.path.join(self.ckpt_path, f"{self.cfg.experiment.ckpt_name}.pth")
        torch.save(
            {
                "iter": self.iter,
                "global_epoch": self.global_epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            save_path,
        )
        logger.info(f"Saved checkpoint: {save_path}")

    def _get_mask_code(
        self,
        code: torch.LongTensor,
        mode: str = "linear",
        value: int = 1024,
        codebook_size: int = 1024,
        masked_ratio: float = 0.4,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """Randomly mask a portion of image tokens.

        Args:
            code: Unmasked token codes of shape (B, H, W).
            mode: Masking schedule ('linear', 'square', 'cosine', 'arccos').
            value: Token value to use for masked positions. If < 0, use random tokens.
            codebook_size: Size of the codebook for random token replacement.
            masked_ratio: Maximum fraction of tokens to mask.

        Returns:
            Tuple of (masked_code, mask) where mask is True for masked positions.
        """
        if self.cfg.mode == "train":
            r = torch.rand(code.size(0))
        elif self.cfg.mode == "eval_recon":
            r = torch.ones(code.size(0))
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
        if mode == "linear":
            val_to_mask = r
        elif mode == "square":
            val_to_mask = r**2
        elif mode == "cosine":
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            raise ValueError(f"Unknown masking mode: {mode}")

        val_to_mask *= masked_ratio

        mask_code = code.detach().clone()
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        if value > 0:
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        return mask_code, mask

    def adap_sche(self, step: int, mode: str = "arccos", leave: bool = False):
        """Create a token prediction scheduler for iterative decoding.

        Args:
            step: Number of prediction steps during inference.
            mode: Schedule shape ('root', 'linear', 'square', 'cosine', 'arccos').
            leave: Whether to leave the tqdm progress bar after completion.

        Returns:
            Iterator of int tensors representing tokens to predict per step.
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":
            val_to_mask = 1 - (r**0.5)
        elif mode == "linear":
            val_to_mask = 1 - r
        elif mode == "square":
            val_to_mask = 1 - (r**2)
        elif mode == "cosine":
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            raise ValueError(f"Unknown schedule mode: {mode}")

        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size**2)
        sche = sche.round()
        sche[sche == 0] = 1
        sche[-1] += (self.patch_size**2) - sche.sum()
        if self.cfg.mode == "train" and self.is_master:
            return tqdm(sche.int(), leave=leave)
        else:
            return sche.int()

    def _train_one_epoch(self, epoch: int) -> float:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.vit.train()
        cum_loss = 0.0
        window_loss = deque(maxlen=self.grad_cum)  # type: ignore
        bar = tqdm(self.train_data, leave=False, desc=f"Epoch {epoch}") if self.is_master else self.train_data
        n = len(self.train_data)
        masked_ratio = self.cfg.evaluation.get("masked_ratio", 0.4)

        for it, (x, y) in enumerate(bar):
            x = x.to(self.device)
            y = y.to(self.device)
            x = 2 * x - 1  # Normalize [0,1] -> [-1,1] for VQGAN

            drop_label = torch.empty(y.size()).uniform_(0, 1) < self.drop_label

            # VQGAN encode to tokens (frozen)
            with torch.no_grad():
                _, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask tokens
            masked_code, mask = self._get_mask_code(
                code,
                value=self.mask_value,
                codebook_size=self.codebook_size,
                mode="linear",
                masked_ratio=masked_ratio,
            )

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda"):
                pred = self.vit(
                    masked_code,
                    y=y,
                    drop_label=drop_label,
                    n_recur=self.n_recur,
                )
                # Cross-entropy on masked tokens only
                loss = (
                    self.criterion(
                        pred.view(-1, self.codebook_size)[mask.reshape(-1)],
                        code[mask].view(-1),
                    )
                    / self.grad_cum
                )

            # Gradient accumulation
            update_grad = self.iter % self.grad_cum == self.grad_cum - 1
            if update_grad:
                self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()

            if update_grad:
                self.scaler.unscale_(self.optimizer)
                grad_norm = compute_grad_norm(self.vit)
                nn.utils.clip_grad_norm_(self.vit.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())

            # Logging
            if update_grad and self.is_master and self.wandb:
                self.wandb.log(
                    {
                        "train/grad_norm": grad_norm,
                        "train/loss": np.array(window_loss).sum(),
                    },
                    step=self.iter,
                )

            # Periodic visualization and checkpoint
            if self.iter % self.log_iter == 0 and self.is_master:
                self._log_reconstruction(x, code, pred, masked_code, mask)
                self._save_checkpoint("current")

            self.iter += 1

        return cum_loss / n

    def _log_reconstruction(
        self,
        x: torch.Tensor,
        code: torch.Tensor,
        pred: torch.Tensor,
        masked_code: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Log reconstruction images to wandb.

        Args:
            x: Original images (B, C, H, W).
            code: Ground-truth token codes (B, pH, pW).
            pred: Predicted logits (B, pH*pW, codebook_size).
            masked_code: Masked token codes (B, pH, pW).
            mask: Boolean mask (B, pH, pW).
        """
        if not self.wandb:
            return

        with torch.no_grad():
            # Sample from masked code
            gen_sample = self.sample(
                init_code=masked_code[:10],
                nb_sample=10,
                sm_temp=1.0,
                r_temp=0.0,
                w=0.0,
                step=12,
            )[0]
            gen_grid = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)

            # Reconstruction
            unmasked_code = torch.softmax(pred, -1).max(-1)[1]
            reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
            reco_grid = vutils.make_grid(reco_sample.data, nrow=10, padding=2, normalize=True)

            self.wandb.log(
                {
                    "images/sampling_from_partial": self.wandb.Image(gen_grid.permute(1, 2, 0).cpu().numpy()),
                    "images/reconstruction": self.wandb.Image(reco_grid.permute(1, 2, 0).cpu().numpy()),
                },
                step=self.iter,
            )

    def train(self) -> None:
        """Main training loop. Runs for cfg.training.epochs epochs."""
        if self.is_master:
            logger.info("Starting training...")

        start = time.time()

        for epoch in range(self.global_epoch, self.max_epochs):
            if self.is_multi_gpus:
                self.train_data.sampler.set_epoch(epoch)

            train_loss = self._train_one_epoch(epoch)

            if self.is_multi_gpus:
                train_loss = self._all_gather(train_loss, torch.cuda.device_count())

            # Periodic checkpoint
            if (epoch + 1) % self.eval_interval == 0 and self.is_master:
                self._save_checkpoint(f"epoch_{self.global_epoch + 1:03d}")

            # Logging
            clock = time.time() - start
            if self.is_master:
                if self.wandb:
                    self.wandb.log(
                        {
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/epoch_loss": train_loss,
                        },
                        step=self.global_epoch,
                    )

                logger.info(
                    f"Epoch {self.global_epoch}, Iter {self.iter}, "
                    f"Loss {train_loss:.4f}, "
                    f"Time: {clock // 3600:.0f}h {(clock % 3600) // 60:.0f}min {clock % 60:.2f}s"
                )

            self.global_epoch += 1

        # Save last checkpoint
        self._save_checkpoint("last")
        if self.is_master:
            logger.info("Training complete!")

    def eval(self) -> dict:
        """Evaluate the model using FID metrics (requires Metrics module).

        Returns:
            Dictionary of evaluation metrics.
        """
        self.vit.eval()
        if self.is_master:
            logger.info(
                f"Evaluation: sched={self.sched_mode}, steps={self.step}, "
                f"sm_temp={self.sm_temp}, cfg_w={self.cfg_w}, r_temp={self.r_temp}"
            )

        from mim.metrics.sample_and_eval import SampleAndEval

        sae = SampleAndEval(device=self.device, num_images=self.cfg.evaluation.num_eval_images)
        metrics = sae.compute_and_log_metrics(self)

        self.vit.train()
        return metrics

    def eval_recon(self) -> None:
        """Evaluate reconstruction quality with MSE, PSNR, SSIM, LPIPS, and FID."""
        import torchmetrics

        self.vit.eval()
        if self.is_master:
            logger.info("Starting reconstruction evaluation...")

        masked_ratio = self.cfg.evaluation.masked_ratio

        # Initialize metrics
        mse_metric = torchmetrics.MeanSquaredError().to(self.device)
        psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(self.device)
        multi_ssim_metric = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(
            self.device
        )
        lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(self.device)

        bar = tqdm(self.test_data, leave=False) if self.is_master else self.test_data
        MSE = PSNR = SSIM_PARTIAL = MULTI_SSIM = LPIPS = 0.0
        NUM = 0

        def normalize(x):
            """Min-max normalize to [0, 1]."""
            _max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            _min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            return (x - _min) / (_max - _min + 1e-8)

        with torch.no_grad():
            for x, y in bar:
                x = x.to(self.device)
                x = 2 * x - 1  # [0,1] -> [-1,1]

                _, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

                masked_code, mask = self._get_mask_code(
                    code,
                    value=self.mask_value,
                    codebook_size=self.codebook_size,
                    mode="linear",
                    masked_ratio=masked_ratio,
                )

                sample_recon, _, _, _ = self.sample(
                    init_code=masked_code,
                    nb_sample=masked_code.shape[0],
                    sched_mode="arccos",
                    sm_temp=1.0,
                    r_temp=0.0,
                    w=0.0,
                    step=24,
                )

                x_norm = normalize(x)
                sample_norm = normalize(sample_recon)
                assert x_norm.max() <= 1.0 and x_norm.min() >= 0.0, (x_norm.max(), x_norm.min())
                assert sample_norm.max() <= 1.0 and sample_norm.min() >= 0.0, (sample_norm.max(), sample_norm.min())

                mse = mse_metric(x_norm, sample_norm)
                psnr = psnr_metric(x_norm, sample_norm)
                ssim = ssim_metric(x_norm, sample_norm)
                multi_ssim = multi_ssim_metric(x_norm, sample_norm)
                lpips_score = lpips_metric(x_norm, sample_norm)

                MSE += mse * x.shape[0]
                PSNR += psnr * x.shape[0]
                SSIM_PARTIAL += ssim * x.shape[0]
                MULTI_SSIM += multi_ssim * x.shape[0]
                LPIPS += lpips_score * x.shape[0]

                fid_metric.update(x_norm, real=True)
                fid_metric.update(sample_norm, real=False)
                NUM += x.shape[0]

        fid = fid_metric.compute()

        results = {
            "NUM": NUM,
            "MSE": (MSE / NUM),
            "PSNR": (PSNR / NUM),
            "SSIM": (SSIM_PARTIAL / NUM),
            "Multi-SSIM": (MULTI_SSIM / NUM),
            "LPIPS": (LPIPS / NUM),
            "FID": fid.item(),
        }

        if self.is_master:
            for k, v in results.items():
                logger.info(f"{k}: {v}")
            if self.wandb:
                self.wandb.log({f"eval_recon/{k}": v for k, v in results.items()})

        self.vit.train()

    def reco(
        self,
        x=None,
        code=None,
        masked_code=None,
        unmasked_code=None,
        mask=None,
    ) -> torch.Tensor:
        """Visualize reconstruction: original, masked, predicted, and combined images.

        Args:
            x: Original images (B, C, H, W).
            code: Ground-truth codes (B, pH, pW).
            masked_code: Masked codes (B, pH, pW).
            unmasked_code: Predicted codes (B, pH, pW).
            mask: Boolean mask (B, pH, pW).

        Returns:
            Concatenated visualization tensor.
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                gt_recon = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size - 1))
                if mask is not None:
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    gt_recon_masked = gt_recon * (
                        1 - F.interpolate(mask, (self.img_size, self.img_size)).to(self.device)
                    )
                    l_visual.append(gt_recon_masked)

            if masked_code is not None:
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                masked_recon = self.ae.decode_code(torch.clamp(masked_code, 0, self.codebook_size - 1))
                l_visual.append(masked_recon)

            if unmasked_code is not None:
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                predicted_recon = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size - 1))
                l_visual.append(predicted_recon)
                l_visual.append(
                    gt_recon_masked
                    + predicted_recon * F.interpolate(mask, (self.img_size, self.img_size)).to(self.device)
                )

        return torch.cat(l_visual, dim=0)

    def sample(
        self,
        init_code=None,
        nb_sample: int = 50,
        labels=None,
        sm_temp: float = 1.0,
        w: float = 3.0,
        randomize: str = "linear",
        r_temp: float = 4.5,
        sched_mode: str = "arccos",
        step: int = 12,
    ):
        """Generate samples using iterative parallel decoding (MaskGIT).

        Args:
            init_code: Starting code (B, pH, pW). If None, start from all masked.
            nb_sample: Number of samples to generate.
            labels: Class labels for conditional generation.
            sm_temp: Softmax temperature.
            w: Classifier-free guidance weight.
            randomize: Noise schedule ('linear', 'warm_up', 'random', 'no').
            r_temp: Gumbel noise temperature.
            sched_mode: Decoding schedule ('root', 'linear', 'square', 'cosine', 'arccos').
            step: Number of decoding steps.

        Returns:
            Tuple of (decoded_images, intermediate_codes, intermediate_masks, final_code).
        """
        self.vit.eval()
        l_codes = []
        l_mask = []

        with torch.no_grad():
            if labels is None:
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                # labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * (nb_sample // 10)
                labels = torch.zeros(nb_sample, device=self.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.device)

            if init_code is not None:
                code = init_code
                mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size**2)
            else:
                if self.mask_value < 0:
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(
                        self.device
                    )
                else:
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.mask_value).to(self.device)
                mask = torch.ones(nb_sample, self.patch_size**2).to(self.device)

            if isinstance(sched_mode, str):
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:
                scheduler = sched_mode

            for indice, t in enumerate(scheduler):
                if mask.sum() < t:
                    t = int(mask.sum().item())
                if mask.sum() == 0:
                    break

                if w != 0:
                    logit = self.vit(
                        torch.cat([code.clone(), code.clone()], dim=0),
                        torch.cat([labels, labels], dim=0),
                        torch.cat([~drop, drop], dim=0),
                        n_recur=self.n_recur,
                    )
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    _w = w * (indice / (len(scheduler) - 1))
                    logit = (1 + _w) * logit_c - _w * logit_u
                else:
                    logit = self.vit(code.clone(), labels, drop_label=~drop, n_recur=self.n_recur)

                if torch.isnan(logit).any():
                    if self.is_master:
                        logger.warning("NaN detected in logits during sampling")
                    break

                prob = torch.softmax(logit * sm_temp, -1)
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size**2, 1))

                if randomize == "linear":
                    ratio = indice / (len(scheduler) - 1)
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size**2)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device)
                elif randomize == "warm_up":
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":
                    conf = torch.rand_like(conf)

                conf[~mask.bool()] = -math.inf

                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                conf_mask = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (
                    mask.view(nb_sample, self.patch_size, self.patch_size).float()
                    * conf_mask.view(nb_sample, self.patch_size, self.patch_size).float()
                ).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0

                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            _code = torch.clamp(code, 0, self.codebook_size - 1)
            x = self.ae.decode_code(_code)

        self.vit.train()
        return x, l_codes, l_mask, _code

    @staticmethod
    def _all_gather(obj, gpus: int, reduce: str = "mean"):
        """Gather values from all GPUs and reduce.

        Args:
            obj: Value to gather.
            gpus: Number of GPUs.
            reduce: Reduction method ('mean', 'sum', 'none').

        Returns:
            Reduced value.
        """
        tensor_list = [torch.zeros(1) for _ in range(gpus)]
        dist.all_gather_object(tensor_list, obj)
        result = torch.FloatTensor(tensor_list)
        if reduce == "mean":
            return result.mean()
        elif reduce == "sum":
            return result.sum()
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")
