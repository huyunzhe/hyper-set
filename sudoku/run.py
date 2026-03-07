"""Sudoku task entry point.

Provides :func:`run_task`, called by the root ``main.py`` via Hydra.
Dispatches to training (``mode=train``), evaluation (``mode=eval``), or
visualization (``mode=vis``) based on ``cfg.mode``.
"""

import torch
from loguru import logger
from omegaconf import DictConfig

from sudoku.trainer import SudokuTrainer
from sudoku.utils import create_model, eval_vis_func, load_datasets


def run_training(
    trainer: SudokuTrainer,
):
    """
    Run the training loop.

    Args:
        trainer: SudokuTrainer instance

    Returns:
        Training results dictionary
    """

    # Run training
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    results = trainer.train()

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)

    # Print results summary
    logger.info("=" * 80)
    logger.info("Training Results Summary")
    logger.info("=" * 80)

    for eval_name, eval_results in results.items():
        if eval_results:
            last_result = eval_results[-1]
            correct, total, single_correct, single_total, _ = last_result
            board_acc = 100 * correct / total
            cell_acc = 100 * single_correct / single_total

            logger.info(f"{eval_name}:")
            logger.info(f"  Board Accuracy: {board_acc:.2f}%")
            logger.info(f"  Cell Accuracy: {cell_acc:.2f}%")

    return results


def run_evaluation(
    trainer: SudokuTrainer,
):
    """
    Run evaluation only (load checkpoint and evaluate).

    Args:
        trainer: SudokuTrainer instance
    """
    logger.info("=" * 80)
    logger.info("Running evaluation mode...")
    logger.info("=" * 80)

    board_acc, cell_acc, correct, total, single_correct, single_total = trainer.evaluate()

    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Board Accuracy: {board_acc:.2f}% ({correct}/{total})")
    logger.info(f"Cell Accuracy: {cell_acc:.2f}% ({single_correct}/{single_total})")
    logger.info(f"Model Recurrence Steps: {trainer.model.n_recur}")
    logger.info("=" * 80)


def run_task(cfg: DictConfig, device: torch.device, wandb_run=None):
    """
    Main entry point for the Sudoku task.

    Args:
        cfg: Hydra configuration object
        device: torch.device for computation
        wandb_run: Optional wandb run object for logging
    """
    logger.info("=" * 80)
    logger.info("Sudoku Task")
    logger.info("=" * 80)

    # Load datasets
    train_dataset, test_dataset, train_dataset_ulb = load_datasets(cfg)

    # Create model
    model = create_model(cfg)

    # Watch model with wandb if enabled
    if wandb_run and hasattr(wandb_run, "watch"):
        wandb_run.watch(model, log_freq=100)

    assert cfg.mode in ("train", "eval", "vis"), f"Unknown mode: {cfg.mode}. Please choose 'train', 'eval', or 'vis'."

    # Create trainer
    trainer = SudokuTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        device=device,
        wandb_run=wandb_run,
        train_dataset_ulb=train_dataset_ulb,
        eval_func=eval_vis_func,
    )

    # Run training or evaluation
    if cfg.mode == "eval" or cfg.mode == "vis":
        run_evaluation(trainer)
    elif cfg.mode == "train":
        run_training(trainer)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Please choose 'train', 'eval', or 'vis'.")

    logger.info("Sudoku task completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/task/sudoku.yaml")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_task(cfg, device)
