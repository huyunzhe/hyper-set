"""
Main entry point for Hyper-SET experiments.

This is the unified entry point for all tasks in the Hyper-SET framework.
It uses Hydra for configuration management, allowing for flexible
configuration through YAML files and command-line overrides.

Usage:
    # Train with default config (sudoku task)
    python main.py

    # Train sudoku with custom hyperparameters
    python main.py task=sudoku training.epochs=300 training.batch_size=32

    # Evaluate a trained model
    python main.py evaluation.evaluate=true

    # Enable wandb logging
    python main.py wandb=true experiment.comment=my_experiment

    # Override config with command line
    python main.py model.n_recur=64 training.learning_rate=1e-3

Configuration Structure:
    - configs/config.yaml: Base configuration for all tasks
    - configs/task/<task_name>.yaml: Task-specific configurations
    - Each task folder (e.g., sudoku/) contains:
        - trainer.py: Task-specific trainer implementation
        - run.py: Task-specific entry point (imported by this main.py)

Author: Yunzhe Hu
"""

import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from utils.training_utils import setup_device, setup_seed, setup_wandb


def import_task_main(task_name: str):
    """
    Dynamically import the main function from the task-specific module.

    Args:
        task_name: Name of the task (e.g., 'sudoku', 'image_classification')

    Returns:
        The main function from the task module

    Raises:
        ImportError: If the task module or main function cannot be found
    """
    try:
        # Import the task-specific main module
        task_module = __import__(f"{task_name}.run", fromlist=["run"])

        if not hasattr(task_module, "run_task"):
            raise AttributeError(f"Task module {task_name}.run must define a 'run_task' function")

        return task_module.run_task

    except ImportError as e:
        logger.error(f"Failed to import task '{task_name}': {e}")
        logger.error(f"Make sure {task_name}/main.py exists and defines a 'run_task' function")
        raise


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Detect multi-GPU environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0

    # Print configuration for debugging (master only)
    if is_master:
        logger.info("=" * 80)
        logger.info("Configuration")
        logger.info("=" * 80 + "\n")
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info("=" * 80)

    # Set up device
    device = setup_device()

    # Set up random seed
    setup_seed(cfg.seed)

    # Set up wandb (master process only to avoid duplicate logging)
    wandb_run = setup_wandb(cfg) if is_master else None

    # Get task name
    task_name = cfg.task_name
    if is_master:
        logger.info(f"Running task: {task_name}")

    # Import and run task-specific main function
    run_task = import_task_main(task_name)

    # Run the task with config and wandb
    run_task(cfg, device, wandb_run)

    if is_master:
        logger.info(f"Task '{task_name}' completed successfully!")

    # Clean up wandb
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
