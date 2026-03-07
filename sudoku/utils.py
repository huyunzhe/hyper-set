import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from sudoku.dataset import Sudoku_Dataset, Sudoku_Dataset_Palm, Sudoku_Dataset_SATNet
from sudoku.models import Model, ModelConfig


def load_datasets(cfg: DictConfig):
    """
    Load Sudoku datasets based on configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        tuple: (train_dataset, test_dataset, train_dataset_unlabeled)
    """
    dataset_name = cfg.data.dataset
    n_train = cfg.data.n_train
    n_test = cfg.data.n_test
    seed = cfg.seed

    logger.info(f"Loading dataset: {dataset_name}")

    train_dataset_ulb = None

    if dataset_name == "70k":
        # Load 70k dataset
        input_path = "../data/easy_130k_given.p"
        label_path = "../data/easy_130k_solved.p"

        full_dataset = Sudoku_Dataset(input_path, label_path, n_train + n_test, seed)

        # Split into train and test
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_test])

    elif dataset_name == "satnet":
        # Load SATNet dataset
        full_dataset = Sudoku_Dataset_SATNet()
        indices = list(range(len(full_dataset)))

        # Use same test set as SATNet paper for comparison
        test_dataset = torch.utils.data.Subset(full_dataset, indices[-1000:])
        train_dataset = torch.utils.data.Subset(full_dataset, indices[: min(9000, n_train)])

    elif dataset_name == "palm":
        # Load PALM dataset
        train_dataset = Sudoku_Dataset_Palm(segment="train", limit=n_train, seed=seed)
        test_dataset = Sudoku_Dataset_Palm(segment="test", limit=n_test, seed=seed)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Handle semi-supervised learning setup
    label_size = cfg.data.label_size
    batch_size = cfg.training.batch_size

    if label_size < batch_size:
        # Split training data into labeled and unlabeled
        n_train_lb = int(n_train * (label_size / batch_size))
        n_train_ulb = n_train - n_train_lb

        logger.info(f"Semi-supervised setup: {n_train_lb} labeled, {n_train_ulb} unlabeled")

        indices = list(range(len(train_dataset)))
        train_dataset_ulb = torch.utils.data.Subset(train_dataset, indices[n_train_lb:])
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:n_train_lb])
    else:
        logger.info(f"Fully supervised setup: {len(train_dataset)} labeled samples")

    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    if train_dataset_ulb:
        logger.info(f"Additional {len(train_dataset_ulb)} unlabeled samples")

    return train_dataset, test_dataset, train_dataset_ulb


def create_model(cfg: DictConfig) -> torch.nn.Module:
    """
    Create and initialize the model for Sudoku solving.

    Args:
        cfg: Hydra configuration object

    Returns:
        Initialized model
    """
    logger.info("Creating model...")

    # Create model configuration
    model_cfg = ModelConfig(
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        time_emb=cfg.model.time_emb,
        multiplier=cfg.model.multiplier,
        num_classes=cfg.model.num_classes,
        n_recur=cfg.model.n_recur,
        model_type=cfg.model.type,
        resid_pdrop=cfg.training.dropout,
        attn_pdrop=cfg.training.dropout,
        adaptive_mode=cfg.model.adaptive_mode,
        phi_func=cfg.model.get("phi_func", None),
        pos_emb=cfg.model.pos_emb,
        input_cond=cfg.model.input_cond,
    )

    # Create model
    model = Model(model_cfg)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created: {cfg.model.type}")
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")
    logger.info(
        f"Architecture: {cfg.model.n_layer} layers, "
        f"{cfg.model.n_head} heads, "
        f"{cfg.model.n_embd} embedding dim, "
        f"{cfg.model.n_recur} recurrence steps"
    )

    return model


def eval_vis_func(model, testLoader, return_stats=False):
    """
    Args:
        model: a Pytorch model
        testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    device = next(model.parameters()).device
    # set up testing mode
    model.eval()
    # check if total prediction is correct
    correct = total = 0
    # check if each single prediction is correct
    singleCorrect = singleTotal = 0
    # save the attenetion for the 1st data instance
    energy_traj_total = []
    energy_attn_traj_total = []
    energy_ff_traj_total = []
    effective_rank_total = []
    average_angle_total = []
    rank_total = []
    with torch.no_grad():
        pbar = tqdm(testLoader, total=len(testLoader))
        for data, target in pbar:
            output = model(data.to(device), return_stats=return_stats)
            if isinstance(output, tuple):
                if return_stats:
                    energy_traj = output[2]
                    energy_attn_traj = output[3]
                    energy_ff_traj = output[4]
                    effective_rank_traj = output[5]
                    average_angle_traj = output[6]
                    rank_traj = output[7]

                    energy_traj_total.append(energy_traj)
                    energy_attn_traj_total.append(energy_attn_traj)
                    energy_ff_traj_total.append(energy_ff_traj)
                    effective_rank_total.append(effective_rank_traj)
                    average_angle_total.append(average_angle_traj)
                    rank_total.append(rank_traj)

                output = output[0]
            try:
                if target.shape == output.shape[:-1]:
                    pred = output.argmax(dim=-1)  # get the index of the max value
                elif target.shape == output.shape:
                    pred = (output >= 0).int()
            except Exception as e:
                logger.error(
                    f"Error processing batch with output shape {output.shape} and target shape {target.shape}: {e}"
                )
            target = target.to(device).view_as(pred)
            correctionMatrix = torch.logical_or(target.int() == pred.int(), target < 0).view(target.shape[0], -1)
            correct += correctionMatrix.all(1).sum().item()
            total += target.shape[0]
            singleCorrect += correctionMatrix[target >= 0].sum().item()
            singleTotal += (target >= 0).sum().item()
    if return_stats:
        return (
            correct,
            total,
            singleCorrect,
            singleTotal,
            torch.cat(energy_traj_total, dim=0),
            torch.cat(energy_attn_traj_total, dim=0),
            torch.cat(energy_ff_traj_total, dim=0),
            torch.cat(effective_rank_total, dim=0),
            torch.cat(average_angle_total, dim=0),
            torch.cat(rank_total, dim=0),
        )
    else:
        return correct, total, singleCorrect, singleTotal
