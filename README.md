<div align="center" style="font-family: charter;">
    <h1>Hyper-SET: Designing Transformers via Hyperspherical Energy Minimization</h1>
</div>

<p align="center">
  <a href="https://hyper-set.github.io/"><img src="https://img.shields.io/badge/🌐%20Project-Page-success.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2502.11646"><img src="https://img.shields.io/badge/arXiv-2502.11646-b31b1b.svg" alt="arXiv"></a>
  <a href="https://openreview.net/forum?id=FinhjyDgYA"><img src="https://img.shields.io/badge/OpenReview-ICLR%202026-blue.svg" alt="OpenReview"></a>
  <a href="https://huggingface.co/Yunzhe/Hyper-SET/tree/main"><img src="https://img.shields.io/badge/🤗%20Models-Hugging%20Face-yellow.svg" alt="Models"></a>
  <a href="https://huggingface.co/Yunzhe/datasets"><img src="https://img.shields.io/badge/🤗%20Data-Hugging%20Face-yellow.svg" alt="Data"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

<div align="center" style="font-family: charter; max-width: 1000px; margin: 0 auto;">
    <blockquote style="border-left: 4px solid #ccc; padding: 15px 20px; background-color: #f9f9f9; font-style: italic;">
        <h3 style="margin: 0; white-space: nowrap; font-size: 1.15em;">
            Can we find or design a function prior that induces a model interpretable by construction?
        </h3>
    </blockquote>
</div>


We think the answer is yes, at least for a family of Transformers. We provide a principled framework for viewing and designing Transformers through the lens of **hyperspherical energy minimization**. A family of self-attention and feedforward layers with structural bias, RMSNorm, and skip connections all emerge from the fundamental objective: **maximum likelihood**. The result is a compact, interpretable, and competitive Transformer—designed top-down, not engineered bottom-up.

**Contributions & Highlights**

- **Principled architecture.** All components derived from first principles—no heuristics, no ad hoc choices.
- **Interpretable by construction.** Token dynamics are a verifiable energy-based dynamical system: energy descent, effective rank, and inter-token angles are all empirically confirmed.
- **Extensible framework.** Not one model—a design principle. Naturally generalizes to linear attention, sigmoid attention, and gated feedforward. 

## Installation

We use **Python 3.9** and **PyTorch 2.5.1**.

### 1. Clone the repository

```bash
git clone https://github.com/huyunzhe/hyper-set.git
cd hyper-set
```

### 2. Create a virtual environment

```bash
conda create -n hyper-set python=3.9 -y
conda activate hyper-set
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
hyper-set/
├── main.py                      # Unified entry point (Hydra)
├── configs/
│   ├── config.yaml              # Base configuration for all tasks
│   └── task/
│       ├── sudoku.yaml          # Sudoku task overrides
│       ├── ic.yaml              # Image classification task overrides
│       └── mim.yaml             # Masked image modeling task overrides
├── layers/                      # Shared model building blocks
│   ├── hyperset.py              # HyperSET layers
│   ├── transformer.py           # Standard transformer layers
│   ├── crate.py                 # CRATE / CRATE-T layers
│   ├── et.py                    # Energy Transformer layers
│   ├── activations.py           # Activation functions (phi / d_phi) for linear attention and gated feedforward
│   ├── pos_embed.py             # Positional embedding utilities
│   └── components/              # Shared attention, feedforward, norm modules
├── sudoku/                      # Sudoku constraint-satisfaction task
│   ├── run.py                   # Task entry point
│   ├── trainer.py               # Sudoku trainer
│   ├── dataset.py               # Dataset loading
│   └── models.py                # Sudoku model definitions
├── ic/                          # Image classification task
│   ├── run.py                   # Task entry point (DDP-aware)
│   ├── trainer.py               # IC trainer
│   ├── models.py                # ViT model variants
│   ├── dataset.py               # Dataset loading (CIFAR, SVHN, ImageNet)
│   └── utils.py                 # Model factory and loss utilities
├── mim/                         # Masked image modeling task (MaskGIT)
│   ├── run.py                   # Task entry point
│   ├── trainer.py               # MIM trainer
│   └── models.py                # MaskTransformer model variants
├── utils/
│   ├── training_utils.py        # Device, seed, W&B setup helpers
│   └── metric_utils.py          # Geometry metrics (effective rank, angle)
├── scripts/                     # Training and evaluation shell scripts
│   ├── sudoku/
│   ├── ic/
│   └── mim/
├── notebooks/                   # Visualization notebooks
├── pretrained_weights/          # Pretrained model download utilities
└── requirements.txt
```

## Getting Started

### Configuration

All hyperparameters are managed with [Hydra](https://hydra.cc/docs/intro/). Configure via:

- **YAML files** – `configs/config.yaml` (base), `configs/task/<task>.yaml` (per-task defaults)
- **Command-line overrides** – `python main.py training.epochs=500 model.n_recur=24`

> [!NOTE]
> To use ImageNet-1k/100 datasets, please specify `data.in100_path` and/or `data.in1k_path` in `./configs/config.yaml` or they will be automatically downloaded from [HuggingFace repository](https://huggingface.co/Yunzhe/datasets) when you run the script for the first time.
>
> For ImageNet-100 subset, you can first download the full ImageNet-1k on the official [ImageNet website](http://www.image-net.org/download), and then extract folders with ID list from [here](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt).
>
> All other datasets (CIFAR-10/100, Sudoku) are downloaded automatically to `./data/`.

### Pretrained Weights

To evaluate or fine-tune our pretrained checkpoints, first download them from Hugging Face:

```bash
python pretrained_weights/download.py
```

### Basic Usage

```bash
# Train with default configuration
python main.py task=sudoku

# Train with customized hyperparameters
# Model architecture
python main.py model.type=hyper-set model.n_layer=1 model.n_recur=12 model.n_embed=512

# Training settings
python main.py training.epochs=300 training.learning_rate=1e-3 training.batch_size=128

# Data settings
python main.py data.dataset=in1k

# Enable W&B logging
python main.py task=sudoku wandb=true experiment.comment=my_experiment

# Evaluate a pretrained model
python main.py task=ic mode=eval experiment.ckpt_path=/path/to/checkpoint.pth
```

---

### Task 1: Solving Sudoku

The dataset is automatically downloaded to `./data/` on the first run.

```bash
# Training
bash scripts/sudoku/train_sudoku.sh

# Evaluation
bash scripts/sudoku/eval_sudoku.sh
```

**Visualizing energy trajectories**

Set `mode=vis` in the evaluation script, or run:

```bash
bash scripts/sudoku/vis_sudoku.sh
```

This saves per-recurrence statistics (energy, effective rank, average angle) of all evaluation samples for visualization. See [`notebooks/sudoku/vis_stats_trajectory.ipynb`](notebooks/sudoku/vis_stats_trajectory.ipynb) for an example.

---

### Task 2: Image Classification

<details>
 <summary>Supported datasets</summary>

| Key     | Dataset      | Resolution |
| ------- | ------------ | ---------- |
| `c10`   | CIFAR-10     | 32×32      |
| `c100`  | CIFAR-100    | 32×32      |
| `svhn`  | SVHN         | 32×32      |
| `in100` | ImageNet-100 | 224×224    |
| `in1k`  | ImageNet-1k  | 224×224    |

</details>

<details>
 <summary>Supported model variants</summary>

| `model.type`         | Description                                      |
| -------------------- | ------------------------------------------------ |
| `hyper-set`          | Default Hyper-SET model (learnable step sizes)   |
| `hyper-set-lora`     | Hyper-SET + depth-wise LoRA                      |
| `hyper-set-basic`    | Hyper-SET with extra RMSNorm (CIFAR scratch)     |
| `hyper-set-alt-attn` | Hyper-SET with configurable attention function   |
| `hyper-set-alt-ff`   | Hyper-SET with configurable feedforward function |
| `hyper-set-ss`       | Hyper-SET fixed step-size ablation               |
| `transformer`        | Standard ViT baseline                            |
| `crate`              | CRATE baseline                                   |
| `crate-T`            | CRATE-T baseline                                 |
| `et`                 | Energy Transformer baseline                      |

</details>

**Training / Evaluation**

```bash
# Single-GPU training
bash scripts/ic/train_ic.sh

# Multi-GPU DDP training
bash scripts/ic/train_ic_multi-gpu.sh

# Fine-tuning from a pretrained checkpoint
bash scripts/ic/finetune_ic.sh

# Evaluation
bash scripts/ic/eval_ic.sh
```

**Training with depth-wise LoRA**

```bash
bash scripts/ic/train_ic_lora.sh
```

**Alternative designs (ablations)**

```bash
bash scripts/ic/train_ic_alternative_attention.sh   # Linear / Sigmoid attention
bash scripts/ic/train_ic_alternative_feedforward.sh # Gated / Softmax feedforward
bash scripts/ic/train_ic_fix_step_size.sh           # Fixed step-size ablation
```

---

### Task 3: Masked Image Modeling

First, download the pretrained VQGAN tokenizer:

```bash
python mim/download_vqgan.py
```

Then run training or evaluation:

```bash
# Single-GPU training
bash scripts/mim/train_mim.sh

# Multi-GPU DDP training
bash scripts/mim/train_mim_multi-gpu.sh

# Evaluation
bash scripts/mim/eval_mim.sh
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

We thank the authors of [recurrent_transformer](https://github.com/azreasoners/recurrent_transformer), [CMC](https://github.com/HobbitLong/CMC), and [Maskgit-pytorch](https://github.com/valeoai/Maskgit-pytorch) for open-sourcing their code and datasets, which we built upon for our experiments.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{
    hu2026hyperset,
    title={Hyper-{SET}: Designing Transformers via Hyperspherical Energy Minimization},
    author={Yunzhe Hu and Difan Zou and Dong Xu},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=FinhjyDgYA}
}
```
