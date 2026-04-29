"""
Configuration for the standalone metasurface augmentation prototype.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    num_samples: int = 5000
    train_split: float = 0.7
    val_split: float = 0.15
    num_wavelengths: int = 20
    num_channels: int = 6
    seed: int = 42

    # Training
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    masked_loss_weight: float = 1.0
    full_loss_weight: float = 0.2

    # Model
    hidden_dim: int = 128
    dropout: float = 0.1

    # Output
    output_dir: str = "experiments/metasurface_aug_test/results"
    checkpoint_name: str = "reconstruct_model.pt"
    summary_name: str = "summary.json"
