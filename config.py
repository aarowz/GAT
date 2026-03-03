"""
Configuration file for GAT-Net training and model parameters.

Modify these values to adjust model architecture, training hyperparameters,
and data processing settings.
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Configuration
DATA_FOLDER = "/home/zhou.aa/data/"  # Relative path to dataset folder containing .mat files
BLOCK_SIZE = 15  # Size of extracted blocks (15x15)
# Refinement factor: structure grid (e.g. 120×120) → E-field grid (e.g. 2883×2883). 2883/120 ≈ 24.
MESH_REFINEMENT_FACTOR = 24
NUM_BLOCKS_PER_METASURFACE = 1550  # ~17k total samples with 11 files (matches reference: 14k train + 3k val)
SEED = 42  # RNG seed for reproducible dataset sampling (matches reference)

# Model Configuration
INPUT_DIM = 5  # Number of node features: [R, H, D_x, D_y, B] (matches reference)
GAT_HIDDEN = 200  # Hidden dimension for GAT layer
GAT_HEADS = 8  # Number of attention heads
GCN_HIDDEN = 1600  # Hidden dimension for GCN layers
CNN_HIDDEN = 64  # Hidden dimension for CNN layers
OUTPUT_CHANNELS = 6  # Output channels: [Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag]

# Training Configuration (aligned with reference train_base_model.py)
EPOCHS = 1  # Train 1 epoch per run; use RESUME_FROM_CHECKPOINT to continue from last save
RESUME_FROM_CHECKPOINT = True  # Load existing weights before training (for incremental epoch-by-epoch runs)
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
TRAIN_SPLIT = 0.7  # 70% train
VAL_SPLIT = 0.15   # 15% validation
TEST_SPLIT = 0.15  # 15% test (held out)
LR_SCHEDULER_FACTOR = 0.8
LR_SCHEDULER_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 4  # Number of data loading workers (0 = main process only; 4+ recommended for GPU)
PIN_MEMORY = True  # Faster CPU→GPU transfer when using CUDA
USE_AMP = True  # Mixed precision (fp16) for faster GPU training

# Device Configuration
DEVICE = 'cuda'  # 'cuda' or 'cpu' (will auto-detect if 'cuda' is set)

# Output Configuration (absolute paths so outputs always go to project root)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints")
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures")
PRINT_FREQ = 10  # Print training progress every N epochs
SAVE_PLOT = True  # Whether to save training loss plot
SAVE_VISUALIZATIONS = True  # Whether to save E-field prediction visualizations