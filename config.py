"""
Configuration file for GAT-Net training and model parameters.

Modify these values to adjust model architecture, training hyperparameters,
and data processing settings.
"""

# Data Configuration
DATA_FOLDER = "data"  # Relative path to dataset folder containing .mat files
BLOCK_SIZE = 15  # Size of extracted blocks (15x15)
# Refinement factor: structure grid (e.g. 120×120) → E-field grid (e.g. 2883×2883). 2883/120 ≈ 24.
MESH_REFINEMENT_FACTOR = 24
NUM_BLOCKS_PER_METASURFACE = 100  # Number of random blocks per .mat file
SEED = None  # Optional RNG seed for reproducible dataset sampling (e.g. 42)

# Model Configuration
INPUT_DIM = 7  # Number of node features: [R, H, D_x, D_y, B, X, Y]
GAT_HIDDEN = 200  # Hidden dimension for GAT layer
GAT_HEADS = 8  # Number of attention heads
GCN_HIDDEN = 1600  # Hidden dimension for GCN layers
CNN_HIDDEN = 64  # Hidden dimension for CNN layers
OUTPUT_CHANNELS = 6  # Output channels: [Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag]

# Training Configuration
EPOCHS = 100
LEARNING_RATE = 5e-6
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
NUM_WORKERS = 0  # Number of data loading workers

# Device Configuration
DEVICE = 'cuda'  # 'cuda' or 'cpu' (will auto-detect if 'cuda' is set)

# Output Configuration
OUTPUT_DIR = "outputs"  # Root directory for all outputs
CHECKPOINT_DIR = "outputs/checkpoints"  # Saved model weights (.pth)
FIGURES_DIR = "outputs/figures"  # Loss plots and E-field visualizations (.png)
PRINT_FREQ = 10  # Print training progress every N epochs
SAVE_PLOT = True  # Whether to save training loss plot
SAVE_VISUALIZATIONS = True  # Whether to save E-field prediction visualizations